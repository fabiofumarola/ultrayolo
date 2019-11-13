# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from .layers.core import (
    DarknetBody, ResNetBody, DenseNetBody,
    MobileNetBody, DarknetConv, YoloHead,
    YoloOutput, DarknetBodyTiny
)
from .losses import process_predictions, non_max_suppression, Loss
from .helpers import darknet
import multiprocessing

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class BaseModel(object):

    def __init__(self, img_shape=(None, None, 3), max_objects=100,
                 iou_threshold=0.7, score_threshold=0.7,
                 anchors=None, num_classes=80, training=False):
        self.img_shape = img_shape
        self.max_objects = max_objects
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.anchors = None
        self.anchors_scaled = None
        self.masks = None
        self.model = None
        self.tiny = None
        self.loss_function = None

    def load_weights(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        print('loaded checkopoint from', str(path.absolute()))
        if path.name.split('.')[-1] == 'weights':
            darknet.load_darknet_weights(self.model, path, self.tiny)
        elif path.name.split('.')[-1] == 'h5':
            self.model.load_weights(str(path.absolute()))

    def get_loss_function(self):
        """singleton for the yolo loss function

        Returns:
            list -- a list of the loss function for each mask
        """
        if self.loss_function is None:
            self.loss_function = [
                Loss(
                    self.num_classes, self.anchors[mask], self.img_shape[0],
                    ignore_iou_threshold=self.iou_threshold
                ) for mask in self.masks
            ]

        return self.loss_function

    def get_optimizer(self, optimizer_name, lrate):
        logging.info('using %s optimize', optimizer_name)
        if optimizer_name == 'adam':
            return Adam(learning_rate=lrate, clipvalue=1)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=lrate, clipvalue=1)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=lrate, momentum=0.95,
                       nesterov=True, clipvalue=1)
        else:
            raise Exception(f'not valid optimizer {optimizer_name}')

    def set_mode_train(self):
        darknet.unfreeze(self.model)

    def set_mode_transfer(self):
        logging.info('freeze backbone')
        darknet.freeze_backbone(self.model)

    def set_mode_fine_tuning(self, num_layers_to_train):
        darknet.unfreeze(self.model)
        darknet.freeze_backbone_layers(self.model, num_layers_to_train)

    def compile(self, optimizer, loss, run_eagerly):
        self.model.compile(optimizer, loss, run_eagerly=run_eagerly)
        self.model.summary()

    def fit(self, train_dataset, val_dataset, epochs, callbacks=None,
            run_eagerly=False, workers=1, max_queue_size=64, initial_epoch=0):

        logging.info('training for %d epochs on the dataset %s',
                     train_dataset.base_path, epochs)
        if workers == -1:
            workers = multiprocessing.cpu_count()

        if workers > 1:
            use_multiprocessing = True

        self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
                       callbacks=callbacks, workers=workers, use_multiprocessing=use_multiprocessing,
                       max_queue_size=64, initial_epoch=initial_epoch)

    def save(self, path, save_format='h5'):
        """save the model to the given path

        Arguments:
            path {str|pathlib.Path} -- the path to save the checkpoint
        """
        path = str(Path(path).absolute())
        self.model.save(path, save_format=save_format)


class YoloV3(BaseModel):

    default_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    default_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                (59, 119), (116, 90), (156, 198), (373, 326)],
                               np.float32)

    def __init__(self, img_shape=(None, None, 3), max_objects=100,
                 iou_threshold=0.7, score_threshold=0.7,
                 anchors=None, num_classes=80, training=False, backbone='DarkNet'):
        super().__init__(img_shape, max_objects, iou_threshold, score_threshold,
                         anchors, num_classes)

        self.masks = self.default_masks
        if anchors is None:
            self.anchors = self.default_anchors

        self.anchors = self.anchors.astype(np.float32)
        self.anchors_scaled = self.anchors / img_shape[1]
        self.training = training
        self.backbone = backbone
        self.tiny = False

        x = inputs = Input(shape=img_shape)
        if backbone == 'DarkNet':
            x36, x61, x = DarknetBody(name=backbone)(x)
        elif 'ResNet' in backbone:
            x36, x61, x = ResNetBody(img_shape, version=backbone)(x)
        elif 'DenseNet' in backbone:
            x36, x61, x = DenseNetBody(img_shape, version=backbone)(x)
        elif 'MobileNet' in backbone:
            x36, x61, x = MobileNetBody(img_shape, version=backbone)(x)

        x = YoloHead(x, 512, name='yolo_head_0')
        output0 = YoloOutput(x, 512, len(
            self.masks[0]), num_classes, name='yolo_output_0')

        x = YoloHead((x, x61), 256, name='yolo_head_1')
        output1 = YoloOutput(x, 256, len(
            self.masks[1]), num_classes, name='yolo_output_1')

        x = YoloHead((x, x36), 128, name='yolo_head_2')
        output2 = YoloOutput(x, 128, len(
            self.masks[2]), num_classes, name='yolo_output_2')

        if training:
            self.model = Model(
                inputs, [output0, output1, output2], name='yolov3')
        else:
            boxes0 = Lambda(
                lambda x: process_predictions(
                    x, self.num_classes, self.anchors_scaled[self.masks[0]]),
                name='yolo_boxes_0'
            )(output0)

            boxes1 = Lambda(
                lambda x: process_predictions(
                    x, self.num_classes, self.anchors_scaled[self.masks[1]]),
                name='yolo_boxes_1'
            )(output1)

            boxes2 = Lambda(
                lambda x: process_predictions(
                    x, self.num_classes, self.anchors_scaled[self.masks[2]]),
                name='yolo_boxes_2'
            )(output2)

            outputs = Lambda(lambda x: non_max_suppression(
                x, self.anchors_scaled, self.masks, self.num_classes, self.iou_threshold, self.score_threshold, self.max_objects, self.img_shape[
                    0]
            ), name='yolo_nms')((boxes0[:3], boxes1[:3], boxes2[:3]))

            self.model = Model(inputs, outputs, name='yolov3')


class YoloV3Tiny(BaseModel):

    default_anchors = np.array([(10, 14), (23, 27), (37, 58),
                                (81, 82), (135, 169), (344, 319)],
                               np.float32)

    default_masks = np.array([[3, 4, 5], [0, 1, 2]])

    def __init__(self, img_shape=(None, None, 3), max_objects=100,
                 iou_threshold=0.7, score_threshold=0.7,
                 anchors=None, num_classes=80, training=False):
        super().__init__(img_shape, max_objects, iou_threshold, score_threshold,
                         anchors, num_classes)

        self.masks = self.default_masks
        if anchors is None:
            self.anchors = self.default_anchors
        self.anchors = self.anchors.astype(np.float32)
        self.anchors_scaled = self.anchors / img_shape[1]
        self.training = training
        self.tiny = True

        x = inputs = Input(shape=img_shape)
        x8, x = DarknetBodyTiny(name='DarkNet')(x)

        x = YoloHead(x, 256, name='yolo_head_0', is_tiny=True)
        output0 = YoloOutput(x, 256, len(
            self.masks[0]), self.num_classes, name='yolo_output_0')

        x = YoloHead((x, x8), 128, name='yolo_head_1', is_tiny=True)
        output1 = YoloOutput(x, 128, len(
            self.masks[1]), self.num_classes, name='yolo_output_1')

        if training:
            self.model = Model(inputs, [output0, output1], name='yolov3')
        else:
            boxes0 = Lambda(
                lambda x: process_predictions(
                    x, self.num_classes, self.anchors_scaled[self.masks[0]]),
                name='yolo_boxes_0'
            )(output0)

            boxes1 = Lambda(
                lambda x: process_predictions(
                    x, self.num_classes, self.anchors_scaled[self.masks[1]]),
                name='yolo_boxes_1'
            )(output1)

            outputs = Lambda(lambda x: non_max_suppression(
                x, self.anchors_scaled, self.masks, self.num_classes, self.iou_threshold, self.score_threshold, self.max_objects, self.img_shape[
                    0]
            ), name='yolo_nms')((boxes0[:3], boxes1[:3]))

            self.model = Model(inputs, outputs, name='yolov3_tiny')
