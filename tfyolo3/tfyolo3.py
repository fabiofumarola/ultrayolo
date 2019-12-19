# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from .layers.core import (
    DarknetBody, ResNetBody, DenseNetBody,
    MobileNetBody, YoloHead,
    YoloOutput, DarknetBodyTiny
)

from .losses import process_predictions, non_max_suppression, make_loss
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

    def summary(self):
        """return the tensorflow model summary

        """
        self.model.summary()

    def load_weights(self, path, backbone):
        """load:
            * saved checkpoints in h5 format
            * saved weights in darknet format

        Arguments:
            path {str} -- the path where the weights are saved
        """
        if not isinstance(path, Path):
            path = Path(path)

        print('loaded checkopoint from', str(path.absolute()))
        if path.name.split('.')[-1] == 'weights' and backbone == 'DarkNet':
            darknet.load_darknet_weights(self.model, path, self.tiny)
        elif path.name.split('.')[-1] == 'h5':
            self.model.load_weights(str(path.absolute()))

    def get_loss_function(self):
        """singleton for the yolo loss function

        Returns:
            list -- a list of the loss function for each mask
        """
        if self.loss_function is None:
            self.loss_function = make_loss(self.num_classes, self.anchors,
                                           self.masks, self.img_shape[0])

        return self.loss_function

    def get_optimizer(self, optimizer_name, lrate):
        """helper to create the optimizer using the class defined members

        Arguments:
            optimizer_name {str} -- the name of the optimizer to use: (values: adam, rmsprop, sgd)
            lrate {float} -- a valid starting value for the learning rate

        Raises:
            Exception: raise an exception if the optimizer is not supported

        Returns:
            tensorflow.keras.optimizer -- an instance of the selected optimizer
        """
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
        """unfreeze all the net read for a full training
        """
        darknet.unfreeze(self.model)

    def set_mode_transfer(self):
        """freeze the backbone of the network, check that the head and output layers are unfreezed
        """
        logging.info('freeze backbone')
        darknet.freeze_backbone(self.model)

    def set_mode_fine_tuning(self, num_layers_to_train):
        """unfreeze the backbone and freeze the first `num_layers_to_train` layers

        Arguments:
            num_layers_to_train {[type]} -- [description]
        """
        darknet.unfreeze(self.model)
        darknet.freeze_backbone_layers(self.model, num_layers_to_train)

    def compile(self, optimizer, loss, run_eagerly, summary=True):
        """compile the model

        Arguments:
            optimizer {tf.keras.optimizers} -- a valid tensorflow optimizer
            loss {tfyolo3.losses.Loss} -- the loss function for yolo
            run_eagerly {bool} -- if True is uses eager mode, that is you can see more explainable stack traces

        Keyword Arguments:
            summary {bool} -- if True print the summary of the model (default: {True})
        """
        self.model.compile(optimizer, loss, run_eagerly=run_eagerly)
        if summary:
            self.model.summary()

    def fit(self, train_dataset, val_dataset, epochs, initial_epoch=0,
            callbacks=None, workers=1, max_queue_size=64):
        """train the model

        Arguments:
            train_dataset {tfyolo3.dataloader.Dataset} -- an instance of the dataset
            val_dataset {tfyolo3.dataloader.Dataset} -- an instance of the dataset
            epochs {int} -- the number of epochs

        Keyword Arguments:
            initial_epoch {int} -- [description] (default: {0})
            callbacks {[type]} -- [description] (default: {None})
            workers {int} -- [description] (default: {1})
            max_queue_size {int} -- [description] (default: {64})

        Returns:
            [type] -- [description]
        """

        logging.info('training for %s epochs on the dataset %d',
                     train_dataset.base_path, epochs)
        if workers == -1:
            workers = multiprocessing.cpu_count()

        use_multiprocessing = False
        if workers > 1:
            use_multiprocessing = True

        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
                              callbacks=callbacks, workers=workers, use_multiprocessing=use_multiprocessing,
                              max_queue_size=64, initial_epoch=initial_epoch)

    def save(self, path, save_format='h5'):
        """save the model to the given path

        Arguments:
            path {str|pathlib.Path} -- the path to save the checkpoint
        """
        path = str(Path(path).absolute())
        self.model.save(path, save_format=save_format)

    def __call__(self, x):
        return self.model(x)


class YoloV3(BaseModel):

    default_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    default_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                (59, 119), (116, 90), (156, 198), (373, 326)],
                               np.float32)

    def __init__(self, img_shape=(None, None, 3), max_objects=100,
                 iou_threshold=0.7, score_threshold=0.7,
                 anchors=None, num_classes=80, training=False, backbone='DarkNet'):
        """The class that implement yolo v3

        Keyword Arguments:
            img_shape {tuple} -- the tuple (Height, Width, Channel) to represent the target image shape (default: {(None, None, 3)})
            max_objects {int} -- the maximum number of objects that can be detected (default: {100})
            iou_threshold {float} -- the intersection over union threshold used to filter out the multiple boxes for the same object (default: {0.7})
            score_threshold {float} -- the minimum confidence score for the output (default: {0.7})
            anchors {np.ndarray} -- the list of the anchors used for the detection (default: {None})
            num_classes {int} -- the number of classes (default: {80})
            training {bool} -- True if the model is used for training (default: {False})
            backbone {str} -- a valid backbone among the following: (default: {'DarkNet'})
                    * DarkNet
                    * 'ResNet50V2', 'ResNet101V2', 'ResNet152V2'
                    * 'DenseNet121', 'DenseNet169', 'DenseNet201'
                    * 'MobileNetV2'
        """
        super().__init__(img_shape, max_objects, iou_threshold, score_threshold,
                         anchors, num_classes)

        self.masks = self.default_masks
        if anchors is None:
            self.anchors = YoloV3.default_anchors.copy()
        else:
            self.anchors = anchors.astype(np.float32)
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
        """The class that implement yolo v3 tiny

        Keyword Arguments:
            img_shape {tuple} -- the tuple (Height, Width, Channel) to represent the target image shape (default: {(None, None, 3)})
            max_objects {int} -- the maximum number of objects that can be detected (default: {100})
            iou_threshold {float} -- the intersection over union threshold used to filter out the multiple boxes for the same object (default: {0.7})
            score_threshold {float} -- the minimum confidence score for the output (default: {0.7})
            anchors {np.ndarray} -- the list of the anchors used for the detection (default: {None})
            num_classes {int} -- the number of classes (default: {80})
            training {bool} -- True if the model is used for training (default: {False})
        """
        super().__init__(img_shape, max_objects, iou_threshold, score_threshold,
                         anchors, num_classes)

        self.masks = self.default_masks
        if anchors is None:
            self.anchors = YoloV3.default_anchors.copy()
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
