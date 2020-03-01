# -*- coding: utf-8 -*-
"""Console script for ultrayolo."""
from argparse import ArgumentParser
import imgaug.augmenters as iaa
from ultrayolo import datasets, YoloV3, YoloV3Tiny, BaseModel
from ultrayolo import helpers
from pathlib import Path
from omegaconf import OmegaConf
from typing import Tuple
import numpy as np

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger('ultrayolo')

Image = Tuple[int, int, int]


def load_or_compute_anchors(mode: str, number: int, path: str, ds_mode: str,
                            ds_train_path: str,
                            image_shape: Tuple) -> np.ndarray:
    """load or compute the anchors given a dataset

    Arguments:
        mode {str} -- siniglefile, multifile, coco
        number {int} -- the number of anchors
        path {str} -- the path to the anchors
        ds_mode {str} -- the mode of the dataset
        ds_train_path {str} -- the path to the dataset
        image_shape {tuple} -- the shape of the image
    Returns:
        np.ndarray -- the anchors for the algorithm
    """
    if mode == 'compute':
        boxes_xywh = datasets.prepare_data(ds_train_path, image_shape, ds_mode)
        anchors = datasets.gen_anchors(boxes_xywh, number)
    elif mode == 'default':
        anchors = YoloV3.default_anchors
    elif mode == 'default_tiny':
        anchors = YoloV3Tiny.default_anchors
    else:
        anchors = datasets.load_anchors(path)
    return anchors


def make_augmentations(percentage: float = 0.2) -> iaa.Sequential:
    """apply data augmentations to the dataset

    Keyword Arguments:
        percentage {float} -- the percentage of the dataset to augment for each epoch (default: 0.2)

    Returns:
        [iaa.Sequential] -- a pipeline of transformations
    """

    pipeline = iaa.Sequential(
        [
            iaa.Crop(percent=(0, percentage)),    # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
            iaa.Sometimes(percentage, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(percentage, iaa.Grayscale(alpha=(0.0, 1.0))),
    # Strengthen or weaken the contrast in each image.
            iaa.Sometimes(percentage, iaa.LinearContrast((0.75, 1.5))),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
            iaa.Sometimes(
                percentage,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
            iaa.Sometimes(percentage, iaa.Multiply(
                (0.8, 1.2), per_channel=0.2)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Sometimes(
                percentage,
                iaa.Affine(scale={
                    "x": (0.8, 1.5),
                    "y": (0.8, 1.5)
                },
                           translate_percent={
                               "x": (-0.3, 0.3),
                               "y": (-0.3, 0.3)
                           },
                           rotate=(-30, 30),
                           shear=(-10, 10))),
        ],
        random_order=True)
    return pipeline


def load_datasets(
        mode: str, image_shape: Image, max_objects: int, batch_size: int,
        base_grid_size: int, anchors: np.ndarray, train_path: str,
        val_path: str, augment: bool, pad_to_fixed_size: bool,
        **kwargs) -> Tuple[datasets.BaseDataset, datasets.BaseDataset]:
    """Load a dataset given the configurations

    Arguments:
        mode {str} -- a value in singlefile, multifile, coco. Repreents the format of the dataset
        image_shape {Image} -- the shape of the image
        max_objects {int} -- [description]
        batch_size {int} -- the size of the batch
        base_grid_size {int} -- the base size of the grid
        anchors {np.ndarray} -- the anchors
        train_path {str} -- the path to the training annotations
        val_path {str} -- the path to the validation annotations
        augment {bool} -- True of False to apply data augmentations
        pad_to_fixed_size {bool} -- True to pad the images to fixed size without distorting the image
    
    Returns:
        Tuple[datasets.BaseDataset, datasets.BaseDataset] -- a valid instance of the training and validation dataset
    """

    masks = datasets.make_masks(len(anchors))
    augmenters = make_augmentations() if augment else None

    if mode == 'multifile':
        train_dataset = datasets.YoloDatasetMultiFile
        val_dataset = datasets.YoloDatasetMultiFile
    elif mode == 'singlefile':
        train_dataset = datasets.YoloDatasetSingleFile
        val_dataset = datasets.YoloDatasetSingleFile
    elif mode == 'coco':
        train_dataset = datasets.CocoFormatDataset
        val_dataset = datasets.CocoFormatDataset

    train_dataset = train_dataset(train_path, image_shape, max_objects,
                                  batch_size, anchors, masks, base_grid_size,
                                  True, augmenters, pad_to_fixed_size)
    val_dataset = val_dataset(val_path, image_shape, max_objects, batch_size,
                              anchors, masks, base_grid_size, True, None,
                              pad_to_fixed_size)
    return train_dataset, val_dataset


def load_model(masks: np.ndarray, dataset: datasets.BaseDataset, iou: float,
               backbone: str, **kwargs) -> BaseModel:
    """load the model for the training
    
    Arguments:
        masks {np.ndarray} -- the mask used
        dataset {datasets.BaseDataset} -- the dataset used to train the model
        iou {float} -- the value of the intersection over union
        backbone {str} -- the backbone used for the model
    
    Returns:
        [BaseModel] -- an instance of the Yolo model
    """
    if len(masks) == 6:
        logger.info('loading tiny model')
        model = YoloV3Tiny(img_shape=dataset.target_shape,
                           max_objects=dataset.max_objects,
                           iou_threshold=iou,
                           score_threshold=None,
                           anchors=dataset.anchors,
                           num_classes=dataset.num_classes,
                           training=True)
    else:
        logger.info('loading large model')
        model = YoloV3(img_shape=dataset.target_shape,
                       max_objects=dataset.max_objects,
                       iou_threshold=iou,
                       score_threshold=None,
                       anchors=dataset.anchors,
                       num_classes=dataset.num_classes,
                       training=True,
                       backbone=backbone)
    return model


def run(dataset, model, fit, **kwargs):
    """the main to train the algorithm

    Arguments:
        config {object} -- the configurations to run the algorithm
    """

    if not dataset.train_path:
        raise Exception('missing checkpoints path')

    checkpoints_path = Path(dataset.train_path).parent / 'checkpoints'
    logger.info('saving checkpoints %s', str(checkpoints_path.absolute()))
    model_run_path = helpers.create_run_path(checkpoints_path)
    model_run_path.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(
        OmegaConf.create({
            'dataset': dataset,
            'model': model,
            'fit': fit
        }), str(model_run_path / 'run_config.yaml'))

    anchors = load_or_compute_anchors(ds_train_path=dataset.train_path,
                                      ds_mode=dataset.mode,
                                      image_shape=dataset.image_shape,
                                      **dataset.object_anchors)

    train_dataset, val_dataset = load_datasets(**dataset, anchors=anchors)
    anchors = train_dataset.anchors
    masks = train_dataset.anchor_masks

    # save classes
    with open(model_run_path / 'classes.txt', 'w') as fp:
        for _, name in train_dataset.classes:
            fp.write(name + '\n')

    # save configurations, anchors
    with open(model_run_path / 'anchors.txt', 'w') as fp:
        fp.write(datasets.anchors_to_string(anchors))

    yolo_model = load_model(masks, train_dataset, **model)

    # check the grid size of the dataset and the model are the sames
    out_test = yolo_model(train_dataset[0][0])
    for out, exp in zip(out_test, train_dataset[0][1]):
        assert out.shape == exp.shape

    if ('reload_weights' in model) and model['reload_weights']:
        logger.info('reload weigths at path %s', model['reload_weights'])
        yolo_model.load_weights(model['reload_weights'])

    logger.debug('using loss {}'.format(model.loss))
    loss = yolo_model.get_loss_function(num_batches=len(train_dataset),
                                        loss_name=model.loss)

    optimizer = yolo_model.get_optimizer(fit.optimizer.name,
                                         fit.optimizer.lrate.value)

    callbacks = helpers.default_callbacks(yolo_model, model_run_path,
                                          fit.optimizer.lrate.mode,
                                          fit.optimizer.lrate.value)

    if fit.mode == 'train':
        logger.info('training the model for %d epochs', fit.epochs.train)
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        yolo_model.fit(train_dataset, val_dataset, fit.epochs.train, 0,
                       callbacks, 1)

    elif fit.mode == 'transfer':
        yolo_model.set_mode_transfer()
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        logger.info('transfer the model for %d epochs', fit.epochs.transfer)

        #double the batch size
        train_dataset.batch_size *= 2
        val_dataset.batch_size *= 2
        yolo_model.fit(train_dataset, val_dataset, fit.epochs.transfer, 0,
                       callbacks, 1)
        train_dataset.batch_size /= 2
        val_dataset.batch_size /= 2

    elif fit.mode == 'finetuning':
        yolo_model.set_mode_transfer()
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        logger.info('transfer the model for %d epochs', fit.epochs.transfer)
        yolo_model.fit(train_dataset, val_dataset, fit.epochs.transfer, 0,
                       callbacks, 1)

        finetuning_epochs = fit.epochs.transfer + fit.epochs.finetuning
        yolo_model.set_mode_fine_tuning(fit.freezed_layers)
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        logger.info('fine tuning the model for %d epochs',
                    fit.epochs.finetuning)
        yolo_model.fit(train_dataset, val_dataset, finetuning_epochs,
                       fit.epochs.transfer, callbacks, 1)

    logging.info('saving final model')
    yolo_model.save(model_run_path / 'final_model.h5')


def main():
    parser = ArgumentParser('train extra yolo')
    parser.add_argument('--config',
                        required=True,
                        help='the path of the yaml config file')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    run(**config)


if __name__ == '__main__':
    main()
