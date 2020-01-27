# -*- coding: utf-8 -*-

"""Console script for tfyolo3."""
import yaml
from dotmap import DotMap
from argparse import ArgumentParser
import imgaug.augmenters as iaa
from tfyolo3 import datasets, YoloV3, YoloV3Tiny
from tfyolo3 import helpers
from pathlib import Path

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger('tfyolo3')

def load_config(path):
    """load config from yaml

    Arguments:
        path {str} -- the path of the yaml file

    Returns:
        object -- an object with the given configurations
    """
    with open(path, 'r') as fh:
        return DotMap(yaml.safe_load(fh))


def load_anchors(dataset_config):
    """load or compute the anchors given a dataset

    Arguments:
        dataset_config {object} -- the configuration of the dataset

    Returns:
        [type] -- [description]
    """
    ismultifile = True if dataset_config.mode == 'multifile' else False
    annotations_path = dataset_config.annotations.train
    num_anchors = dataset_config.anchors.number

    if dataset_config.anchors.mode == 'compute':
        anchors = datasets.gen_anchors(
            annotations_path, num_anchors, ismultifile)
    elif dataset_config.anchors.mode == 'default':
        anchors = YoloV3.default_anchors
    elif dataset_config.anchors.mode == 'default_tiny':
        anchors = YoloV3Tiny.default_anchors
    else:
        anchors = datasets.load_anchors(dataset_config.anchors.path)
    return anchors


def make_augmentations(max_number_augs=5):
    """apply data augmentations to the dataset

    Keyword Arguments:
        max_number_augs {int} -- the max number of augmentation to apply to each image (default: {5})

    Returns:
        [type] -- [description]
    """
    augmentation = iaa.SomeOf((0, max_number_augs), [
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.Affine(scale=(1., 2.5), rotate=(-90, 90), shear=(-16, 16),
                   translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.LinearContrast((0.5, 1.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0)),
        iaa.LogContrast(gain=(0.6, 1.4)),
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.Clouds(),
        iaa.Alpha(
            (0.0, 1.0),
            first=iaa.Add(100),
            second=iaa.Multiply(0.2)),
        iaa.MotionBlur(k=5),
        iaa.MultiplyHueAndSaturation((0.5, 1.0), per_channel=True),
        iaa.AddToSaturation((-50, 50)),
        iaa.Noop()
    ])
    return augmentation


def load_datasets(ds_conf):
    """load a dataset from configs

    Arguments:
        ds_conf {object} -- the configutations

    Returns:
        tuple -- train and test dataset tf.keras.Sequence objects
    """
    if isinstance(ds_conf.image_shape, str):
        ds_conf.image_shape = to_tuple(ds_conf.image_shape)
    else:
        ds_conf.image_shape = ds_conf.image_shape
    anchors = load_anchors(ds_conf)
    masks = datasets.make_masks(len(anchors))

    # FIXME make 4 a parameter
    augmenters = make_augmentations(4) if ds_conf.augment else None

    if ds_conf.mode == 'multifile':
        train_dataset = datasets.YoloDatasetMultiFile
        val_dataset = datasets.YoloDatasetMultiFile
    elif ds_conf.mode == 'singlefile':
        train_dataset = datasets.YoloDatasetSingleFile
        val_dataset = datasets.YoloDatasetSingleFile

    train_dataset = train_dataset(
        ds_conf.annotations.train, ds_conf.image_shape,
        ds_conf.max_objects, ds_conf.batch_size, anchors, masks, True,
        augmenters, ds_conf.pad_to_fixed_size
    )
    val_dataset = val_dataset(
        ds_conf.annotations.val, ds_conf.image_shape,
        ds_conf.max_objects, ds_conf.batch_size, anchors, masks, True,
        None, ds_conf.pad_to_fixed_size
    )
    return train_dataset, val_dataset


def to_tuple(value):
    values = value[1:-1].split(',')
    values = [int(v.strip()) for v in values]
    return tuple(values)


def main(config):
    """the main to train the algorithm

    Arguments:
        config {object} -- the configurations to run the algorithm
    """

    config.dataset.image_shape = to_tuple(config.dataset.image_shape)
    train_dataset, val_dataset = load_datasets(config.dataset)

    if len(train_dataset.anchor_masks) == 6:
        logger.info('loading tiny model')
        model = YoloV3Tiny(
            img_shape=config.dataset.image_shape,
            max_objects=config.dataset.max_objects,
            iou_threshold=config.model.thresholds.intersection_over_union,
            score_threshold=config.model.thresholds.object_score,
            anchors=train_dataset.anchors,
            num_classes=train_dataset.num_classes,
            training=True
        )
    else:
        logger.info('loading large model')
        model = YoloV3(
            img_shape=config.dataset.image_shape,
            max_objects=config.dataset.max_objects,
            iou_threshold=config.model.thresholds.intersection_over_union,
            score_threshold=config.model.thresholds.object_score,
            anchors=train_dataset.anchors,
            num_classes=train_dataset.num_classes,
            training=True,
            backbone=config.model.backbone
        )

    if len(config.model.reload.path):
        logger.info('reload weigths at path %s', config.model.reload.path)
        model.load_weights(config.model.reload.path, config.model.backbone)

    loss = model.get_loss_function()
    optimizer = model.get_optimizer(config.fit.optimizer.name,
                                    config.fit.optimizer.lrate.value)

    checkpoints_path = Path(config.model.checkpoints.path)
    checkpoints_path.mkdir(exist_ok=True)
    logger.info('saving checkpoints %s', str(checkpoints_path.absolute()))
    model_run_path = helpers.create_run_path(checkpoints_path)

    callbacks = helpers.default_callbacks(
        model, model_run_path, config.fit.optimizer.lrate.mode,
        config.fit.optimizer.lrate.value
    )

    if config.fit.mode == 'train':
        logger.info('training the model for %d epochs', config.fit.epochs.train)
        model.compile(optimizer, loss, config.fit.run_eagerly)
        model.fit(train_dataset, val_dataset, config.fit.epochs.train,
                  0, callbacks, 1)

    elif config.fit.mode == 'transfer':
        model.set_mode_transfer()
        model.compile(optimizer, loss, config.fit.run_eagerly)
        logger.info(
            'transfer the model for %d epochs',
            config.fit.epochs.transfer)
        model.fit(train_dataset, val_dataset, config.fit.epochs.transfer,
                  0, callbacks, 1)

    elif config.fit.mode == 'finetuning':
        model.set_mode_transfer()
        model.compile(optimizer, loss, config.fit.run_eagerly)
        logger.info(
            'transfer the model for %d epochs',
            config.fit.epochs.transfer)
        model.fit(train_dataset, val_dataset, config.fit.epochs.transfer,
                  0, callbacks, 1)

        finetuning_epochs = config.fit.epochs.transfer + config.fit.epochs.finetuning
        model.set_mode_fine_tuning(config.fit.freezed_layers)
        model.compile(optimizer, loss, config.fit.run_eagerly)
        logger.info(
            'fine tuning the model for %d epochs',
            config.fit.epochs.finetuning)
        model.fit(train_dataset, val_dataset, finetuning_epochs,
                  config.fit.epochs.transfer, callbacks, 1)

    logging.info('saving final model')
    model.save(model_run_path / 'final_model.h5')


if __name__ == '__main__':
    parser = ArgumentParser('train extra yolo')
    parser.add_argument('--config', required=True,
                        help='the path of the yaml config file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
