# -*- coding: utf-8 -*-

"""Console script for ultrayolo."""
import yaml
from dotmap import DotMap
from argparse import ArgumentParser
import imgaug.augmenters as iaa
from ultrayolo import datasets, YoloV3, YoloV3Tiny
from ultrayolo import helpers
from pathlib import Path

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger('ultrayolo')


def load_config(path):
    """load config from yaml

    Arguments:
        path {str} -- the path of the yaml file

    Returns:
        object -- an object with the given configurations
    """
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)


def load_anchors(mode, number, path, ds_mode, ds_train_path):
    """load or compute the anchors given a dataset

    Arguments:
        mode {str} -- siniglefile, multifile, coco
        number {int} -- the number of anchors
        path {str} -- the path to the anchors
        ds_mode {str} -- the mode of the dataset
        ds_train_path {str} -- the path to the dataset
    Returns:
        np.ndarray -- the anchors for the algorithm
    """
    ismultifile = True if ds_mode == 'multifile' else False

    if mode == 'compute':
        anchors = datasets.gen_anchors(ds_train_path, number, ismultifile)
    elif mode == 'default':
        anchors = YoloV3.default_anchors
    elif mode == 'default_tiny':
        anchors = YoloV3Tiny.default_anchors
    else:
        anchors = datasets.load_anchors(path)
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


def load_datasets(mode, image_shape, anchors, train_path, val_path,
                  augment, max_objects, batch_size, pad_to_fixed_size):
    """load a dataset from configs

    Arguments:
        ds_conf {object} -- the configutations

    Returns:
        tuple -- train and test dataset tf.keras.Sequence objects
    """

    anchors = load_anchors(ds_mode=mode, ds_train_path=train_path, **anchors)
    masks = datasets.make_masks(len(anchors))

    # FIXME make 4 a parameter
    augmenters = make_augmentations(4) if augment else None

    if mode == 'multifile':
        train_dataset = datasets.YoloDatasetMultiFile
        val_dataset = datasets.YoloDatasetMultiFile
    elif mode == 'singlefile':
        train_dataset = datasets.YoloDatasetSingleFile
        val_dataset = datasets.YoloDatasetSingleFile
    elif mode == 'coco':
        train_dataset = datasets.CocoFormatDataset
        val_dataset = datasets.CocoFormatDataset

    train_dataset = train_dataset(
        train_path, image_shape,
        max_objects, batch_size, anchors, masks, True,
        augmenters, pad_to_fixed_size
    )
    val_dataset = val_dataset(
        val_path, image_shape,
        max_objects, batch_size, anchors, masks, True,
        None, pad_to_fixed_size
    )
    return train_dataset, val_dataset, anchors, masks


def load_model(num_masks, dataset, iou, object_score, backbone, **kwargs):
    if num_masks == 6:
        logger.info('loading tiny model')
        model = YoloV3Tiny(
            img_shape=dataset.target_shape,
            max_objects=dataset.max_objects,
            iou_threshold=iou,
            score_threshold=object_score,
            anchors=dataset.anchors,
            num_classes=dataset.num_classes,
            training=True
        )
    else:
        logger.info('loading large model')
        model = YoloV3(
            img_shape=dataset.target_shape,
            max_objects=dataset.max_objects,
            iou_threshold=iou,
            score_threshold=object_score,
            anchors=dataset.anchors,
            num_classes=dataset.num_classes,
            training=True,
            backbone=backbone
        )
    return model


def main(dataset, model, fit, **kwargs):
    """the main to train the algorithm

    Arguments:
        config {object} -- the configurations to run the algorithm
    """
    train_path = dataset['train_path']
    if not train_path:
        raise Exception('missing checkpoints path')

    checkpoints_path = Path(train_path).parent / 'checkpoints'
    logger.info('saving checkpoints %s', str(checkpoints_path.absolute()))
    model_run_path = helpers.create_run_path(checkpoints_path)
    model_run_path.mkdir(exist_ok=True, parents=True)

    train_dataset, val_dataset, anchors, masks = load_datasets(**dataset)

    # save classes 
    with open(model_run_path / 'classes.txt', 'w') as fp:
        for _, name in train_dataset.classes:
            fp.write(name + '\n')

    # save configurations, anchors
    with open(model_run_path / 'anchors.txt', 'w') as fp:
        fp.write(datasets.anchors_to_string(anchors))

    yolo_model = load_model(len(masks), train_dataset, **model)

    if ('reload_weights' in model) and model['reload_weights']:
        logger.info('reload weigths at path %s', model['reload_weights'])
        yolo_model.load_weights(model['reload_weights'])

    loss = yolo_model.get_loss_function()
    fit = DotMap(fit)
    optimizer = yolo_model.get_optimizer(
        fit.optimizer.name, fit.optimizer.lrate.value)

    callbacks = helpers.default_callbacks(
        yolo_model, model_run_path, fit.optimizer.lrate.mode,
        fit.optimizer.lrate.value
    )

    if fit.mode == 'train':
        logger.info('training the model for %d epochs', fit.epochs.train)
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        yolo_model.fit(train_dataset, val_dataset, fit.epochs.train,
                       0, callbacks, 1)

    elif fit.mode == 'transfer':
        yolo_model.set_mode_transfer()
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        logger.info(
            'transfer the model for %d epochs',
            fit.epochs.transfer)
        
        #double the batch size
        train_dataset.batch_size *= 2
        val_dataset.batch_size *= 2
        yolo_model.fit(train_dataset, val_dataset, fit.epochs.transfer,
                       0, callbacks, 1)
        train_dataset.batch_size /= 2
        val_dataset.batch_size /= 2

    elif fit.mode == 'finetuning':
        yolo_model.set_mode_transfer()
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        logger.info(
            'transfer the model for %d epochs',
            fit.epochs.transfer)
        yolo_model.fit(train_dataset, val_dataset, fit.epochs.transfer,
                       0, callbacks, 1)

        finetuning_epochs = fit.epochs.transfer + fit.epochs.finetuning
        yolo_model.set_mode_fine_tuning(fit.freezed_layers)
        yolo_model.compile(optimizer, loss, fit.run_eagerly)
        logger.info(
            'fine tuning the model for %d epochs',
            fit.epochs.finetuning)
        yolo_model.fit(train_dataset, val_dataset, finetuning_epochs,
                       fit.epochs.transfer, callbacks, 1)

    logging.info('saving final model')
    yolo_model.save(model_run_path / 'final_model.h5')


if __name__ == '__main__':
    parser = ArgumentParser('train extra yolo')
    parser.add_argument('--config', required=True,
                        help='the path of the yaml config file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(**config)
