# -*- coding: utf-8 -*-
"""Console script for ultrayolo."""
from argparse import ArgumentParser
import imgaug.augmenters as iaa
from ultrayolo import datasets, YoloV3, YoloV3Tiny
from ultrayolo import helpers
from pathlib import Path
from omegaconf import OmegaConf

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger('ultrayolo')


def load_anchors(mode, number, path, ds_mode, ds_train_path, image_shape):
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
    ismultifile = True if ds_mode == 'multifile' else False

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


def make_augmentations():
    """apply data augmentations to the dataset

    Keyword Arguments:
        max_number_augs {int} -- the max number of augmentation to apply to each image (default: {5})

    Returns:
        [type] -- [description]
    """
    pipeline = iaa.Sequential(
        [
            iaa.Crop(percent=(0, 0.2)),    # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.0, 1.0))),
    # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(scale={
                "x": (0.8, 1.5),
                "y": (0.8, 1.5)
            },
                       translate_percent={
                           "x": (-0.3, 0.3),
                           "y": (-0.3, 0.3)
                       },
                       rotate=(-30, 30),
                       shear=(-12, 12)),
        ],
        random_order=True)
    return pipeline


def load_datasets(mode, image_shape, anchors, train_path, val_path, augment,
                  max_objects, batch_size, pad_to_fixed_size):
    """load a dataset from configs

    Arguments:
        ds_conf {object} -- the configutations

    Returns:
        tuple -- train and test dataset tf.keras.Sequence objects
    """

    anchors = load_anchors(ds_mode=mode,
                           ds_train_path=train_path,
                           image_shape=image_shape,
                           **anchors)
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
                                  batch_size, anchors, masks, True, augmenters,
                                  pad_to_fixed_size)
    val_dataset = val_dataset(val_path, image_shape, max_objects, batch_size,
                              anchors, masks, True, None, pad_to_fixed_size)
    return train_dataset, val_dataset, anchors, masks


def load_model(num_masks, dataset, iou, object_score, backbone, **kwargs):
    if num_masks == 6:
        logger.info('loading tiny model')
        model = YoloV3Tiny(img_shape=dataset.target_shape,
                           max_objects=dataset.max_objects,
                           iou_threshold=iou,
                           score_threshold=object_score,
                           anchors=dataset.anchors,
                           num_classes=dataset.num_classes,
                           training=True)
    else:
        logger.info('loading large model')
        model = YoloV3(img_shape=dataset.target_shape,
                       max_objects=dataset.max_objects,
                       iou_threshold=iou,
                       score_threshold=object_score,
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
        }), str(model_run_path / 'run_config.yml'))

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
