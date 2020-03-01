#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `ultrayolo` package."""

import pytest
from ultrayolo import YoloV3, YoloV3Tiny
from pathlib import Path
from ultrayolo.datasets import YoloDatasetMultiFile, common
from ultrayolo import losses
import numpy as np
import tensorflow as tf

BASE_PATH = Path(__file__).parent / 'data'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotations'


@pytest.fixture()
def test_anchors():
    x = np.arange(5, 46, 5)
    anchors = np.array(list(zip(x, x)))
    anchors[:, 1] += np.random.randint(0, 10, 9)
    return anchors.astype(np.float32)


@pytest.fixture()
def test_masks():
    return YoloV3.default_masks


@pytest.fixture()
def test_classes():
    classes = common.load_classes(BASE_PATH / 'classes.txt')
    return classes


@pytest.fixture()
def test_dataset(test_anchors, test_masks, test_classes):
    ds = YoloDatasetMultiFile(annotations_path=BASE_PATH / 'manifest.txt',
                              img_shape=(256, 256, 3),
                              max_objects=10,
                              batch_size=2,
                              anchors=test_anchors,
                              anchor_masks=test_masks)
    return ds


@pytest.fixture()
def test_dataset_grid256_64(test_anchors, test_masks, test_classes):
    ds = YoloDatasetMultiFile(annotations_path=BASE_PATH / 'manifest.txt',
                              img_shape=(256, 256, 3),
                              max_objects=10,
                              batch_size=2,
                              anchors=test_anchors,
                              anchor_masks=test_masks,
                              base_grid_size=64)
    return ds


@pytest.mark.travis
def test_model_darknet(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='DarkNet',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


@pytest.mark.travis
def test_model_darknet_grid256_64(test_dataset_grid256_64, test_anchors,
                                  test_masks, test_classes):
    img_shape = test_dataset_grid256_64.target_shape
    model = YoloV3(img_shape,
                   test_dataset_grid256_64.max_objects,
                   backbone='DarkNet',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True,
                   base_grid_size=test_dataset_grid256_64.base_grid_size)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset_grid256_64))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset_grid256_64[0][0])
    assert res is not None


@pytest.mark.travis
def test_model_resnet50(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='ResNet50V2',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


@pytest.fixture()
def test_dataset_grid640_128(test_anchors, test_masks, test_classes):
    ds = YoloDatasetMultiFile(annotations_path=BASE_PATH / 'manifest.txt',
                              img_shape=(640, 640, 3),
                              max_objects=10,
                              batch_size=2,
                              anchors=test_anchors,
                              anchor_masks=test_masks,
                              base_grid_size=128)
    return ds


@pytest.mark.travis
def test_model_resnet50_grid640_128(test_dataset_grid640_128, test_anchors,
                                    test_masks, test_classes):
    img_shape = test_dataset_grid640_128.target_shape
    model = YoloV3(img_shape,
                   test_dataset_grid640_128.max_objects,
                   backbone='ResNet50V2',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True,
                   base_grid_size=test_dataset_grid640_128.base_grid_size)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset_grid640_128))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    model.summary()
    res = model(test_dataset_grid640_128[0][0])
    assert res is not None


def test_model_resnet101(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='ResNet101V2',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


def test_model_resnet152(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='ResNet152V2',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


@pytest.fixture()
def test_dataset_grid608_64(test_anchors, test_masks, test_classes):
    ds = YoloDatasetMultiFile(annotations_path=BASE_PATH / 'manifest.txt',
                              img_shape=(608, 608, 3),
                              max_objects=10,
                              batch_size=2,
                              anchors=test_anchors,
                              anchor_masks=test_masks,
                              base_grid_size=64)
    return ds


def test_model_DenseNet121_grid608_64(test_dataset_grid608_64, test_anchors,
                                      test_masks, test_classes):
    img_shape = test_dataset_grid608_64.target_shape
    model = YoloV3(img_shape,
                   test_dataset_grid608_64.max_objects,
                   backbone='DenseNet121',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset_grid608_64))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset_grid608_64[0][0])
    assert res is not None


def test_model_DenseNet121(test_dataset, test_anchors, test_masks,
                           test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='DenseNet121',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


def test_model_DenseNet169(test_dataset, test_anchors, test_masks,
                           test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='DenseNet169',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


def test_model_DenseNet201(test_dataset, test_anchors, test_masks,
                           test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='DenseNet201',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


@pytest.mark.travis
def test_model_MobileNetV2(test_dataset, test_anchors, test_masks,
                           test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='MobileNetV2',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    res = model(test_dataset[0][0])
    assert res is not None


@pytest.mark.travis
def test_reload_model(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='MobileNetV2',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=True)

    loss_fn = losses.make_loss(model.num_classes, test_anchors, test_masks,
                               img_shape[0], len(test_dataset))
    optimizer = model.get_optimizer('adam', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    model.fit(test_dataset, test_dataset, 1)

    save_path = BASE_PATH.parent / 'model.h5'
    model.save(save_path)

    del model

    model = YoloV3(img_shape,
                   test_dataset.max_objects,
                   backbone='MobileNetV2',
                   anchors=test_anchors,
                   num_classes=len(test_classes),
                   training=False)
    model.load_weights(save_path)

    _, _, _, valid_detections = model.predict(tf.zeros((1, *img_shape)))
    assert len(valid_detections) > 0
