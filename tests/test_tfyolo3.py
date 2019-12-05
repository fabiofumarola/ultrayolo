#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `tfyolo3` package."""

import pytest
from tfyolo3 import YoloV3, YoloV3Tiny
from pathlib import Path
from tfyolo3.dataloaders import YoloDatasetMultiFile, common
from tfyolo3 import losses
import numpy as np


BASE_PATH = Path(__file__).parent / 'data'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotations'


@pytest.fixture()
def test_anchors():
    x = np.arange(5, 46, 5)
    anchors = np.array(list(zip(x, x)))
    anchors[:, 1] += np.random.randint(0, 10, 9)
    return anchors


@pytest.fixture()
def test_masks():
    return YoloV3.default_masks


@pytest.fixture()
def test_classes():
    classes = common.load_classes(BASE_PATH / 'classes.txt')
    return classes


@pytest.fixture()
def test_dataset(test_anchors, test_masks, test_classes):
    annotations_filepath = BASE_PATH / 'manifest.txt'
    ds = YoloDatasetMultiFile(annotations_filepath, (256, 256, 3), 10,
                              2, test_anchors, test_masks, len(test_classes))
    return ds


def test_model_darknet(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='DarkNet',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None


def test_model_resnet50(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='ResNet50V2',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None


def test_model_resnet101(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='ResNet101V2',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None


def test_model_resnet152(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='ResNet152V2',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None


def test_model_DenseNet121(test_dataset, test_anchors,
                           test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='DenseNet121',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None


def test_model_DenseNet169(test_dataset, test_anchors,
                           test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='DenseNet169',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None


def test_model_DenseNet201(test_dataset, test_anchors,
                           test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='DenseNet201',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None


def test_model_MobileNetV2(test_dataset, test_anchors,
                           test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects, backbone='MobileNetV2',
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = losses.make_loss(
        model.num_classes,
        test_anchors,
        test_masks,
        img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=True)
    history = model.fit(test_dataset, test_dataset, 1)
    assert history is not None
