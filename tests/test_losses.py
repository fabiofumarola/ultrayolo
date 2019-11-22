import pytest
from pathlib import Path
from tfyolo3.losses import Loss
from tfyolo3.dataloaders import YoloDatasetMultiFile, common
from tfyolo3 import YoloV3
import numpy as np
from pytest import approx

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


def test_loss_initialized_yolo(
        test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects,
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = Loss(len(test_classes), test_anchors, test_masks, img_shape[0])
    x_true, y_true_grids = test_dataset[0]
    y_pred_grids = model(x_true)

    for i, loss in enumerate(loss_fn):
        loss_value = loss(y_true_grids[i], y_pred_grids[i])
        assert np.all(loss_value > 0)
        print('loss', i, loss_value)


def test_loss_yolo(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    loss_fn = Loss(len(test_classes), test_anchors, test_masks, img_shape[0])
    _, y_true_grids = test_dataset[0]
    y_pred_grids = y_true_grids

    for i, loss in enumerate(loss_fn):
        loss_value = loss(y_true_grids[i], y_pred_grids[i])
        assert np.all(loss_value >= 0)
        print('loss', i, loss_value)


def test_compare_losses(test_dataset, test_anchors, test_masks, test_classes):
    img_shape = test_dataset.target_shape
    model = YoloV3(img_shape, test_dataset.max_objects,
                   anchors=test_anchors, num_classes=len(test_classes), training=True)

    loss_fn = Loss(len(test_classes), test_anchors, test_masks, img_shape[0])
    x_true, y_true_grids = test_dataset[0]
    y_pred_grids = model(x_true)
    y_pred_grids_true = y_true_grids

    for i, loss in enumerate(loss_fn):
        loss_value = loss(y_true_grids[i], y_pred_grids[i])
        loss_value_true = loss(y_true_grids[i], y_pred_grids_true[i])
        assert np.all(loss_value_true <= loss_value)
