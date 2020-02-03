from ultrayolo import cli, YoloV3, YoloV3Tiny
import pytest
from pathlib import Path
import numpy as np

BASE_PATH = Path(__file__).parent / 'data'

@pytest.fixture()
def test_config():
    path = BASE_PATH / 'train.yml'
    config = cli.load_config(path)
    return config


def test_load_config(test_config):
    assert test_config is not None


def test_load_anchors_default():
    anchors = cli.load_anchors('default', 9, None, None, None)
    assert len(anchors) == 9
    assert np.all(anchors == YoloV3.default_anchors)


def test_load_anchors_default_tiny():
    anchors = cli.load_anchors('default_tiny', 6, None, None, None)
    assert len(anchors) == 6
    assert np.all(anchors == YoloV3Tiny.default_anchors)


def test_load_anchors_compute():
    anchors = cli.load_anchors(
        'compute',
        2,
        None,
        'multifile',
        './tests/data/manifest.txt')
    assert len(anchors) == 2


def test_load_anchors_from_file(test_config):
    anchors = cli.load_anchors(
        '', 9, './tests/data/yolov3_anchors.txt', None, None)
    assert len(anchors) == 9
    assert np.all(anchors == YoloV3.default_anchors)


def test_load_datasets(test_config):
    print(test_config['dataset'])
    train_ds, val_ds, _, _ = cli.load_datasets(**test_config['dataset'])
    assert train_ds is not None
    assert val_ds is not None
