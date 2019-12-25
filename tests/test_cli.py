from tfyolo3 import cli, YoloV3, YoloV3Tiny
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


def test_load_anchors_default(test_config):
    test_config.dataset.anchors.mode = 'default'
    anchors = cli.load_anchors(test_config.dataset)
    assert len(anchors) == 9
    assert np.all(anchors == YoloV3.default_anchors)

def test_load_anchors_default_tiny(test_config):
    test_config.dataset.anchors.mode = 'default_tiny'
    anchors = cli.load_anchors(test_config.dataset)
    assert len(anchors) == 6
    assert np.all(anchors == YoloV3Tiny.default_anchors)

def test_load_anchors_compute(test_config):
    test_config.dataset.anchors.mode = 'compute'
    test_config.dataset.anchors.number = 2
    anchors = cli.load_anchors(test_config.dataset)
    assert len(anchors) == 2

def test_load_anchors_from_file(test_config):
    test_config.dataset.anchors.mode = ''
    anchors = cli.load_anchors(test_config.dataset)
    assert len(anchors) == 9
    assert np.all(anchors == YoloV3.default_anchors)

def test_load_datasets(test_config):
    train_ds, val_ds = cli.load_datasets(test_config.dataset)
    assert train_ds is not None
    assert val_ds is not None

def test_to_tuple():
    value = '(256,256,3)'
    assert isinstance(cli.to_tuple(value), tuple)

def test_no_tuple():
    value = '(256,256,)'
    with pytest.raises(ValueError):
        assert isinstance(cli.to_tuple(value), tuple)