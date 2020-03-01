from ultrayolo import cli, YoloV3, YoloV3Tiny
import pytest
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

BASE_PATH = Path(__file__).parent / 'data'


@pytest.fixture()
def test_config():
    path = str(BASE_PATH / 'train.yaml')
    config = OmegaConf.load(path)
    return config


@pytest.mark.travis
def test_load_config(test_config):
    assert test_config is not None


@pytest.mark.travis
def test_load_anchors_default():
    anchors = cli.load_or_compute_anchors('default', 9, None, None, None, None)
    assert len(anchors) == 9
    assert np.all(anchors == YoloV3.default_anchors)


@pytest.mark.travis
def test_load_anchors_default_tiny():
    anchors = cli.load_or_compute_anchors('default_tiny', 6, None, None, None,
                                          None)
    assert len(anchors) == 6
    assert np.all(anchors == YoloV3Tiny.default_anchors)


@pytest.mark.travis
def test_load_anchors_compute():
    anchors = cli.load_or_compute_anchors('compute', 2, None, 'multifile',
                                          './tests/data/manifest.txt',
                                          (608, 608, 3))
    assert len(anchors) == 2


@pytest.mark.travis
def test_load_anchors_from_file(test_config):
    anchors = cli.load_or_compute_anchors('', 9,
                                          './tests/data/yolov3_anchors.txt',
                                          None, None, None)
    assert len(anchors) == 9
    assert np.all(anchors == YoloV3.default_anchors)


@pytest.mark.travis
def test_load_datasets(test_config):
    dataset = test_config.dataset
    print(dataset)
    anchors = cli.load_or_compute_anchors(ds_mode=dataset.mode,
                                          ds_train_path=dataset.train_path,
                                          image_shape=dataset.image_shape,
                                          **dataset.object_anchors)
    train_ds, val_ds = cli.load_datasets(**dataset, anchors=anchors)
    assert train_ds is not None
    assert val_ds is not None
