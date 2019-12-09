import pytest
from pathlib import Path
import numpy as np
from tfyolo3.dataloaders import anchors

BASE_PATH = Path(__file__).parent.parent / 'data'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotations'

@pytest.fixture
def test_path():
    path = BASE_PATH / 'annotations.txt'
    return path

def test_prepare_single_file(test_path):
    boxes_xywh = anchors.prepare_single_file(test_path)
    assert boxes_xywh.shape == (4,4)

def test_prepare_multi_file():
    path = BASE_PATH / 'manifest.txt'
    boxes_xywh = anchors.prepare_multi_file(path)
    assert boxes_xywh.shape == (4,4)

def test_gen_anchors():
    path = BASE_PATH / 'manifest.txt'
    anchor_masks = anchors.gen_anchors(
        path, 2, True
    )
    assert len(anchor_masks) > 0