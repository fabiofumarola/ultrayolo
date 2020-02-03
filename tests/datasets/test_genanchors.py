import pytest
from pathlib import Path
import numpy as np
from ultrayolo.datasets import genanchors

BASE_PATH = Path(__file__).parent.parent / 'data'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotations'


@pytest.fixture
def test_path():
    path = BASE_PATH / 'annotations.txt'
    return path


def test_prepare_single_file(test_path):
    boxes_xywh = genanchors.prepare_single_file(test_path)
    assert boxes_xywh.shape == (5, 4)


def test_gen_anchors_single_file(test_path):
    anchors = genanchors.gen_anchors(
        test_path, 2, False
    )
    assert len(anchors) > 0


def test_prepare_multi_file():
    path = BASE_PATH / 'manifest.txt'
    boxes_xywh = genanchors.prepare_multi_file(path)
    assert boxes_xywh.shape == (6, 4)


def test_gen_anchors_multi_file():
    path = BASE_PATH / 'manifest.txt'
    anchors = genanchors.gen_anchors(
        path, 2, True
    )
    assert len(anchors) > 0
