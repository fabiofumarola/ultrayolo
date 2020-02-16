import pytest
from pathlib import Path
import numpy as np
from ultrayolo.datasets import genanchors

BASE_PATH = Path(__file__).parent.parent / 'data'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotations'


def test_prepare_data_coco():
    path = BASE_PATH / 'coco_dataset.json'
    boxes_xywh = genanchors.prepare_data(path, (512, 512, 3), 'coco')
    assert boxes_xywh.shape == (6, 4)


# def test_prepare_data_singlefile():
#     path = BASE_PATH / 'annotations.txt'
#     boxes_xywh = genanchors.prepare_data(path, (512, 512, 3), 'singlefile')
#     assert boxes_xywh.shape == (5, 4)


def test_prepare_data_multifile():
    path = BASE_PATH / 'annotations.txt'
    boxes_xywh = genanchors.prepare_data(path, (512, 512, 3), 'singlefile')
    assert boxes_xywh.shape == (5, 4)


def test_gen_anchors_coco():
    path = BASE_PATH / 'coco_dataset.json'
    boxes_xywh = genanchors.prepare_data(path, (512, 512, 3), 'coco')
    anchors = genanchors.gen_anchors(boxes_xywh, 3, True)
    print(anchors)
    assert len(anchors) > 0


def test_gen_anchors_singlefile():
    path = BASE_PATH / 'annotations.txt'
    boxes_xywh = genanchors.prepare_data(path, (512, 512, 3), 'singlefile')
    anchors = genanchors.gen_anchors(boxes_xywh, 3, True)
    print(anchors)
    assert len(anchors) > 0


def test_gen_anchors_multifile():
    path = BASE_PATH / 'manifest.txt'
    boxes_xywh = genanchors.prepare_data(path, (512, 512, 3), 'multifile')
    anchors = genanchors.gen_anchors(boxes_xywh, 3, True)
    print(anchors)
    assert len(anchors) > 0
