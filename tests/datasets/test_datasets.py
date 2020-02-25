from ultrayolo.datasets import YoloDatasetMultiFile, YoloDatasetSingleFile, CocoFormatDataset, load_classes
from ultrayolo import YoloV3, YoloV3Tiny, cli
from pathlib import Path
import pytest
import numpy as np

BASE_PATH = Path(__file__).parent.parent / 'data'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotations'


@pytest.fixture()
def test_classes():
    classes_path = BASE_PATH / 'classes.txt'
    classes = load_classes(classes_path)
    yield classes


def test_dataset_multi_file(test_classes):

    ds = YoloDatasetMultiFile(annotations_path=BASE_PATH / 'manifest.txt',
                              img_shape=(256, 256, 3),
                              max_objects=10,
                              batch_size=2,
                              anchors=YoloV3.default_anchors,
                              anchor_masks=YoloV3.default_masks,
                              is_training=True,
                              augmenters=None,
                              pad_to_fixed_size=True)
    assert len(ds) == 2

    for images, grid_data in ds:
        assert images.shape == (2, 256, 256, 3)
        assert len(grid_data) == 3
        grid_len = ds.grid_len
        for grid in grid_data:
            # print(grid.shape)
            assert grid.shape == (2, grid_len, grid_len, 3, 10)
            grid_len *= 2


def test_dataset_single_file(test_classes):

    ds = YoloDatasetSingleFile(annotations_path=BASE_PATH / 'annotations.txt',
                               img_shape=(256, 256, 3),
                               max_objects=10,
                               batch_size=2,
                               anchors=YoloV3.default_anchors,
                               anchor_masks=YoloV3.default_masks,
                               is_training=True,
                               augmenters=None,
                               pad_to_fixed_size=True)
    assert len(ds) == 2

    for images, grid_data in ds:
        assert images.shape == (2, 256, 256, 3)
        assert len(grid_data) == 3
        grid_len = ds.grid_len
        for grid in grid_data:
            # print(grid.shape)
            assert grid.shape == (2, grid_len, grid_len, 3, 10)
            grid_len *= 2


def test_coco_dataset():
    ds = CocoFormatDataset(annotations_path=BASE_PATH / 'coco_dataset.json',
                           img_shape=(256, 256, 3),
                           max_objects=10,
                           batch_size=2,
                           anchors=YoloV3.default_anchors,
                           anchor_masks=YoloV3.default_masks,
                           is_training=True,
                           augmenters=None,
                           pad_to_fixed_size=True,
                           images_folder='images')
    assert len(ds) == 2

    for images, grid_data in ds:
        assert images.shape == (2, 256, 256, 3)
        assert len(grid_data) == 3
        grid_len = ds.grid_len
        for grid in grid_data:
            # print(grid.shape)
            assert grid.shape == (2, grid_len, grid_len, 3, 10)
            grid_len *= 2


def test_coco_dataset_no_annotations():
    ds = CocoFormatDataset(annotations_path=BASE_PATH /
                           'coco_dataset_no_annotations.json',
                           img_shape=(256, 256, 3),
                           max_objects=10,
                           batch_size=2,
                           anchors=YoloV3.default_anchors,
                           anchor_masks=YoloV3.default_masks,
                           is_training=True,
                           augmenters=None,
                           pad_to_fixed_size=True,
                           images_folder='images')
    assert len(ds) == 2

    for images, grid_data in ds:
        assert images.shape == (2, 256, 256, 3)
        assert len(grid_data) == 3
        grid_len = ds.grid_len
        for grid in grid_data:
            # print(grid.shape)
            assert grid.shape == (2, grid_len, grid_len, 3, 10)
            grid_len *= 2


def test_coco_dataset_notraining():
    ds = CocoFormatDataset(annotations_path=BASE_PATH / 'coco_dataset.json',
                           img_shape=(256, 256, 3),
                           max_objects=10,
                           batch_size=2,
                           anchors=YoloV3.default_anchors,
                           anchor_masks=YoloV3.default_masks,
                           is_training=False,
                           augmenters=None,
                           pad_to_fixed_size=True,
                           images_folder='images')
    assert len(ds) == 2
    for _, _, batch_classes in ds:
        assert np.any(batch_classes > 0)


def test_coco_dataset_notraining_aug():
    ds = CocoFormatDataset(annotations_path=BASE_PATH / 'coco_dataset.json',
                           img_shape=(256, 256, 3),
                           max_objects=10,
                           batch_size=2,
                           anchors=YoloV3.default_anchors,
                           anchor_masks=YoloV3.default_masks,
                           is_training=False,
                           augmenters=cli.make_augmentations(),
                           pad_to_fixed_size=True,
                           images_folder='images')
    assert len(ds) == 2
    for _, _, batch_classes in ds:
        assert np.any(batch_classes > 0)


def test_coco_dataset_training_aug():
    ds = CocoFormatDataset(annotations_path=BASE_PATH / 'coco_dataset.json',
                           img_shape=(256, 256, 3),
                           max_objects=10,
                           batch_size=2,
                           anchors=YoloV3.default_anchors,
                           anchor_masks=YoloV3.default_masks,
                           is_training=True,
                           augmenters=cli.make_augmentations(),
                           pad_to_fixed_size=True,
                           images_folder='images')
    assert len(ds) == 2
    for _, batch_target in ds:
        res = False
        for batch in batch_target:
            res_batch = np.any(batch > 0)
            if res_batch:
                res = res_batch
        assert res