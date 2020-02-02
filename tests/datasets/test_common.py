import pytest
from pathlib import Path
import numpy as np
from tfyolo3.datasets import common


BASE_PATH = Path(__file__).parent.parent / 'data'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotations'


@pytest.fixture()
def test_image():
    img_path = IMAGES_PATH / 'AK65JZA-FORD-CAR_0.jpg'
    img = common.open_image(img_path)
    yield img


@pytest.fixture()
def test_boxes_classes():
    box_path = ANNOTATIONS_PATH / 'AK65JZA-FORD-CAR_0.txt'
    boxes, classes = common.open_boxes(box_path)
    yield boxes, classes


@pytest.fixture()
def test_boxes_classes2():
    boxes = np.array([
        [0, 22, 520, 258],
    ])
    classes = np.array([
        [2],
    ])
    yield boxes, classes


def test_open_image(test_image):
    # (H,W,C)
    assert test_image.shape == (251, 502, 3)


def test_open_boxes(test_boxes_classes):
    assert np.all(test_boxes_classes[0] == np.array([
        [0, 22, 520, 258], [0, 0, 442, 500]
    ]))
    assert np.all(test_boxes_classes[1] == np.array([[2], [2]]))


def test_parse_boxes():
    line = '178,226,196,236,1 74,190,98,200,3 128,62,152,72,2 215,166,239,176,2 235,58,259,68,3 93,199,117,209,0 175,124,199,134,0 68,31,86,41,1 50,198,74,208,2'
    boxes, classes = common.parse_boxes(line)
    assert len(boxes) == 9
    assert boxes.shape == (9, 4)
    assert classes.shape == (9, 1)


def test_parse_boxes_batch():
    lines = [
        'image_path 178,226,196,236,1 74,190,98,200,3 128,62,152,72,2 215,166,239,176,2 235,58,259,68,3 93,199,117,209,0 175,124,199,134,0 68,31,86,41,1 50,198,74,208,2',
        'image_path 178,226,196,236,1 74,190,98,200,3 128,62,152,72,2 215,166,239,176,2 235,58,259,68,3 93,199,117,209,0 175,124,199,134,0 68,31,86,41,1'
    ]
    boxes, classes = common.parse_boxes_batch(lines)
    assert len(boxes) == 2
    assert len(classes) == 2
    assert boxes[0].shape == (9, 4)
    assert boxes[1].shape == (8, 4)
    assert classes[0].shape == (9, 1)
    assert classes[1].shape == (8, 1)


def test_pad_to_fixed_size(test_image, test_boxes_classes):
    test_boxes = test_boxes_classes[0]
    target_shape = (512, 512)
    image, boxes = common.pad_to_fixed_size(
        test_image, target_shape, test_boxes)

    assert image.shape[:2] == target_shape
    assert boxes.shape == (2, 4)

    target_shape = (608, 608)
    image, boxes = common.pad_to_fixed_size(
        test_image, target_shape, test_boxes)
    assert image.shape[:2] == target_shape
    assert boxes.shape == (2, 4)


def test_pad_to_fixed_size_images(test_image):
    target_shape = (512, 512)
    image = common.pad_to_fixed_size(test_image, target_shape)
    assert image.shape[:2] == target_shape

    target_shape = (608, 608)
    image = common.pad_to_fixed_size(test_image, target_shape)
    assert image.shape[:2] == target_shape


def test_pad_batch_to_fixed_size(
        test_image, test_boxes_classes, test_boxes_classes2):
    batch_images = [test_image, test_image]
    batch_boxes = [test_boxes_classes[0], test_boxes_classes2[0]]

    target_shape = (608, 608)
    images, boxes = common.pad_batch_to_fixed_size(
        batch_images, target_shape, batch_boxes
    )

    for img, src_box, dst_box in zip(images, batch_boxes, boxes):
        assert img.shape[:2] == target_shape
        assert len(src_box) == len(dst_box)


def test_pad_boxes():
    boxes = np.zeros((10, 4))
    boxes_padded = common.pad_boxes(boxes, 5)
    assert boxes_padded.shape == (5, 4)
    assert isinstance(boxes_padded, np.ndarray)

    boxes = np.zeros((10, 4))
    boxes_padded = common.pad_boxes(boxes, 15)
    assert boxes_padded.shape == (15, 4)
    assert isinstance(boxes_padded, np.ndarray)


def test_pad_list_boxes():
    list_boxes = [np.zeros((10, 4)), np.zeros((20, 4))]
    boxes_padded = np.array([common.pad_boxes(boxes, 12)
                             for boxes in list_boxes])
    assert boxes_padded.shape == (2, 12, 4)


def test_pad_classes():
    list_classes = [np.zeros((10, 1)), np.zeros((20, 1))]
    classes_padded = np.array([common.pad_classes(classes, 12)
                               for classes in list_classes])
    assert classes_padded.shape == (2, 12, 1)


def test_prepare_batch(test_image, test_boxes_classes, test_boxes_classes2):
    max_objects = 100
    target_shape = (512, 512)

    batch_images = [test_image, test_image]
    batch_boxes = [test_boxes_classes[0], test_boxes_classes2[0]]
    batch_classes = [test_boxes_classes[1], test_boxes_classes2[1]]

    images, boxes, classes = common.prepare_batch(
        batch_images, batch_boxes, batch_classes, target_shape, max_objects, pad=True
    )

    assert images.shape == (2, *target_shape, 3)
    assert boxes.shape == (2, 100, 4)
    assert classes.shape == (2, 100, 1)


def test_prepare_batch_no_boxes(test_image):
    max_object = 100
    target_shape = (512, 512)

    batch_images = [test_image, test_image]

    images, boxes, classes = common.prepare_batch(
        batch_images, None, None, target_shape, max_object, pad=True
    )

    assert images.shape == (2, *target_shape, 3)
    assert len(boxes) == 0
    assert len(classes) == 0


def test_to_center_width_height():
    array_box = np.array([[2, 1, 5, 3], [2, 2, 3, 3]])
    result = common.to_center_width_height(array_box)

    assert np.all(result == np.array([[3.5, 2., 3., 2.],
                                      [2.5, 2.5, 1., 1.]]))


def test_best_anchors_iou():
    y_train = np.array([[[474., 40.5, 498., 57.5, 0.],
                         [80.5, 66., 607.5, 384., 0.],
                         [406.5, 36., 447.5, 62., 0.],
                         [0., 54., 54., 88., 0.],
                         [226.5, 49., 265.5, 73., 0.],
                         [124., 49., 164., 73., 0.],
                         [277.5, 49.5, 322.5, 76.5, 0.],
                         [343., 38.5, 391., 63.5, 0.],
                         [74.5, 57., 115.5, 87., 0.],
                         [733., 29.5, 883., 254.5, 0.],
                         [0., 0., 0., 0., 0.]]], dtype=np.float32)

    anchors = np.array([
        [10., 13.],
        [16., 30.],
        [33., 23.],
        [30., 61.],
        [62., 45.],
        [59., 119.],
        [116., 90.],
        [156., 198.],
        [373., 326.]], dtype=np.float32
    )

    result = common.best_anchors_iou(y_train, anchors)

    expected = np.array([[[2],
                          [8],
                          [2],
                          [4],
                          [2],
                          [2],
                          [2],
                          [2],
                          [2],
                          [7],
                          [0]]])

    # print(result)

    assert np.all(result == expected)


def test_transform_target():

    boxes_data = np.array([[[3, 10, 45, 50],
                            [92, 16, 255, 68],
                            [42, 28, 173, 103],
                            [102, 24, 255, 91],
                            [121, 105, 255, 255],
                            [0, 28, 45, 104],
                            [0, 117, 33, 255],
                            [42, 99, 174, 255],
                            [99, 11, 255, 52],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]])

    classes_data = np.array([[[1],
                              [0],
                              [2],
                              [1],
                              [3],
                              [2],
                              [1],
                              [0],
                              [2],
                              [0],
                              [0],
                              [0],
                              [0],
                              [0],
                              [0],
                              [0],
                              [0],
                              [0],
                              [0],
                              [0]]])

    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)],
                       np.float32)

    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    y_data_prepared = common.transform_target(
        boxes_data, classes_data, anchors, masks, 8, 4, (256, 256))

    assert y_data_prepared[0].shape == (1, 8, 8, 3, 9)

    img_id_list, grid_x_list, grid_y_list, box_id_list = np.where(
        np.sum(y_data_prepared[0], axis=-1) > 0)

    for img_id, grid_x, grid_y, box_id in zip(
            img_id_list, grid_x_list, grid_y_list, box_id_list):
        data = y_data_prepared[0][img_id, grid_x, grid_y, box_id]
        box = np.ceil(data[:4] * 256).astype(int)
        class_ = np.where(data[5:])[0]
        ebox = box.tolist() + class_.tolist()
        # check that the row is in the original y_data
        assert np.all(np.isin(ebox, np.dstack([boxes_data, classes_data])))


def test_masks():
    masks = common.make_masks(9)

    assert np.all(masks == np.array([
        [6, 7, 8],
        [3, 4, 5],
        [0, 1, 2]
    ]))

    masks = common.make_masks(6)

    assert np.all(masks == np.array([
        [3, 4, 5],
        [0, 1, 2]
    ]))

def test_anchors_to_string():
    a = np.arange(18).reshape((-1,2))
    res = common.anchors_to_string(a)
    assert res == '0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15 16,17'
