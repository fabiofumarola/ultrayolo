from pathlib import Path
import numpy as np


def height(box):
    return box[3] - box[1]


def width(box):
    return box[2] - box[0]


def center(box):
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
    return (x_max + x_min) / 2, (y_max + y_min) / 2


def area(box):
    return width(box) * height(box)


def resize(bboxes, input_shape, target_shape):
    ih, iw = input_shape[:2]
    th, tw = target_shape[:2]
    scale_h = th / ih
    scale_w = tw / iw

    new_bboxes = bboxes.copy()
    new_bboxes[:, [0, 2]] = np.round(new_bboxes[:, [0, 2]] * scale_w)
    new_bboxes[:, [1, 3]] = np.round(new_bboxes[:, [1, 3]] * scale_h)

    # all the negatives are set to 0
    new_bboxes[new_bboxes < 0] = 0
    # this is due to the fact that some annotations have coordinate in pixels greater that the image shape
    new_bboxes[new_bboxes >= target_shape[0]] = target_shape[0] - 1

    return new_bboxes
