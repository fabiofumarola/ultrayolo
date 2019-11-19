import imageio
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.batches import Batch
from pathlib import Path


def load_anchors(path):
    """read the anchors from a file saved in the format
    x1,y1, x2,y2, ..., x9, y9

    Arguments:
        path {str} -- the path of the file to read

    Returns:
        numpy.ndarray -- an array of tuples [(x1,y1), (x2,y2), ..., (x9, y9)]
    """

    if isinstance(path, str):
        path = Path(path)

    text = path.read_text().strip()
    anchors = [[int(x) for x in pair.split(',')] for pair in text.split()]
    return np.array(anchors, dtype=np.int32)


def load_classes(path, as_dict=False):
    """it expect to read a file with one class per line sorted in the same order with respect
        to the class name.
        example:
        dog
        cat
        will be codified as
        dog -> 0
        cat -> 1
        .... -> 2
        The index 0 is used to represent no class

    Arguments:
        path {str} -- the path where the file is saved

    Keyword Arguments:
        as_dict {bool} -- load the classes as dictionary (idx, class) (default: {False})

    Returns:
        list|dict -- the list of the classes
    """
    if isinstance(path, str):
        path = Path(path)

    classes = path.read_text().strip().split('\n')

    if as_dict:
        classes = dict(enumerate(classes, 0))

    return classes


def open_image(path):
    """Open an image using imageio

    Arguments:
        path {str} -- the path of the image

    Returns:
        numpy.ndarray -- format (H,W,C)
    """
    img = imageio.imread(path)

    if len(img.shape) == 2:
        img3d = np.zeros((*img.shape, 3))
        img3d[:, :, 0] = img
        img = img3d

    return img


def batch_open_image(paths):
    images = [open_image(path) for path in paths]
    return images


def open_boxes(path):
    """Read the boxes from a file
    """
    boxes_class = pd.read_csv(path, header=None).values
    return boxes_class


def open_boxes_batch(paths):
    boxes = [open_boxes(path) for path in paths]
    return boxes


def save_image(img, path):
    """save an image

    Arguments:
        img {numpy.ndarray} -- an image as numpy array
        path {str} -- the path
    """
    imageio.imsave(path, img)


def parse_boxes(str_boxes):
    """Parse annotations in the form x_min,y_min,x_max,y_max  x_min,y_min,x_max,y_max ...

    Arguments:
        str_boxes {str} -- annotations in the form x_min,y_min,x_max,y_max  x_min,y_min,x_max,y_max ...

    Returns:
        numpy.ndarray -- a numpy array with the boxes extracted from the input
    """
    if isinstance(str_boxes, str):
        str_boxes = str_boxes.split(' ')

    boxes_class = []
    for sbox in str_boxes:
        sbox_split = [int(x) for x in sbox.split(',')]
        boxes_class.append(sbox_split[:5])
    boxes_class = np.array(boxes_class)

    return boxes_class


def __transform(image, augmenters, boxes=None):
    """Apply a transformation over a given image and boxes
    Arguments
    --------
    image: an image with shape (H,W,C)
    augmenters: a pipeline of imgaug augmenters
    boxes: an array of format (xmin, ymin, xmax, ymax)

    Returns
    -------
    image_aug: the image transformed with the given augmenters
    boxes_aug: the boxes transformed with the given augmenters (optional)

    """
    if boxes is not None:
        bbs = BoundingBoxesOnImage(
            [BoundingBox(*b[:4]) for b in boxes],
            shape=image.shape
        )

        image_aug, boxes_aug = augmenters(image=image, bounding_boxes=bbs)
        # add back the class
        boxes_aug = np.concatenate([
            boxes_aug.to_xyxy_array(),
            boxes[..., -1:]
        ], axis=-1)

        return image_aug, boxes_aug
    else:
        return augmenters(image=image)


def pad_to_fixed_size(image, target_shape, boxes=None):
    """Resize and pad images and boxes to the target shape
    Arguments
    --------
    image: an image with shape (H,W,C)
    target_shape: a shape of type (H,W,C)
    boxes: an array of format (xmin, ymin, xmax, ymax)

    Returns
    -------
    image_pad: the image padded
    boxes_pad: the boxes padded (optional: if boxes is not None)

    """
    augmenters = iaa.Sequential([
        iaa.Resize(
            {"longer-side": target_shape[0], "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(
            height=target_shape[0], width=target_shape[1], position=(1, 1))
    ])
    return __transform(image, augmenters, boxes)


def resize(image, target_shape, boxes=None, keep_aspect_ratio=True):
    """Resize images and boxes to the target shape
    Arguments
    --------
    image: an image with shape (H,W,C)
    target_shape: a shape of type (H,W,C)
    boxes: an array of format (xmin, ymin, xmax, ymax)
    keep_aspect_ratio: (default: True)

    Returns
    -------
    image_resized: the image resized
    boxes_resized: the boxes resized (optional: if boxes is not None)
    """

    if keep_aspect_ratio:
        aug = iaa.Sequential([
            iaa.Resize(
                {"longer-side": target_shape[0], "shorter-side": "keep-aspect-ratio"}),
        ])
    else:
        aug = iaa.Sequential([
            iaa.Resize({"height": target_shape[0], "width": target_shape[1]}),
        ])

    return __transform(image, boxes, aug)


def __transform_batch(batch_images, augmenters, batch_boxes=None):
    """Apply a transformation over a given array of image and boxes
    Arguments
    --------
    batch_images: a list of images with shape (H,W,C)
    augmenters: a pipeline of imgaug augmenters
    batch_boxes: an list of an array ob boxes with format (xmin, ymin, xmax, ymax)

    Returns
    -------
    images_aug: a list of images transformed with the given augmenters
    boxes_aug: a list of a list of boxes transformed with the given augmenters (optional: if boxes is not None)

    """
    if batch_boxes is not None:
        batch_bbs = []
        for image, boxes in zip(batch_images, batch_boxes):
            bbs = BoundingBoxesOnImage(
                [BoundingBox(*b[:4]) for b in boxes],
                shape=image.shape
            )
            batch_bbs.append(bbs)

        # create the batch
        batch = Batch(images=batch_images, bounding_boxes=batch_bbs)
        # process the data
        batch_processed = augmenters.augment_batch(batch)

        # transform back the boxes to the right form and add back the class
        boxes_aug = []
        for src_boxes, dst_boxes in zip(
                batch_boxes, batch_processed.bounding_boxes_aug):
            dst_boxes = dst_boxes.to_xyxy_array().tolist()
            final_boxes = []
            for src_row, dst_row in zip(src_boxes, dst_boxes):
                dst_row.append(src_row[-1])
                final_boxes.append(dst_row)

            boxes_aug.append(final_boxes)

        images_aug = batch_processed.images_aug
        return images_aug, boxes_aug
    else:
        batch = Batch(images=batch_images)
         # process the data
        batch_processed = augmenters.augment_batch(batch)
        return batch_processed.images_aug


def pad_batch_to_fixed_size(batch_images, target_shape, batch_boxes=None):
    """Resize and pad images and boxes to the target shape
    Arguments
    --------
    batch_images: an array of images with shape (H,W,C)
    target_shape: a shape of type (H,W,C)
    batch_boxes: an array of array with format (xmin, ymin, xmax, ymax, class_name)

    Returns
    ------
    images_aug: a list of augmented images
    boxes_aug: a list of augmented boxes (optional: if boxes is not None)
    """
    aug = iaa.Sequential([
        iaa.Resize(
            {"longer-side": target_shape[0], "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(
            height=target_shape[0], width=target_shape[1], position=(1, 1))
    ])
    return __transform_batch(batch_images, aug, batch_boxes)


def resize_batch(batch_images, target_shape, batch_boxes=None):
    """Resize and pad images and boxes to the target shape
    Arguments
    --------
    batch_images: an array of images with shape (H,W,C)
    target_shape: a shape of type (H,W,C)
    batch_boxes: an array of array with format (xmin, ymin, xmax, ymax, class_name)

    Returns
    ------
    images_aug: a list of augmented images
    boxes_aug: a list of augmented boxes (optional: if boxes is not None)
    """
    aug = iaa.Sequential([
        iaa.Resize({"height": target_shape[0], "width": target_shape[1]}),
    ])
    return __transform_batch(batch_images, aug, batch_boxes)


def pad_boxes(boxes, max_objects):
    """Pad boxes to desired size
    Arguments
    --------
    boxes: an array of boxes with shape (N,X Y X Y C)
    max_objects: the maximum number of boxes

    Returns
    ------
    boxes: with shape (max_objects, X Y X Y C)
    """
    if len(boxes) > max_objects:
        paddings = [[0, 0], [0, 0]]
        boxes_padded = np.pad(
            boxes[:max_objects], paddings, mode='constant')
    else:
        paddings = [[0, max_objects - len(boxes)], [0, 0]]
        boxes_padded = np.pad(boxes, paddings, mode='constant')

    return boxes_padded


def prepare_batch(batch_images, batch_boxes, target_shape, max_objects,
                  augmenters=None, pad=True):
    """prepare a batch of images and boxes

    Arguments:
        batch_images {numpy.ndarry} -- an array of images with shape (H,W,C)
        batch_boxes {[float, float, float, float, float]} --
            an array of array with format (xmin, ymin, xmax, ymax, class_name)
        target_shape {tuple} -- a shape of type (H,W,C)
        max_objects {int} -- the maximum number of boxes to track

    Keyword Arguments:
        augmenters {imgaug.augmenters} -- ImgAug augmenters (default: {None})
        pad {bool} -- if the images should be padded (default: {True})

    Returns:
        [type] -- [description]
    """
    resizing_func = pad_batch_to_fixed_size if pad else resize_batch
    batch_images_pad, batch_boxes_pad = resizing_func(
        batch_images, target_shape, batch_boxes
    )
    # apply augmentation if defined
    if augmenters:
        batch_images_pad, batch_boxes_pad = __transform_batch(
            batch_images_pad, augmenters, batch_boxes_pad
        )

    batch_boxes_pad = [pad_boxes(boxes, max_objects)
                       for boxes in batch_boxes_pad]

    batch_images_pad = np.array(batch_images_pad)
    batch_boxes_pad = np.array(batch_boxes_pad)
    # clip the values to the max size of the immage
    batch_boxes_pad = np.clip(batch_boxes_pad, 0, target_shape[0] - 1)

    # scale images
    batch_images_pad = batch_images_pad / 255.

    return batch_images_pad.astype(np.float32), batch_boxes_pad


def to_center_width_height(boxes):
    """transform a numpy array of boxes from
    [x_min, y_min, x_max, y_max] --in--> [x_center, y_center, width, height]
    """
    result = boxes.copy().astype(np.float32)
    boxes_xy = (result[..., 0:2] + result[..., 2:4]) / 2
    boxes_wh = result[..., 2:4] - result[..., 0:2]
    result[..., 0:2] = boxes_xy
    result[..., 2:4] = boxes_wh

    return result


def best_anchors_iou(boxes, anchors):
    """
    Parameters
    --------
    boxes: a numpy array of shape (num_examples, num_bboxes) of type (x_min, y_min, x_max, y_max)
    anchors: a numpy array with the anchors to be used for the object detection (num_anchors, (W, H))
    """
    boxes_xywh = to_center_width_height(boxes)
    boxes_wh = np.expand_dims(boxes_xywh[..., 2:4], -2)
    boxes_wh = np.tile(boxes_wh, [1, 1, len(anchors), 1])

    intersection = np.minimum(boxes_wh[..., 0], anchors[..., 0]) *\
        np.minimum(boxes_wh[..., 1], anchors[..., 1])

    anchors_area = anchors[:, 0] * anchors[:, 1]
    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]

    iou = intersection / (boxes_area + anchors_area - intersection)
    best_anchors_idx = np.expand_dims(np.argmax(iou, axis=-1), -1)

    return best_anchors_idx


def transform_target(y_data, anchors, anchor_masks,
                     grid_len, num_classes, target_shape):
    """Transform y_data in yolo format

    """
    # get the anchor id
    obj_anchors_idx = best_anchors_iou(y_data, anchors)

    y_data_transformed = []

    num_grid_cells = grid_len
    for masks in anchor_masks:

        y_out = np.zeros(
            (len(y_data), num_grid_cells, num_grid_cells,
                len(masks), 4 + 1 + num_classes),
            dtype=np.float32
        )

        for i in range(y_data.shape[0]):
            for j in range(y_data.shape[1]):
                if np.equal(y_data[i, j, 2], 0):
                    continue

                valid_anchor = np.equal(
                    masks, obj_anchors_idx[i, j, 0]).astype(np.int32)

                if np.any(valid_anchor):
                    box = y_data[i, j, 0:4] / target_shape[0]
                    box_xy = (box[0:2] + box[2:4]) / 2

                    anchor_idx = np.where(valid_anchor)
                    grid_xy = (box_xy // (1 / num_grid_cells)
                               ).astype(np.int32)

                    one_hot = np.zeros(num_classes, np.float32)
                    one_hot[int(y_data[i, j, 4])] = 1.

                    # grid[i, y, x, anchor] = (tx, ty, bw, bh, obj, class)
                    y_out[i, grid_xy[1], grid_xy[0],
                          anchor_idx[0][0]] = [*box, 1, *one_hot]

        y_data_transformed.append(y_out)
        num_grid_cells *= 2

    return tuple(y_data_transformed)
