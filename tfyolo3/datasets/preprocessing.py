import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.batches import Batch
import imageio
import numpy as np
import pandas as pd


def open_image(path):
    """
    Read an image using imageio
    Returns
    ------

    image: format (H,W,C)
    """
    img = imageio.imread(path)

    if len(img.shape) == 2:
        img3d = np.zeros((*img.shape, 3))
        img3d[:, :, 0] = img
        img = img3d

    return img


def save_image(img, path):
    imageio.imsave(path, img)


def open_image_batch(paths):
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


def parse_boxes(str_boxes):
    """Parse annotations in the form x_min,y_min,x_max,y_max  x_min,y_min,x_max,y_max ...
    Arguments
    --------
    str_boxes: annotations in the form x_min,y_min,x_max,y_max  x_min,y_min,x_max,y_max ...

    Returns
    ------
    boxes_class: 
    """

    if isinstance(str_boxes, str):
        str_boxes = str_boxes.split(' ')

    boxes_class = []
    for sbox in str_boxes:
        sbox_split = [int(x) for x in sbox.split(',')]
        boxes_class.append(sbox_split[:5])
    boxes_class = np.array(boxes_class)

    return boxes_class


def __transform(image, boxes, augmenters):
    """Apply a transformation over a given image and boxes
    Arguments
    --------
    image: an image with shape (H,W,C)
    boxes: an array of format (xmin, ymin, xmax, ymax)
    augmenters: a pipeline of imgaug augmenters

    Returns
    -------
    image_aug: the image transformed with the given augmenters
    boxes_aug: the boxes transformed with the given augmenters

    """
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


def pad_to_fixed_size(image, boxes, target_shape):
    """Resize and pad images and boxes to the target shape
    Arguments
    --------
    image: an image with shape (H,W,C)
    boxes: an array of format (xmin, ymin, xmax, ymax)
    target_shape: a shape of type (H,W,C)
    Returns
    -------
    image_pad: the image padded
    boxes_pad: the boxes padded

    """

    if boxes is None:
        boxes = np.array([[1, 2, 3, 4, 5]])

    aug = iaa.Sequential([
        iaa.Resize(
            {"longer-side": target_shape[0], "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(
            height=target_shape[0], width=target_shape[1], position=(1, 1))
    ])
    return __transform(image, boxes, aug)


def resize(image, boxes, target_shape, keep_aspect_ratio=True):
    """Resize images and boxes to the target shape
    Arguments
    --------
    image: an image with shape (H,W,C)
    boxes: an array of format (xmin, ymin, xmax, ymax)
    target_shape: a shape of type (H,W,C)
    Returns
    -------
    image_resized: the image resized
    boxes_resized: the boxes resized
    """
    if boxes is None:
        boxes = np.array([[1, 2, 3, 4, 5]])

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


def __transform_batch(batch_images, batch_boxes, augmenters):
    """Apply a transformation over a given array of image and boxes
    Arguments
    --------
    batch_images: a list of images with shape (H,W,C)
    batch_boxes: an list of an array ob boxes with format (xmin, ymin, xmax, ymax)
    augmenters: a pipeline of imgaug augmenters

    Returns
    -------
    images_aug: a list of images transformed with the given augmenters
    boxes_aug: a list of a list of boxes transformed with the given augmenters

    """
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
    for src_boxes, dst_boxes in zip(batch_boxes, batch_processed.bounding_boxes_aug):
        dst_boxes = dst_boxes.to_xyxy_array().tolist()
        final_boxes = []
        for src_row, dst_row in zip(src_boxes, dst_boxes):
            dst_row.append(src_row[-1])
            final_boxes.append(dst_row)

        boxes_aug.append(final_boxes)

    images_aug = batch_processed.images_aug

    return images_aug, boxes_aug


def pad_batch_to_fixed_size(batch_images, batch_boxes, target_shape):
    """Resize and pad images and boxes to the target shape
    Arguments
    --------
    batch_images: an array of images with shape (H,W,C)
    batch_boxes: an array of array with format (xmin, ymin, xmax, ymax, class_name)
    target_shape: a shape of type (H,W,C)
    Returns
    ------
    images_aug: a list of augmented images
    boxes_aug: a list of augmented boxes
    """
    aug = iaa.Sequential([
        iaa.Resize(
            {"longer-side": target_shape[0], "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(
            height=target_shape[0], width=target_shape[1], position=(1, 1))
    ])
    return __transform_batch(batch_images, batch_boxes, aug)


def resize_batch(batch_images, batch_boxes, target_shape):
    """Resize and pad images and boxes to the target shape
    Arguments
    --------
    batch_images: an array of images with shape (H,W,C)
    batch_boxes: an array of array with format (xmin, ymin, xmax, ymax, class_name)
    target_shape: a shape of type (H,W,C)
    Returns
    ------
    images_aug: a list of augmented images
    boxes_aug: a list of augmented boxes
    """
    aug = iaa.Sequential([
        iaa.Resize({"height": target_shape[0], "width": target_shape[1]}),
    ])
    return __transform_batch(batch_images, batch_boxes, aug)


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
    """Prepare a Batch of images and boxes
    Arguments
    --------
    batch_images: an array of images with shape (H,W,C)
    batch_boxes: an array of array with format (xmin, ymin, xmax, ymax, class_name)
    target_shape: a shape of type (H,W,C)
    max_objects: the maximum number of boxes for padding
    augmenters: other augmenters for the batch
    pad: if the images should be padded
    """
    resizing_func = pad_batch_to_fixed_size if pad else resize_batch
    batch_images_pad, batch_boxes_pad = resizing_func(
        batch_images, batch_boxes, target_shape
    )
    # apply augmentation if defined
    if augmenters:
        batch_images_pad, batch_boxes_pad = __transform_batch(
            batch_images_pad, batch_boxes_pad, augmenters
        )

    batch_boxes_pad = [pad_boxes(boxes, max_objects)
                       for boxes in batch_boxes_pad]

    batch_images_pad = np.array(batch_images_pad)
    batch_boxes_pad = np.array(batch_boxes_pad)
    # clip the values to the max size of the immage
    batch_boxes_pad = np.clip(batch_boxes_pad, 0, target_shape[0]-1)

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


def transform_target(y_data, anchors, anchor_masks, grid_len, num_classes, target_shape):
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
