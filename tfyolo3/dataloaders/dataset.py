from tensorflow.keras.utils import Sequence
from pathlib import Path
import numpy as np
import math
from . import common


class YoloModDataset(Sequence):

    def __init__(self, annotations_path, img_shape, max_objects, batch_size,
                 anchors, anchor_masks, grid_len, num_classes,
                 is_training=True, augmenters=None, pad_to_fixed_size=True):
        """Create a dataset that expectes
        An Annotation file with image_name, boxes


        Arguments:
            annotations_path {str} -- [description]
            img_shape {tuple} -- the target shape of the image
            max_objects {int} -- the max number of objects that can be detected from an image
            batch_size {int} -- the size of the batch for the generator
            anchors {numpy.ndarray} -- the anchors to anchor the images in the dataset
            anchor_masks {numpy.ndarray} -- the mask used for the dataset
            grid_len {int} -- the base grid length (example: for 256 -> 8, for 512 -> 16)
            num_classes {int} -- the number of classes

        Keyword Arguments:
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size, 
                otherwise the images are resized to the img_shape (default: {True})

        Returns:
            tensorflow.keras.utils.Sequence -- a dataset sequence
        """
        if not isinstance(annotations_path, Path):
            annotations_path = Path(annotations_path)

        self.images_path = annotations_path.parent / 'images'
        self.lines = annotations_path.read_text().strip().split('\n')
        np.random.shuffle(self.lines)

        self.target_shape = img_shape
        self.batch_size = batch_size
        self.num_classes = num_classes

        # add scaling for the anchors
        self.anchors = anchors.astype(np.float32) / img_shape[0]
        self.anchor_masks = anchor_masks
        self.grid_len = grid_len
        self.is_training = is_training
        self.max_objects = max_objects
        self.augmenters = augmenters
        self.pad_to_fixed_size = pad_to_fixed_size

    def on_epoch_end(self):
        np.random.shuffle(self.lines)

    def __len__(self):
        return math.ceil(len(self.lines) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size
        batch = self.lines[start:stop]

        batch_images = []
        batch_boxes = []

        for line in batch:
            split = line.split(' ')
            img_path = self.images_path / split[0]
            batch_images.append(common.open_image(img_path))
            str_boxes = ' '.join(split[1:])
            batch_boxes.append(
                common.parse_boxes(str_boxes)
            )

        batch_images, batch_boxes = preprocessing.prepare_batch(batch_images, batch_boxes,
                                                                self.target_shape, self.max_objects, self.augmenters,
                                                                self.pad_to_fixed_size)

        if self.is_training:
            batch_boxes = preprocessing.transform_target(
                batch_boxes, self.anchors, self.anchor_masks, self.grid_len,
                self.num_classes, self.target_shape
            )

        return batch_images, batch_boxes
