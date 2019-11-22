
from tensorflow.keras.utils import Sequence
from pathlib import Path
import numpy as np
import math
from . import preprocessing

np.random.seed = 42


class SingleFileDataset(Sequence):

    def __init__(self, filepath, target_shape, max_objects, batch_size,
                 anchors, anchor_masks, grid_len, num_classes,
                 is_training=True, augmenters=None, pad_to_fixed_size=True):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        self.images_path = filepath.parent / 'images'
        self.lines = filepath.read_text().strip().split('\n')
        np.random.shuffle(self.lines)

        self.target_shape = target_shape
        self.batch_size = batch_size
        self.num_classes = num_classes

        # add scaling for the anchors
        self.anchors = anchors.astype(np.float32) / target_shape[0]
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
            batch_images.append(preprocessing.open_image(img_path))
            str_boxes = ' '.join(split[1:])
            batch_boxes.append(
                preprocessing.parse_boxes(str_boxes)
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


class YoloDataset(Sequence):

    def __init__(self, image_filepath, target_shape, max_objects, batch_size,
                 anchors, anchor_masks, grid_len, num_classes,
                 is_training=True, augmenters=None, pad_to_fixed_size=True):
        if not isinstance(image_filepath, Path):
            image_filepath = Path(image_filepath)

        self.base_path = image_filepath.parent

        img_path = self.base_path / 'images'
        annot_path = self.base_path / 'annotations'
        # contains all the images name in the dataset
        images_file = image_filepath.read_text().strip().split('\n')

        self.images_path = np.array(
            [img_path / img_name for img_name in images_file])
        self.annotations_path = []
        for img_name in images_file:
            name = img_name.split('.')[0] + '.txt'
            self.annotations_path.append(annot_path / name)
        self.annotations_path = np.array(self.annotations_path)

        self.target_shape = target_shape
        self.batch_size = batch_size
        self.num_classes = num_classes

        # add scaling for the anchors
        self.anchors = anchors.astype(np.float32) / target_shape[0]
        self.anchor_masks = anchor_masks
        self.grid_len = grid_len
        self.is_training = is_training
        self.max_objects = max_objects
        self.augmenters = augmenters
        self.pad_to_fixed_size = pad_to_fixed_size

    def on_epoch_end(self):
        idxs = np.arange(0, len(self.images_path))
        np.random.shuffle(idxs)
        self.images_path = self.images_path[idxs]
        self.annotations_path = self.annotations_path[idxs]

    def __len__(self):
        return math.ceil(len(self.images_path) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size
        batch_images = preprocessing.open_image_batch(
            self.images_path[start:stop])
        batch_boxes = preprocessing.open_boxes_batch(
            self.annotations_path[start:stop])

        batch_images, batch_boxes = preprocessing.prepare_batch(batch_images, batch_boxes,
                                                                self.target_shape, self.max_objects, self.augmenters,
                                                                self.pad_to_fixed_size)

        if self.is_training:
            batch_boxes = preprocessing.transform_target(
                batch_boxes, self.anchors, self.anchor_masks, self.grid_len,
                self.num_classes, self.target_shape
            )

        return batch_images, batch_boxes
