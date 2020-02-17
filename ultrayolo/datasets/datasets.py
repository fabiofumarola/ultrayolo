from pathlib import Path
import numpy as np
import math
import json
from . import common
import logging
from tqdm import tqdm
import tensorflow as tf


class BaseDataset(tf.keras.utils.Sequence):

    def __init__(self,
                 annotations_path,
                 img_shape,
                 max_objects,
                 batch_size,
                 anchors,
                 anchor_masks,
                 is_training=True,
                 augmenters=None,
                 pad_to_fixed_size=True):
        """Create a dataset that expects
        An Annotation file with image_name, boxes


        Arguments:
            annotations_path {str} -- a file that contains the images, annotations
            img_shape {tuple} -- the target shape of the image
            max_objects {int} -- the max number of objects that can be detected from an image
            batch_size {int} -- the size of the batch for the generator
            anchors {numpy.ndarray} -- the anchors to anchor the images in the dataset
            anchor_masks {numpy.ndarray} -- the mask used for the dataset

        Keyword Arguments:
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})

        Returns:
            tensorflow.keras.utils.Sequence -- a dataset sequence
        """

        self.grid_len = img_shape[0] / 32
        if img_shape[0] % 32 != 0:
            raise Exception(
                f'the image {img_shape} shape must have same height and width and be divisible per 32'
            )
        self.grid_len = int(self.grid_len)
        self.annotations_path = Path(annotations_path)
        self.base_path = self.annotations_path.parent

        self.images_path = self.base_path / 'images'
        if not self.images_path.exists():
            raise Exception(
                f'missing images path in the folder {self.base_path}')

        self.target_shape = img_shape
        self.batch_size = batch_size

        self.classes = list(
            enumerate(common.load_classes(self.base_path / 'classes.txt')))
        self.num_classes = len(self.classes)

        # add scaling for the anchors
        self.anchors = anchors
        if anchors is not None:
            self.anchors_scaled = anchors.astype(np.float32) / img_shape[0]
        self.anchor_masks = anchor_masks
        self.is_training = is_training
        self.max_objects = max_objects
        self.augmenters = augmenters
        self.pad_to_fixed_size = pad_to_fixed_size


class YoloDatasetSingleFile(BaseDataset):

    def __init__(self,
                 annotations_path,
                 img_shape,
                 max_objects,
                 batch_size,
                 anchors,
                 anchor_masks,
                 is_training=True,
                 augmenters=None,
                 pad_to_fixed_size=True):
        """Create a dataset that expects
        An Annotation file with image_name, boxes


        Arguments:
            annotations_path {str} -- a file that contains the images, annotations
            img_shape {tuple} -- the target shape of the image
            max_objects {int} -- the max number of objects that can be detected from an image
            batch_size {int} -- the size of the batch for the generator
            anchors {numpy.ndarray} -- the anchors to anchor the images in the dataset
            anchor_masks {numpy.ndarray} -- the mask used for the dataset

        Keyword Arguments:
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})

        Returns:
            tensorflow.keras.utils.Sequence -- a dataset sequence
        """
        super().__init__(annotations_path, img_shape, max_objects, batch_size,
                         anchors, anchor_masks, is_training, augmenters,
                         pad_to_fixed_size)
        self.lines = annotations_path.read_text().strip().split('\n')
        np.random.shuffle(self.lines)

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
        batch_classes = []

        for line in batch:
            split = line.split(' ')
            img_path = self.images_path / split[0]
            batch_images.append(common.open_image(img_path))
            str_boxes = ' '.join(split[1:])
            boxes, classes = common.parse_boxes(str_boxes)
            batch_boxes.append(boxes)
            batch_classes.append(classes)

        batch_images, batch_boxes, batch_classes = common.prepare_batch(
            batch_images, batch_boxes, batch_classes, self.target_shape,
            self.max_objects, self.augmenters, self.pad_to_fixed_size)

        if self.is_training:
            classes = [v[0] for v in self.classes]
            batch_boxes = common.transform_target(batch_boxes, batch_classes,
                                                  self.anchors_scaled,
                                                  self.anchor_masks,
                                                  self.grid_len,
                                                  self.num_classes,
                                                  self.target_shape, classes)
            return batch_images, batch_boxes
        else:
            return batch_images, batch_boxes, batch_classes


class YoloDatasetMultiFile(BaseDataset):

    def __init__(self,
                 annotations_path,
                 img_shape,
                 max_objects,
                 batch_size,
                 anchors,
                 anchor_masks,
                 is_training=True,
                 augmenters=None,
                 pad_to_fixed_size=True):
        """Create a dataset that expectes
        An Annotation file with image_name, boxes


        Arguments:
            annotations_path {str} -- path of the file that enumerate the images name
            img_shape {tuple} -- the target shape of the image
            max_objects {int} -- the max number of objects that can be detected from an image
            batch_size {int} -- the size of the batch for the generator
            anchors {numpy.ndarray} -- the anchors to anchor the images in the dataset
            anchor_masks {numpy.ndarray} -- the mask used for the dataset

        Keyword Arguments:
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})

        Returns:
            tensorflow.keras.utils.Sequence -- a dataset sequence
        """
        super().__init__(annotations_path, img_shape, max_objects, batch_size,
                         anchors, anchor_masks, is_training, augmenters,
                         pad_to_fixed_size)

        # contains all the images name in the dataset
        image_names = self.annotations_path.read_text().strip().split('\n')
        images_path = self.images_path
        self.images_path = np.array(
            [images_path / img_name for img_name in image_names])

        # create the annotations array
        annot_path = self.annotations_path.parent / 'annotations'
        self.annotations_path = []
        for img_name in image_names:
            name = img_name.split('.')[0] + '.txt'
            self.annotations_path.append(annot_path / name)
        self.annotations_path = np.array(self.annotations_path)

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
        batch_images = common.batch_open_image(self.images_path[start:stop])
        batch_boxes, batch_classes = common.open_boxes_batch(
            self.annotations_path[start:stop])

        batch_images, batch_boxes, batch_classes = common.prepare_batch(
            batch_images, batch_boxes, batch_classes, self.target_shape,
            self.max_objects, self.augmenters, self.pad_to_fixed_size)
        if self.is_training:
            classes = [v[0] for v in self.classes]
            batch_boxes = common.transform_target(batch_boxes, batch_classes,
                                                  self.anchors_scaled,
                                                  self.anchor_masks,
                                                  self.grid_len,
                                                  self.num_classes,
                                                  self.target_shape, classes)
            return batch_images, batch_boxes
        else:
            return batch_images, batch_boxes, batch_classes


class CocoFormatDataset(tf.keras.utils.Sequence):
    """this class handles dataset into the `COCO format <http://cocodataset.org/>`_.

    Arguments:
        Sequence {tf.kera.utils.Sequence} -- [description]
    """

    def __init__(self,
                 annotations_path,
                 img_shape,
                 max_objects,
                 batch_size,
                 anchors,
                 anchor_masks,
                 is_training=True,
                 augmenters=None,
                 pad_to_fixed_size=True,
                 images_folder='images'):
        """Create a dataset from taking a file in coco format


        Arguments:
            annotations_path {str} -- the path to the annotations file
            img_shape {tuple} -- the target shape of the image
            max_objects {int} -- the max number of objects that can be detected from an image
            batch_size {int} -- the size of the batch for the generator
            anchors {numpy.ndarray} -- the anchors to anchor the images in the dataset
            anchor_masks {numpy.ndarray} -- the mask used for the dataset

        Keyword Arguments:
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})
            image_folder {str} -- the subfodler where the images are stored
        """
        self.grid_len = img_shape[0] / 32
        if img_shape[0] % 32 != 0:
            raise Exception(
                'the image shape must have same height and width and be divisible per 32 '
            )
        self.grid_len = int(self.grid_len)
        self.annotations_path = Path(annotations_path)
        self.base_path = self.annotations_path.parent

        self.images_path = self.base_path / images_folder
        if not self.images_path.exists():
            raise Exception(f'missing images folder in {self.images_path}')

        self.target_shape = img_shape
        self.batch_size = batch_size

        # add scaling for the anchors
        self.anchors = anchors
        if anchors is not None:
            self.anchors_scaled = anchors.astype(np.float32) / img_shape[0]
        self.anchor_masks = anchor_masks
        self.is_training = is_training
        self.max_objects = max_objects
        self.augmenters = augmenters
        self.pad_to_fixed_size = pad_to_fixed_size

        with open(self.annotations_path, 'r') as fp:
            self.coco_data = json.load(fp)
        self.classes = sorted(
            [(cat['id'], cat['name']) for cat in self.coco_data['categories']],
            key=lambda x: x[0])
        self.num_classes = len(self.classes)

        self.idx_image_doc = {
            doc['id']: doc for doc in self.coco_data['images']
        }
        self.idx_annotations_doc = dict()
        for ann in tqdm(self.coco_data['annotations'], 'load coco annotations'):
            if ann['image_id'] not in self.idx_annotations_doc:
                self.idx_annotations_doc[ann['image_id']] = []
            self.idx_annotations_doc[ann['image_id']].append(ann)
        self.idxs = np.array(list(self.idx_image_doc.keys()))

    def on_epoch_end(self):
        np.random.shuffle(self.idxs)

    def __len__(self):
        return math.ceil(len(self.idxs) / self.batch_size)

    def __to_xymin_xymax(self, x, y, width, height):
        return [x, y, x + width, y + height]

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size
        idxs_batch = self.idxs[start:stop]

        # get the docs for the images in the batch
        batch_images_doc = [self.idx_image_doc[idx] for idx in idxs_batch]
        batch_images = []
        batch_boxes = []
        batch_classes = []
        for doc in batch_images_doc:
            # process the image
            img_id = doc['id']
            if img_id in self.idx_annotations_doc:
                img_path = self.images_path / doc['file_name']
                img = common.open_image(img_path)
                assert np.all(img >= 0)
                batch_images.append(img.astype(np.uint8))

                boxes = []
                classes = []

                for doc in self.idx_annotations_doc[img_id]:
                    img_boxes = self.__to_xymin_xymax(*doc['bbox'])
                    assert np.all(np.array(img_boxes) >= 0)
                    boxes.append(img_boxes)
                    classes.append([doc['category_id']])
                batch_boxes.append(np.array(boxes, np.float32))
                batch_classes.append(np.array(classes, np.int32))

        batch_images, batch_boxes, batch_classes = common.prepare_batch(
            batch_images, batch_boxes, batch_classes, self.target_shape,
            self.max_objects, self.augmenters, self.pad_to_fixed_size)

        if self.is_training:
            classes = [np.float32(v[0]) for v in self.classes]
            batch_boxes = common.transform_target(batch_boxes, batch_classes,
                                                  self.anchors_scaled,
                                                  self.anchor_masks,
                                                  self.grid_len,
                                                  self.num_classes,
                                                  self.target_shape, classes)

            return batch_images, batch_boxes

        return batch_images, batch_boxes, batch_classes
