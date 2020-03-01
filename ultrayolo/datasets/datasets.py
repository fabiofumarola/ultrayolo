from pathlib import Path
import numpy as np
import math
import json
from . import common
import logging
from tqdm import tqdm
import tensorflow as tf
from typing import Tuple, List
import imgaug.augmenters as iaa


class BaseDataset(tf.keras.utils.Sequence):

    def __init__(self,
                 annotations_path: str,
                 img_shape: Tuple[int, int, int],
                 max_objects: int,
                 batch_size: int,
                 anchors: np.ndarray,
                 anchor_masks: np.ndarray,
                 base_grid_size: int = 32,
                 is_training: bool = True,
                 augmenters: iaa.Sequential = None,
                 pad_to_fixed_size: bool = True,
                 images_folder='images'):
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
            base_grid_size {int} -- the base size for the yolo grid system
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})
            image_folder {str} -- the subfodler where the images are stored

        Returns:
            tensorflow.keras.utils.Sequence -- a dataset sequence
        """
        if (base_grid_size % 32) != 0:
            raise ValueError(
                f'the value {base_grid_size} for base_grid_size must divisible per 32'
            )

        self.grid_sizes = common.get_grid_sizes(img_shape, base_grid_size)
        self.base_grid_size = base_grid_size
        # self.grid_len = int(img_shape[0] / base_grid_size)
        self.annotations_path = Path(annotations_path)
        self.base_path = self.annotations_path.parent

        self.images_path = self.base_path / images_folder
        if not self.images_path.exists():
            raise Exception(
                f'missing images path in the folder {self.base_path}')

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


class YoloDatasetSingleFile(BaseDataset):

    def __init__(self,
                 annotations_path: str,
                 img_shape: Tuple[int, int, int],
                 max_objects: int,
                 batch_size: int,
                 anchors: np.ndarray,
                 anchor_masks: np.ndarray,
                 base_grid_size: int = 32,
                 is_training: bool = True,
                 augmenters: iaa.Sequential = None,
                 pad_to_fixed_size: bool = True,
                 images_folder='images'):
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
            base_grid_size {int} -- the base size for the yolo grid system
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})
            image_folder {str} -- the subfodler where the images are stored

        Returns:
            tensorflow.keras.utils.Sequence -- a dataset sequence
        """
        super().__init__(annotations_path, img_shape, max_objects, batch_size,
                         anchors, anchor_masks, base_grid_size, is_training,
                         augmenters, pad_to_fixed_size)

        self.lines = annotations_path.read_text().strip().split('\n')
        self.classes = list(
            enumerate(common.load_classes(self.base_path / 'classes.txt')))
        self.num_classes = len(self.classes)
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        np.random.shuffle(self.lines)

    def __len__(self) -> int:
        return math.ceil(len(self.lines) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple:
        """return a batch of images
        
        Arguments:
            idx {int} -- the batch id
        
        Returns:
            Tuple -- basing on the values self.is_training
                if True -> Tuple[batch_images, batch_boxes]
                if False -> Tuple[batch_images, batch_boxes, batch_classes]
        """
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
                                                  self.grid_sizes,
                                                  self.num_classes,
                                                  self.target_shape, classes)
            return batch_images, batch_boxes

        return batch_images, batch_boxes, batch_classes


class YoloDatasetMultiFile(BaseDataset):

    def __init__(self,
                 annotations_path: str,
                 img_shape: Tuple[int, int, int],
                 max_objects: int,
                 batch_size: int,
                 anchors: np.ndarray,
                 anchor_masks: np.ndarray,
                 base_grid_size: int = 32,
                 is_training: bool = True,
                 augmenters: iaa.Sequential = None,
                 pad_to_fixed_size: bool = True,
                 images_folder='images'):
        """Create a dataset that expectes
        An Annotation file with image_name, boxes. Boxes are in the form [x_min, y_min, x_max, y_max]


        Arguments:
            annotations_path {str} -- path of the file that enumerate the images name
            img_shape {tuple} -- the target shape of the image
            max_objects {int} -- the max number of objects that can be detected from an image
            batch_size {int} -- the size of the batch for the generator
            anchors {numpy.ndarray} -- the anchors to anchor the images in the dataset
            anchor_masks {numpy.ndarray} -- the mask used for the dataset

        Keyword Arguments:
            base_grid_size {int} -- the base size for the yolo grid system
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})
            image_folder {str} -- the subfodler where the images are stored

        Returns:
            tensorflow.keras.utils.Sequence -- a dataset sequence
        """
        super().__init__(annotations_path, img_shape, max_objects, batch_size,
                         anchors, anchor_masks, base_grid_size, is_training,
                         augmenters, pad_to_fixed_size)

        self.classes = list(
            enumerate(common.load_classes(self.base_path / 'classes.txt')))
        self.num_classes = len(self.classes)

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
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        idxs = np.arange(0, len(self.images_path))
        np.random.shuffle(idxs)
        self.images_path = self.images_path[idxs]
        self.annotations_path = self.annotations_path[idxs]

    def __len__(self) -> int:
        return math.ceil(len(self.images_path) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple:
        """return a batch of images
        
        Arguments:
            idx {int} -- the batch id
        
        Returns:
            Tuple -- basing on the values self.is_training
                if True -> Tuple[batch_images, batch_boxes]
                if False -> Tuple[batch_images, batch_boxes, batch_classes]
        """
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
                                                  self.grid_sizes,
                                                  self.num_classes,
                                                  self.target_shape, classes)
            return batch_images, batch_boxes

        return batch_images, batch_boxes, batch_classes


class CocoFormatDataset(BaseDataset):
    """this class handles dataset into the `COCO format <http://cocodataset.org/>`_.

    Data Format
    ---------
        annotation{
            "id": int, 
            "image_id": int, 
            "category_id": int, 
            "segmentation": RLE or [polygon], 
            "area": float, 
            "bbox": [x,y,width,height],
            "iscrowd": 0 or 1,
        }
        categories[{
        "id": int, "name": str, "supercategory": str,
        }]

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
                 base_grid_size: int = 32,
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
            base_grid_size {int} -- the base size for the yolo grid system
            is_training {bool} -- true if the dataset is used for training false if used to display (default: {True})
            augmenters {imgaug.augmenters} -- the augmenters used for data augmentation (default: {None})
            pad_to_fixed_size {bool} -- if the image is padded to fixed size,
                otherwise the images are resized to the img_shape (default: {True})
            image_folder {str} -- the subfodler where the images are stored
        """
        super().__init__(annotations_path, img_shape, max_objects, batch_size,
                         anchors, anchor_masks, base_grid_size, is_training,
                         augmenters, pad_to_fixed_size)

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
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        np.random.shuffle(self.idxs)

    def __len__(self) -> int:
        return math.ceil(len(self.idxs) / self.batch_size)

    def __to_xymin_xymax(self, x: float, y: float, width: float,
                         height: float) -> List:
        return [x, y, x + width, y + height]

    def __getitem__(self, idx: int) -> Tuple:
        """return a batch of images
        
        Arguments:
            idx {int} -- the batch id
        
        Returns:
            Tuple -- basing on the values self.is_training
                if True -> Tuple[batch_images, batch_boxes]
                if False -> Tuple[batch_images, batch_boxes, batch_classes]
        """
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
                    # the model expects that boxes are in the format x_min, y_min, x_max, y_max
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
                                                  self.grid_sizes,
                                                  self.num_classes,
                                                  self.target_shape, classes)

            return batch_images, batch_boxes

        return batch_images, batch_boxes, batch_classes
