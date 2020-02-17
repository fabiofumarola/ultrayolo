import numpy as np
import random
from pathlib import Path
from argparse import ArgumentParser
from . import common
from .datasetmode import DatasetMode
from .datasets import YoloDatasetMultiFile, YoloDatasetSingleFile, CocoFormatDataset
from tqdm.autonotebook import tqdm
np.random.seed = 42

# def prepare_single_file(filepath):
#     filepath = Path(filepath)

#     lines = filepath.read_text().strip().split('\n')
#     boxes, _ = common.parse_boxes_batch(lines)
#     boxes_xywh = np.concatenate(
#         [common.to_center_width_height(b) for b in boxes])

#     return boxes_xywh

# def prepare_multi_file(filepath):
#     if not isinstance(filepath, Path):
#         filepath = Path(filepath)

#     images_name = filepath.read_text().strip().split('\n')
#     annotations = []
#     for img_name in images_name:
#         annotation_name = img_name.split('.')[0] + '.txt'
#         annotation_path = filepath.parent / 'annotations' / annotation_name
#         annotations.append(annotation_path)
#     boxes, _ = common.open_boxes_batch(annotations)
#     boxes = np.concatenate(boxes, axis=0)
#     boxes_xywh = np.array(
#         [common.to_center_width_height(b) for b in boxes])
#     return boxes_xywh


def save_anchors(outfilename, anchors):
    if not isinstance(outfilename, Path):
        outfilename = Path(outfilename)

    result = ''
    for anchor in anchors:
        result += ','.join([str(int(x)) for x in anchor]) + ' '
    with outfilename.open('w') as f:
        f.write(result.strip())


class AnchorsGenerator(object):

    def __init__(self, num_clusters, scaling_factor, dist_fn=np.median):
        self.num_clusters = num_clusters
        self.scaling_factor = scaling_factor
        self.dist_fn = dist_fn

    def iou(self, boxes, clusters):    # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.num_clusters

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def fit(self, boxes, random_seed=42):

        num_obs = len(boxes)
        distances = np.empty((num_obs, self.num_clusters))
        last_nearest = np.zeros((num_obs))

        sample = np.random.choice(num_obs, self.num_clusters, replace=False)
        clusters = boxes[sample]

        while True:

            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=-1)

            if (last_nearest == current_nearest).all():
                break

            for cl_id in range(self.num_clusters):
                clusters[cl_id] = self.dist_fn(boxes[current_nearest == cl_id],
                                               axis=0)

            last_nearest = current_nearest

        clusters = np.ceil(clusters * self.scaling_factor)
        anchors = clusters[:, 2:4]
        anchors = anchors[np.lexsort(anchors.T[0, None])]
        return anchors


def prepare_data(annotations_path, image_shape, datasetmode):
    """read a dataset and transform it into a list of boxes
    
    Arguments:
        annotations_path {str} -- the path
        image_shape {[type]} -- [description]
        datasetmode {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if datasetmode == 'singlefile':
        dataset = YoloDatasetSingleFile(annotations_path, image_shape, 20, 1,
                                        None, None, False)
    elif datasetmode == 'multifile':
        dataset = YoloDatasetMultiFile(annotations_path, image_shape, 20, 1,
                                       None, None, False)
    elif datasetmode == 'coco':
        dataset = CocoFormatDataset(annotations_path, image_shape, 20, 1, None,
                                    None, False)

    boxes = []
    for _, batch_boxes, _ in tqdm(dataset):
        if len(batch_boxes) > 0:
            for box in common.to_center_width_height(batch_boxes[0]):
                if np.sum(box) > 0:
                    boxes.append(box)
    boxes = np.array(boxes)
    return boxes


def gen_anchors(boxes_xywh, num_clusters, scaling_factor=1.1):
    """generate anchors
    
    Arguments:
        boxes_xywh {np.ndarray} -- the boxes used to crreate the anchors
        num_clusters {int} -- the number of clusters to generate
    
    Keyword Arguments:
        scaling_factor {float} -- a multiplicator factor to increase thebox size (default: {1.0})
    
    Returns:
        [type] -- [description]
    """
    model = AnchorsGenerator(num_clusters, scaling_factor)
    anchors = model.fit(boxes_xywh)
    return anchors


if __name__ == '__main__':
    parser = ArgumentParser("generate the anchors from the dataset boxes")
    parser.add_argument('--dataset', type=str, help='the path to the dataset')
    parser.add_argument('--num_clusters',
                        type=int,
                        default=9,
                        help='the number of centroids')
    parser.add_argument('--outfilename',
                        help='the filename where the anchors are saved')
    parser.add_argument(
        '--scaling_factor',
        type=int,
        default=1.1,
        help=
        'change this value to a value lower than 1 when you need scale the boxe sizes'
    )
    parser.add_argument('--datasetmode',
                        type=DatasetMode,
                        choices=list(DatasetMode),
                        required=True,
                        help='Select the mode of the dataset')
    parser.add_argument('--image_shape',
                        nargs='+',
                        type=int,
                        default=[608, 608, 3],
                        help='The shape of the images as (Width, Heigth, 3)')

    args = parser.parse_args()
    boxes_xywh = prepare_data(args.dataset, args.image_shape, args.datasetmode)
    anchors = gen_anchors(boxes_xywh, args.num_clusters, args.scaling_factor)
    save_anchors(args.outfilename, anchors)
