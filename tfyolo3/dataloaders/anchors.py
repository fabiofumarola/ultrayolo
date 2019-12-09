import numpy as np
import random
from pathlib import Path
from argparse import ArgumentParser
from . import common

np.random.seed = 42


def prepare_single_file(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    lines = filepath.read_text().strip().split('\n')
    images_boxes = [common.parse_boxes(
        line.split(' ')[1:]) for line in lines]

    boxes = np.array([b[:4] for _, box in images_boxes for b in box])
    boxes_xywh = np.array(
        [common.to_center_width_height(b) for b in boxes])

    return boxes_xywh


def prepare_multi_file(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    images_name = filepath.read_text().strip().split('\n')
    annotations = []
    for img_name in images_name:
        annotation_name = img_name.split('.')[0] + '.txt'
        annotation_path = filepath.parent / 'annotations' / annotation_name
        annotations.append(annotation_path)
    boxes = common.open_boxes_batch(annotations)
    boxes = np.concatenate(boxes, axis=0)
    boxes = boxes[:, :4]
    boxes_xywh = np.array(
        [common.to_center_width_height(b) for b in boxes])
    return boxes_xywh


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

    def iou(self, boxes, clusters):  # 1 box -> k clusters
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
                clusters[cl_id] = self.dist_fn(
                    boxes[current_nearest == cl_id], axis=0)

            last_nearest = current_nearest

        clusters = np.ceil(clusters * self.scaling_factor)
        anchors = clusters[:, 2:4]
        anchors = anchors[np.lexsort(anchors.T[0, None])]
        return anchors


def gen_anchors(annotations_path, num_clusters, multifile, scaling_factor=1.0):
    if multifile:
        data = prepare_multi_file(annotations_path)
    else:
        data = prepare_single_file(annotations_path)
    model = AnchorsGenerator(num_clusters, scaling_factor)
    anchors = model.fit(data)
    return anchors


if __name__ == '__main__':
    parser = ArgumentParser("generate the anchors from the dataset boxes")
    parser.add_argument('--dataset', type=str, help='the path to the dataset')
    parser.add_argument('--num_clusters', type=int,
                        default=9, help='the number of centroids')
    parser.add_argument(
        '--outfilename', help='the filename where the anchors are saved')
    parser.add_argument('--scaling_factor', type=int, default=1,
                        help='change this value to a value lower than 1 when you need scale the boxe sizes')
    parser.add_argument('--multifile', action='store_true',
                        help='select to use the multifile style dataset as input')

    args = parser.parse_args()
    multifile = args.multifile
    anchors = gen_anchors(
        args.dataset, args.num_clusters, 
        multifile, args.scaling_factor)
    save_anchors(args.outfilename, anchors)
