import logging
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

YOLOV3_LAYER_LIST = [
    'DarkNet',
    'yolo_head_0',
    'yolo_output_0',
    'yolo_head_1',
    'yolo_output_1',
    'yolo_head_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'DarkNet',
    'yolo_head_0',
    'yolo_output_0',
    'yolo_head_1',
    'yolo_output_1',
]


def set_trainable(layer, value):
    layer.trainable = value
    if isinstance(layer, tf.keras.Model):
        for sub_layer in layer.layers:
            set_trainable(sub_layer, value)


def freeze(model):
    value = False
    model.trainable = value
    set_trainable(model, value)


def unfreeze(model):
    value = True
    model.trainable = value
    set_trainable(model, value)


def freeze_backbone(model):
    freeze(model.layers[1])


def freeze_backbone_layers(model, num_layers):
    """
    Arguments
    --------
    model: a yolo model
    num_layers: the number of layers starting from the last layer of darknet to freeze
    """
    backbone = model.layers[1]
    for layer in backbone.layers[:num_layers]:
        layer.trainable = False


def load_darknet_weights(model,
                         weights_file,
                         tiny=False,
                         for_transfer=False,
                         debug=False):

    if debug:
        logger.setLevel(logging.DEBUG)

    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    logger.info('version major %s, minor %s, revision %s, seen %s', major,
                minor, revision, seen)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        logger.debug('processing layer %s', layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf,
                                         dtype=np.float32,
                                         count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(wf,
                                       dtype=np.float32,
                                       count=np.product(conv_shape))

            logger.debug("%s/%s %s %s", sub_model.name, layer.name,
                         'bn' if batch_norm else 'bias', conv_shape)

            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose(
                [2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    if (len(wf.read()) != 0) and not for_transfer:
        logger.error('failed to read all data')
        wf.close()
        return False
    wf.close()
    return True
