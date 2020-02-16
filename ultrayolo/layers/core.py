import tensorflow as tf
from tensorflow.keras.layers import (ZeroPadding2D, Conv2D, LeakyReLU, Add,
                                     MaxPool2D, BatchNormalization,
                                     UpSampling2D, Concatenate, Lambda)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import (ResNet50V2, ResNet101V2, ResNet152V2,
                                           DenseNet121, DenseNet169,
                                           DenseNet201, MobileNetV2)


def DarknetConv(x, filters, kernel, batch_norm, downsample):
    """the darknet convolution layer

    Arguments:
        x {tf.tensor} -- a valid tensor
        filters {int} -- the number of filters for the convolution
        kernel {int} -- the size of the kernel patch
        batch_norm {boolean} -- to use batch normalization
        downsample {boolean} -- to use o not the downsampling

    Returns:
        tf.tensor -- the output tensor
    """
    if downsample:
        # top left half-padding
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    x = Conv2D(filters=filters,
               kernel_size=kernel,
               strides=strides,
               padding=padding,
               use_bias=not batch_norm,
               kernel_regularizer=l1_l2(0.0005, 0.0005),
               kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=.1)(x)

    return x


def DarknetResidual(x, filters):
    """A residual layer

    Arguments:
        x {tf.tensor} -- a valid tensor
        filters {int} -- the number of filters

    Returns:
        tf.tensor -- the output tensor
    """
    input_ = x
    x = DarknetConv(x,
                    filters=filters,
                    kernel=1,
                    batch_norm=True,
                    downsample=False)
    x = DarknetConv(x,
                    filters=filters * 2,
                    kernel=3,
                    batch_norm=True,
                    downsample=False)
    x = Add()([input_, x])
    return x


def DarknetBody(name=None):
    """the body of the darknet model

    Keyword Arguments:
        name {str} -- the name of the model (default: {None})

    Returns:
        tensorflow.keras.Model -- the tensorflow darknet model
    """
    x = inputs = Input(shape=[None, None, 3], name='Input_N_H_W_C')
    x = DarknetConv(x, filters=32, kernel=3, batch_norm=True, downsample=False)

    x = DarknetConv(x, filters=64, kernel=3, batch_norm=True, downsample=True)
    for _ in range(1):
        x = DarknetResidual(x, filters=32)

    x = DarknetConv(x, filters=128, kernel=3, batch_norm=True, downsample=True)
    for _ in range(2):
        x = DarknetResidual(x, filters=64)

    x = DarknetConv(x, filters=256, kernel=3, batch_norm=True, downsample=True)
    for _ in range(8):
        x = DarknetResidual(x, filters=128)
    x = x36 = x

    x = DarknetConv(x, filters=512, kernel=3, batch_norm=True, downsample=True)
    for _ in range(8):
        x = DarknetResidual(x, filters=256)
    x = x61 = x

    x = DarknetConv(x, filters=1024, kernel=3, batch_norm=True, downsample=True)
    for _ in range(4):
        x = DarknetResidual(x, filters=512)

    return Model(inputs, [x36, x61, x], name=name)


def ResNetBody(input_shape, version='ResNet50V2', num_branches=3):
    """a Darknet body implemented via a ResNetV2

    Arguments:
        input_shape {tuple} -- a tuple of 3 values (example: {(X,X,3)})

    Keyword Arguments:
        version {str} -- [description] (default: {'ResNet50V2'}, values: [
            'ResNet50V2', 'ResNet101V2', 'ResNet152V2'
        ])
        num_branches {int} -- [description] (default: {3}, values: [3,4])

    Raises:
        Exception: in case a not valid value is provided

    Returns:
        tensorflow.keras.Model  --
    """
    if version == 'ResNet50V2':
        model = ResNet50V2(input_shape=input_shape, include_top=False)
    elif version == 'ResNet101V2':
        model = ResNet101V2(input_shape=input_shape, include_top=False)
    elif version == 'ResNet152V2':
        model = ResNet152V2(input_shape=input_shape, include_top=False)
    else:
        msg = f'invalid value {version} for the class name'
        raise Exception(msg)

    inputs = model.input
    x36 = model.get_layer('conv3_block3_out').output
    x61 = model.get_layer('conv4_block4_out').output
    x96 = model.get_layer('conv4_block4_out').output
    x = model.output

    if num_branches == 3:
        return Model(inputs, [x36, x61, x], name=version)
    else:
        return Model(inputs, [x36, x61, x96, x], name=version)


def DenseNetBody(input_shape, version='DenseNet121', num_branches=3):
    """a Yolo backbone using DenseNet

    Arguments:
        input_shape {tuple} -- a valid image shape tuple (values: {(X,X,3)})

    Keyword Arguments:
        version {str} -- a version of densnet (default: {'DenseNet121'},
        values: [
            'DenseNet121', 'DenseNet169', 'DenseNet201'
        ])
        num_branches {int} -- [description] (default: {3}, values: [3,4])

    Raises:
        Exception: in case a not valid value is provided

    Returns:
        tensorflow.keras.Model  --
    """
    if version == 'DenseNet121':
        model = DenseNet121(input_shape=input_shape, include_top=False)
    elif version == 'DenseNet169':
        model = DenseNet169(input_shape=input_shape, include_top=False)
    elif version == 'DenseNet201':
        model = DenseNet201(input_shape=input_shape, include_top=False)
    else:
        msg = f'invalid value {version} for the class name'
        raise Exception(msg)

    inputs = model.input
    x36 = model.get_layer('conv3_block12_concat').output
    x61 = model.get_layer('conv4_block24_concat').output
    x96 = model.get_layer('conv4_block24_concat').output
    x = model.output

    if num_branches == 3:
        return Model(inputs, [x36, x61, x], name=version)
    else:
        return Model(inputs, [x36, x61, x96, x], name=version)


def MobileNetBody(input_shape, version='MobileNetV2', num_branches=3):
    """a yolo mobile net body
    It suports only shapes as [96, 128, 160, 192, 224]

    Arguments:
        input_shape {tuple} -- [description]

    Keyword Arguments:
        version {str} -- a version of mobilenet (default: {'MobileNetV2'},
            values: ['MobileNetV2'])
        num_branches {int} -- [description] (default: {3}, values: [3,4])

    Raises:
        Exception: in case a not valid value is provided

    Returns:
        tensorflow.keras.Model  --
    """
    if version == 'MobileNetV2':
        model = MobileNetV2(input_shape=input_shape, include_top=False)

        inputs = model.input
        x36 = model.get_layer('block_5_add').output
        x61 = model.get_layer('block_12_add').output
        x96 = model.get_layer('block_12_add').output
        x = model.output
    else:
        msg = f'invalid value {version} for the class name'
        raise Exception(msg)

    if num_branches == 3:
        return Model(inputs, [x36, x61, x], name=version)
    else:
        return Model(inputs, [x36, x61, x96, x], name=version)


def DarknetBodyTiny(name=None):
    """the Tiny version of darknet

    Keyword Arguments:
        name {str} -- a name for the model (default: {None})

    Returns:
        tensorflow.keras.Model -- an instance of the darknet tiny model
    """
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, filters=16, kernel=3, batch_norm=True, downsample=False)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, filters=32, kernel=3, batch_norm=True, downsample=False)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, filters=64, kernel=3, batch_norm=True, downsample=False)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, filters=128, kernel=3, batch_norm=True, downsample=False)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x,
                          filters=256,
                          kernel=3,
                          batch_norm=True,
                          downsample=False)    # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, filters=512, kernel=3, batch_norm=True, downsample=False)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x,
                    filters=1024,
                    kernel=3,
                    batch_norm=True,
                    downsample=False)
    return Model(inputs, [x_8, x], name=name)


def YoloHead(x_inputs, filters, name=None, is_tiny=False):
    if isinstance(x_inputs, tuple):
        x = Input(shape=x_inputs[0].shape[1:])
        x_skip = Input(shape=x_inputs[1].shape[1:])
        inputs = x, x_skip

        # concat with skip connection
        x = DarknetConv(x,
                        filters=filters,
                        kernel=1,
                        batch_norm=True,
                        downsample=False)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_skip])
    else:
        x = inputs = Input(x_inputs.shape[1:])

    i = 3 if not is_tiny else 1
    for j in range(i):
        x = DarknetConv(x,
                        filters=filters,
                        kernel=1,
                        batch_norm=True,
                        downsample=False)
        if j < i - 1:
            x = DarknetConv(x,
                            filters=filters * 2,
                            kernel=3,
                            batch_norm=True,
                            downsample=False)

    return Model(inputs, x, name=name)(x_inputs)


def YoloOutput(x_in, filters, num_mask, num_classes, name=None):
    x = input_ = Input(x_in.shape[1:])
    x = DarknetConv(x,
                    filters=filters * 2,
                    kernel=3,
                    batch_norm=True,
                    downsample=False)
    x = DarknetConv(x,
                    filters=num_mask * (5 + num_classes),
                    kernel=1,
                    batch_norm=False,
                    downsample=False)
    x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                        num_mask, 5 + num_classes)))(x)
    # add this layers to replace all the nan with 0
    x = Lambda(lambda w: tf.where(tf.math.is_nan(w), tf.zeros_like(w), w))(x)

    return Model(input_, x, name=name)(x_in)
