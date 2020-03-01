import pytest
from ultrayolo.layers import core
from tensorflow.keras import Input
import tensorflow as tf

# def test_darknetbody():
#     model = core.DarknetBody()
#     x = tf.zeros((1, 256, 256, 3))
#     x36, x61, x = model(x)
#     assert x36.shape[1:] == (32, 32, 256)
#     assert x61.shape[1:] == (16, 16, 512)
#     assert x.shape[1:] == (8, 8, 1024)

    # def test_resnetbody():
    #     for version in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']:
    #         input_shape = (256, 256, 3)
    #         x = tf.zeros((1, *input_shape))
    #         model = core.ResNetBody(input_shape, version=version)
    #         output = model(x)
    #         assert output[0].shape[1:] == (32, 32, 512)
    #         assert output[1].shape[1:] == (16, 16, 1024)
    #         assert output[2].shape[1:] == (8, 8, 2048)
