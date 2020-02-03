from ultrayolo import YoloV3
from ultrayolo.helpers import darknet
import pytest
import wget
from pathlib import Path


@pytest.fixture
def test_model():
    model = YoloV3(
        img_shape=(256, 256, 3),
        training=True
    )
    yield model.model


def test_freeze(test_model):
    darknet.freeze(test_model)
    assert len(test_model.trainable_weights) == 0


def test_freeze_backbone(test_model):
    darknet.freeze_backbone(test_model)
    assert len(test_model.trainable_weights) == 66


def test_unfreeze(test_model):
    darknet.freeze(test_model)
    assert len(test_model.trainable_weights) == 0
    darknet.unfreeze(test_model)
    assert len(test_model.trainable_weights) == 222


def test_freeze_backbone_layers(test_model):
    num_layers = -10
    darknet.freeze_backbone_layers(test_model, num_layers)
    backbone = test_model.layers[1]
    for l in backbone.layers[:num_layers]:
        assert l.trainable is False
    for l in backbone.layers[num_layers:]:
        assert l.trainable is True


def test_load_darknet():
    # download the file
    filepath = Path('yolov3.weights')
    if not filepath.exists():
        filepath = Path(
            wget.download('https://pjreddie.com/media/files/yolov3.weights'))
    model = YoloV3(img_shape=(608, 608, 3), training=True).model
    assert darknet.load_darknet_weights(model, filepath) is True
