from .callbacks import (save_model, lr_scheduler, default_callbacks)
from datetime import datetime
from pathlib import Path
import tensorflow as tf


def create_run_path(checkpoints_path):
    """create the run path to save the checkpoints of the model

    Arguments:
        checkpoints_path {str} -- the path to save the checkpoints

    Returns:
        Path -- the path to save the checkpoints
    """
    run_folder = 'run_' + datetime.now().strftime('%Y%m%d_%H:%M.%S')
    run_path = Path(checkpoints_path) / run_folder
    return run_path


def unfreeze_checkpoint(path):
    """fix an issue in tensorflow that not allow you to reload checkpoints where some layers are freezed
    
    Arguments:
        path {pathlib.Path} -- the path to the h5 file
    """

    if isinstance(path, Path):
        path = str(path.absolute())

    m = tf.keras.models.load_model(path, compile=False)
    m.trainable = True
    m.save(path)
