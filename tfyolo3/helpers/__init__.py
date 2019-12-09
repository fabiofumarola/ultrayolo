from .callbacks import (
    save_model, lr_scheduler, default_callbacks
)
from datetime import datetime
from pathlib import Path

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