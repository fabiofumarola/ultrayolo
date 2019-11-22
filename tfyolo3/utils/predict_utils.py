import numpy as np
import pandas as pd
from pathlib import Path
from tfyolo3.datasets import preprocessing
import math


def batched_iterator(path, batch_size, target_shape, extensions_filter=['jpg', 'png', 'jpeg']):
    """
    Arguments
    -------
    path: the path that contains the images
    batch_size: the size of the batch
    target_shape = the shape of the image (608, 608)

    Returns
    ----- 
    image_names: the name of the image
    image_resizeds: the image resize the the target size by keeping the aspect ratio
    batch: the batch of the images to fed into the neural network

    """
    if not isinstance(path, Path):
        path = Path(path)

    files = [f for f in path.glob('*.*') if f.suffix[1:] in extensions_filter]

    num_batches = math.ceil(len(files) / batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        if end > len(files):
            end = len(files)

        batch = np.zeros((end - start, *target_shape, 3))
        image_names = []
        image_resizeds = []
        for idx, img_path in enumerate(files[start:end]):
            image_names.append(img_path.name)
            img = preprocessing.open_image(img_path)

            img_resized, _ = preprocessing.resize(img, None, target_shape)
            image_resizeds.append(img_resized)

            img_padded, _ = preprocessing.pad_to_fixed_size(
                img, None, target_shape)
            img_padded = img_padded / 255.
            batch[idx] = img_padded
        yield image_names, image_resizeds, batch
