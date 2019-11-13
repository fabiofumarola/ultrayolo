import imageio
import numpy as np


def open_image(path):
    """Open an image using imageio

    Arguments:
        path {str} -- the path of the image

    Returns:
        numpy.ndarray -- format (H,W,C)
    """
    img = imageio.imread(path)

    if len(img.shape) == 2:
        img3d = np.zeros((*img.shape, 3))
        img3d[:, :, 0] = img
        img = img3d

    return img


def save_image(img, path):
    """save an image

    Arguments:
        img {numpy.ndarray} -- an image as numpy array
        path {str} -- the path
    """
    imageio.imsave(path, img)
