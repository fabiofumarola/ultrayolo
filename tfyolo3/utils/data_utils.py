from pathlib import Path
import numpy as np


def read_anchors(path):
    """read the anchors from a file saved in the format
    x1,y1, x2,y2, ..., x9, y9

    Parameters
    ---------
    path: the path of the file to read

    Returns
    -------
    values: an array of tuples [(x1,y1), (x2,y2), ..., (x9, y9)]
    """
    if isinstance(path, str):
        path = Path(path)

    text = path.read_text().strip()
    anchors = [[int(x) for x in pair.split(',')] for pair in text.split()]
    return np.array(anchors, dtype=np.int32)


def load_classes(path, as_dict=False):
    """it expect to read a file with one class per line sorted in the same order with respect
        to the class name.
        example:
        dog
        cat
        will be codified as
        dog -> 0
        cat -> 1
        .... -> 2
        The index 0 is used to represent no class

    Parameters
    ------------
    path: the path where the file is saved
    as_dict: load the classes as dictionary (idx, class)
    Returns
    -------
    values: the list of the classes
    """
    if isinstance(path, str):
        path = Path(path)

    classes = path.read_text().strip().split('\n')

    if as_dict:
        classes = dict(enumerate(classes, 0))

    return classes