from tfyolo3.utils import boundingbox
import numpy as np
from matplotlib import patches, patheffects
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa


def show_img(im, figsize=None, ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_point(ax, xy, color='red'):
    ax.add_patch(patches.Circle(
        xy, fill=True, edgecolor=color, color=color, lw=0))


def draw_rect(ax, b, color='white', lw=4):
    """
    Parameters
    --------
    ax: the axis to plot
    b: a bounding box of type (x_min, y_min, x_max, y_max)

    """
    patch = ax.add_patch(patches.Rectangle(
        b[:2], boundingbox.width(b), boundingbox.height(b), fill=False, edgecolor=color, lw=2))
    draw_outline(patch, lw)


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top',
                   color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


def draw_grid(ax, target_shape, grid_len):
    size_grid_x = int(target_shape[0] / grid_len)
    size_grid_y = int(target_shape[1] / grid_len)

    x_ticks = list(range(0, target_shape[0] + size_grid_x, size_grid_x))
    y_ticks = list(range(0, target_shape[0] + size_grid_y, size_grid_y))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.grid(which='both')


def get_cell_responsible_on_grid(box_xywh, img_size, grid_len):
    """Computes the grid cell responsible to detect the box
    """
    grid_size = img_size / grid_len
    grid_xy = ((box_xywh[..., :2]) // grid_size).astype(int)
    grid_xy = np.clip(grid_xy, 0, grid_len - 1)
    return grid_xy
