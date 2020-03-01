=========
Ultrayolo
=========

.. image:: https://img.shields.io/pypi/v/ultrayolo.svg
        :target: https://pypi.python.org/pypi/ultrayolo

.. image:: https://img.shields.io/travis/fabiofumarola/ultrayolo.svg
        :target: https://travis-ci.org/fabiofumarola/ultrayolo

.. image:: https://readthedocs.org/projects/ultrayolo/badge/?version=latest
        :target: https://ultrayolo.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/fabiofumarola/ultrayolo/shield.svg
     :target: https://pyup.io/repos/github/fabiofumarola/ultrayolo/
     :alt: Updates

Tensorflow 2 implementation of yolo version 3. You only look once (YOLO) is a state-of-the-art, real-time object detection system

.. image:: ./images/example_image.png

Features
--------

Ultrayolo implements YOLO object detection paper with the following extensions:

Custom Backbones
^^^^^^^^^^^^^^^^^

- `Darknet <https://pjreddie.com/darknet/yolo/>`_: the original architecture
- `ResNet <https://arxiv.org/abs/1512.03385>`_: *ResNet50V2, ResNet101V2, ResNet152V2*
- `DenseNet <https://arxiv.org/abs/1608.06993>`_: *DenseNet121, DenseNet169, DenseNet201*
- `MobileNetV2 <https://arxiv.org/abs/1608.06993>`_: *MobileNetV2*

Customizable grid size
^^^^^^^^^^^^^^^^^^^^^^^^

By Default YOLO used a grid system that divides the images in cells of *32, 16, and 8* pixels. 
Ultrayolo is modified so that it accepts values multiples of 32. 
That is, given an image of shape *(608,608,3)* it can use *32, 64, 128 and 256* as base grid. 
This parameter can be applied to several image shapes, but sometimes can generate mismatch in the target shape of the network and/or the dataset.
The general rule is to avoid using a value of base grid size greater than image widht and height.

* Free software: Apache Software License 2.0
* Documentation: https://ultrayolo.readthedocs.io.


TODO
=====

* [ ] add mean average precision evaluation script

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
