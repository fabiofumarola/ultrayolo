# -*- coding: utf-8 -*-

"""Top-level package for tfyolo3."""

__author__ = """Fabio Fumarola"""
__email__ = 'fabiofumarola@gmail.com'
__version__ = '1.0.0'

import logging
logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(threadName)s %(message)s')

from tfyolo3.tfyolo3 import YoloV3Tiny, YoloV3
from tfyolo3.helpers import callbacks
