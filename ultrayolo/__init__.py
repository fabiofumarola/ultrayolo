# -*- coding: utf-8 -*-
"""Top-level package for ultrayolo."""

__author__ = """Fabio Fumarola"""
__email__ = 'fabiofumarola@gmail.com'
__version__ = '0.7.0'

from ultrayolo.helpers import callbacks
from ultrayolo.ultrayolo import YoloV3Tiny, YoloV3
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(relativeCreated)6d %(threadName)s %(message)s')
