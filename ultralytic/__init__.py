# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = '8.0.75'

from ultralytic.hub import start
from ultralytic.yolo.engine.model import YOLO
from ultralytic.yolo.utils.checks import check_yolo as checks

__all__ = '__version__', 'YOLO', 'checks', 'start'  # allow simpler import
