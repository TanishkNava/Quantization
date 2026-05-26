"""Moved to exps/dock/yolox_voc_s3.py — update imports to use the new path."""

import warnings

warnings.warn(
    "model/yolox_voc_s3.py moved to exps/dock/yolox_voc_s3.py",
    DeprecationWarning,
    stacklevel=2,
)

from exps.dock.yolox_voc_s3 import *  # noqa: F401, F403
