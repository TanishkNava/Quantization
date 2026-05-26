# encoding: utf-8
"""
Gate model YOLOX exp — adjust depth/width/epochs to match your trained checkpoint.
Classes are read from models/gate/classes.txt (see configs/gate.yaml).
"""
import os

import numpy as np
import pickle
import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.data.datasets.voc import VOCDetection, AnnotationTransform
from yolox.evaluators.voc_eval import voc_eval
from yolox.exp import Exp as MyExp


def _model_name():
    return os.path.basename(os.path.dirname(os.path.abspath(__file__)))


def _classes_path():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo, "models", _model_name(), "classes.txt")


def _load_classes():
    with open(_classes_path(), encoding="utf-8") as f:
        return tuple(
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        )


def _target_transform():
    names = _load_classes()
    return AnnotationTransform(class_to_ind=dict(zip(names, range(len(names)))))


class CustomVOCDetection(VOCDetection):
    def __init__(self, *args, classes=None, **kwargs):
        self._custom_classes = tuple(classes) if classes is not None else _load_classes()
        super().__init__(*args, **kwargs)
        self._classes = list(self._custom_classes)
        self.cats = [{"id": idx, "name": val} for idx, val in enumerate(self._classes)]
        self.class_ids = list(range(len(self._classes)))


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = len(_load_classes())
        # TODO: set these to match your gate training run
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1
        self.max_epoch = 60
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
