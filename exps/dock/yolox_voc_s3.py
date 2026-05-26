# encoding: utf-8
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
    """Custom-class VOC dataset; upstream VOCDetection hardcodes VOC_CLASSES."""

    def __init__(self, *args, classes=None, **kwargs):
        self._custom_classes = tuple(classes) if classes is not None else _load_classes()
        super().__init__(*args, **kwargs)
        self._classes = list(self._custom_classes)
        self.cats = [{"id": idx, "name": val} for idx, val in enumerate(self._classes)]
        self.class_ids = list(range(len(self._classes)))

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets is None or (isinstance(dets, np.ndarray) and dets.shape[0] == 0):
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(
            self.root, "annotations_cache", "VOC" + self._year, name
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        use_07_metric = True if int(self._year) < 2010 else False
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
        return np.mean(aps)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = len(_load_classes())
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1
        self.max_epoch = 60
        self.eval_interval = 10

        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master, get_local_rank

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = CustomVOCDetection(
                data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
                image_sets=[("2012", "train")],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                target_transform=_target_transform(),
                classes=_load_classes(),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import ValTransform

        valdataset = CustomVOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[("2012", "valid")],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            target_transform=_target_transform(),
            classes=_load_classes(),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        return VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
