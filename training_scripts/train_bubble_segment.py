"""Train both bubble outputs with real masks and a separate box-only warmup."""

import argparse
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"models/ultralytics_config"))

import torch
from filelock import FileLock
from ultralytics import YOLO, __version__
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer

from training_scripts.bubble_metrics import FixedDetectionValidator, FixedSegmentationValidator
from training_scripts.bubble_models import (
    build_small_segment, detection_copy, load_mayocream, restore_detection_weights,
)


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_network(network, path, task):
    model=deepcopy(network).cpu().float()
    model.args={**DEFAULT_CFG_DICT, "task":task, "single_cls":True}
    torch.save({"model":model, "ema":None, "epoch":-1,
                "train_args":model.args, "version":__version__}, path)


class FixedTrainerMixin:
    """Choose checkpoints by the exact validation criterion used for stopping."""

    target_f1=0.90

    def build_dataset(self, img_path, mode="train", batch=None):
        # Ultralytics writes shared label caches non-atomically on Windows.
        # Serialize cache setup while allowing model training to overlap.
        lock=Path(self.args.data).parent/".bubble_dataset_cache.lock"
        with FileLock(str(lock), timeout=600):
            return super().build_dataset(img_path, mode, batch)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode=="train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        workers=self.args.workers
        self.args.workers=0
        try:
            return super().get_dataloader(dataset_path, 1, rank, mode)
        finally:
            self.args.workers=workers

    def validate(self):
        metrics=self.validator(self)
        fixed=self.validator.fixed_report
        values=[fixed["boxes"]["f1"]]
        if self.args.task=="segment":
            values.append(fixed["masks"]["f1"])
        # Ultralytics rounds returned metric dictionaries to five decimals.
        # Use original count-derived values for stopping and checkpoint choice.
        fitness=min(values)
        metrics.pop("fitness", None)
        if self.best_fitness is None or fitness>self.best_fitness:
            self.best_fitness=fitness
        if fixed["valid"] and fitness>=self.target_f1:
            self.stop=True
        report={"epoch":self.epoch+1, "fitness":fitness,
                "target_f1":self.target_f1, "target_reached":fixed["valid"] and fitness>=self.target_f1,
                "fixed":fixed, "metrics":metrics}
        write_json(self.save_dir/f"validation_epoch_{self.epoch+1:03d}.json", report)
        print(f"FIXED epoch={self.epoch+1} criterion={fitness:.6f} "
              f"target={self.target_f1:.6f}", flush=True)
        return metrics, fitness


class BoxTrainer(FixedTrainerMixin, DetectionTrainer):
    def get_validator(self):
        self.loss_names="box_loss", "cls_loss", "dfl_loss"
        return FixedDetectionValidator(self.test_loader, save_dir=self.save_dir,
                                       args=copy(self.args), _callbacks=self.callbacks)


class MaskTrainer(FixedTrainerMixin, SegmentationTrainer):
    def get_validator(self):
        self.loss_names="box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return FixedSegmentationValidator(self.test_loader, save_dir=self.save_dir,
                                          args=copy(self.args), _callbacks=self.callbacks)


def train_stage(network, data, output, task, epochs, args, target):
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory: {output}")
    trainer_class=MaskTrainer if task=="segment" else BoxTrainer
    options=dict(model=network.yaml["yaml_file"], data=str(data.resolve()),
                 project=str(output.parent.resolve()), name=output.name,
                 epochs=epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                 device=args.device, seed=args.seed, deterministic=True,
                 optimizer="AdamW", lr0=args.lr, lrf=0.1, weight_decay=0.0005,
                 warmup_epochs=1.0, patience=args.patience, single_cls=True,
                 mosaic=0.5, close_mosaic=5, mixup=0.0, copy_paste=0.0,
                 fliplr=0.0, degrees=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
                 scale=0.25, translate=0.1, overlap_mask=False, mask_ratio=4,
                 conf=0.001, iou=0.5, max_det=300, val=True,
                 plots=False, save=True, exist_ok=False, amp=False,
                 cache=False, verbose=False)
    trainer=trainer_class(overrides=options)
    # BaseTrainer.setup_model preserves an nn.Module supplied directly.
    # YOLO(yaml).train() would rebuild an uncheckpointed wrapper and lose weights.
    trainer.model=network
    trainer.target_f1=target
    started=time.time()
    trainer.train()
    selected=trainer.best if trainer.best.exists() else trainer.last
    if not selected.exists():
        raise RuntimeError("Training did not save a checkpoint")
    report={"checkpoint":str(selected.resolve()), "sha256":digest(selected),
            "seconds":time.time()-started, "target_f1":target,
            "fixed":trainer.validator.fixed_report,
            "target_reached":trainer.validator.fixed_report["valid"] and
            trainer.validator.fixed_report["boxes"]["f1"]>=target and
            (task!="segment" or trainer.validator.fixed_report["masks"]["f1"]>=target)}
    write_json(output/"stage_result.json", report)
    result=YOLO(str(selected)).model.float().cpu()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, report


def run(args):
    torch.set_num_threads(4)
    if args.device!="cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    data_root=Path(args.data_root).resolve()
    for name in ("boxes.yaml", "segments.yaml", "provenance.json"):
        if not (data_root/name).is_file():
            raise FileNotFoundError(data_root/name)
    output=Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.segment_checkpoint:
        network=YOLO(args.segment_checkpoint).model.float()
        if network.args["task"]!="segment":
            raise ValueError("Continuation requires a segmentation checkpoint")
        provenance={"source":args.segment_checkpoint, "sha256":digest(args.segment_checkpoint),
                    "continuation":"Weights only; optimizer and scheduler start fresh"}
    else:
        factory=build_small_segment if args.track=="small" else load_mayocream
        network, provenance=factory(seed=args.seed)
    run_record={"track":args.track, "arguments":vars(args), "initialization":provenance,
                "data_provenance_sha256":digest(data_root/"provenance.json"),
                "torch":torch.__version__, "ultralytics":__version__, "stages":{},
                "status":"running"}
    write_json(output/"run.json", run_record)
    save_network(network, output/"initial_segment.pt", "segment")
    if args.box_epochs:
        detector, transfer=detection_copy(network)
        detector, box_report=train_stage(detector, data_root/"boxes.yaml", output/"boxes",
                                         "detect", args.box_epochs, args, args.box_target)
        restored=restore_detection_weights(network, detector)
        run_record["stages"]["boxes"]={**box_report, "copy":transfer, "restore":restored}
        write_json(output/"run.json", run_record)
        save_network(network, output/"after_boxes_segment.pt", "segment")
        del detector
    _, segment_report=train_stage(network, data_root/"segments.yaml", output/"segments",
                                  "segment", args.epochs, args, args.target_f1)
    run_record["stages"]["segments"]=segment_report
    run_record["status"]="validation_target_reached" if segment_report["target_reached"] else "needs_review"
    run_record["test_status"]="not_evaluated"
    write_json(output/"run.json", run_record)
    print(json.dumps({"status":run_record["status"], "checkpoint":segment_report["checkpoint"]}), flush=True)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("small", "mayocream"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment-checkpoint")
    parser.add_argument("--box-epochs", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--target-f1", type=float, default=0.90)
    parser.add_argument("--box-target", type=float, default=0.97)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


if __name__=="__main__":
    run(parse_args())
