"""Train real bubble masks and box-only pages together, with separate quality gates."""

import argparse
from collections import defaultdict
from copy import copy, deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from filelock import FileLock
from ultralytics import YOLO, __version__
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import unwrap_model

from training_scripts.bubble_metrics import FixedDetectionValidator, count_metrics
from training_scripts.bubble_models import detection_copy
from training_scripts.joint_bubble_model import joint_model_from_segment
from training_scripts.joint_sampling import DEFAULT_RECTANGLE_SOURCE, build_sampling_policy, make_weighted_loader
from training_scripts.train_bubble_segment import MaskTrainer, digest, save_network, write_json
from training_scripts.verify_joint_bubble_data import verify_joint_data


def image_key(path):
    return str(Path(path).resolve()).casefold()


def grouped_gate(rows, assignments, current, current_paths, mask_target=0.9, box_target=0.965):
    """No strong source can conceal another source's below-target score."""
    expected={image_key(path):group for path, group in assignments.items()}
    if len(expected)!=len(assignments) or set(expected.values())!={"ai4va", "manga"}:
        raise ValueError("Expected unique calibration paths for both mask sources")
    seen=set()
    grouped=defaultdict(list)
    for row in rows:
        key=image_key(row["image"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or repeated validation page")
        seen.add(key)
        grouped[expected[key]].append(row)
    if seen!=set(expected):
        raise ValueError("Validation did not cover every assigned calibration page")
    scores={}
    normalized=[]
    for group in ("ai4va", "manga"):
        group_rows=grouped[group]
        score={"images":len(group_rows), "targets":sum(row["targets"] for row in group_rows)}
        if score["targets"]<=0:
            raise ValueError("Each mask source must have positive targets")
        for kind in ("boxes", "masks"):
            totals={key:sum(row[kind][key] for row in group_rows) for key in ("tp", "fp", "fn")}
            score[kind]=count_metrics(**totals)
            normalized.append(score[kind]["f1"]/mask_target)
        scores[group]=score
    if not current["valid"] or current["targets"]<=0:
        raise ValueError("Current box validation must contain targets")
    expected_current={image_key(path) for path in current_paths}
    seen_current=[image_key(row["image"]) for row in current["image_rows"]]
    if (len(expected_current)!=len(current_paths) or len(set(seen_current))!=len(seen_current)
            or set(seen_current)!=expected_current or current["images"]!=len(seen_current)):
        raise ValueError("Current box validation coverage differs from its manifest")
    normalized.append(current["boxes"]["f1"]/box_target)
    fitness=min(normalized)
    return {"sources":scores, "current_boxes":current, "fitness":fitness,
            "target_reached":fitness>=1.0,
            "targets":{"mask_sources_box_and_mask_f1":mask_target, "current_box_f1":box_target}}


class JointTrainer(MaskTrainer):
    """Use two label loaders and one concatenated model forward per training step."""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        group=self.sampling_policy["groups"]["masks"]
        if mode!="train" or not group["enabled"]:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank not in (-1, 0):
            raise ValueError("Weighted sampling requires a single training device")
        dataset=self.build_dataset(dataset_path, mode, batch_size)
        return make_weighted_loader(dataset, batch_size, self.args.workers, group,
                                     pin_memory=self.device.type=="cuda")

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode=="train":
            return super().build_dataset(img_path, mode, batch)
        # DetectionTrainer forces rectangular validation regardless of args.rect.
        # Construct square letterboxed validation explicitly to match deployment.
        cfg=copy(self.args)
        cfg.rect=False
        stride=max(int(unwrap_model(self.model).stride.max()), 32)
        lock=Path(self.args.data).parent/".bubble_dataset_cache.lock"
        with FileLock(str(lock), timeout=600):
            return build_yolo_dataset(cfg, img_path, batch, self.data,
                                      mode=mode, rect=False, stride=stride)

    def _setup_train(self):
        super()._setup_train()
        if self.world_size>1 or self.args.multi_scale:
            raise ValueError("Joint training currently requires one device and fixed image size")
        data_path=self.joint_root/"current_boxes.yaml"
        self.box_data=check_det_dataset(str(data_path))
        cfg=copy(self.args)
        cfg.task="detect"
        cfg.data=str(data_path)
        cfg.rect=False
        self.box_cfg=cfg
        with FileLock(str(self.joint_root/".current_box_cache.lock"), timeout=600):
            train=build_yolo_dataset(cfg, self.box_data["train"], self.box_batch,
                                     self.box_data, mode="train", stride=self.stride)
            validation=build_yolo_dataset(cfg, self.box_data["val"], 1,
                                          self.box_data, mode="val", stride=self.stride)
        group=self.sampling_policy["groups"]["current_boxes"]
        if group["enabled"]:
            self.box_loader=make_weighted_loader(train, self.box_batch, self.box_workers, group,
                                                 pin_memory=self.device.type=="cuda")
        else:
            self.box_loader=build_dataloader(train, self.box_batch, self.box_workers,
                                            shuffle=True, rank=-1)
        self.box_iterator=iter(self.box_loader)
        self.box_validation_loader=build_dataloader(validation, 1, 0, shuffle=False, rank=-1)
        self.box_batches_seen=0

    def preprocess_batch(self, batch):
        batch=super().preprocess_batch(batch)
        try:
            box_batch=next(self.box_iterator)
        except StopIteration:
            self.box_iterator=iter(self.box_loader)
            box_batch=next(self.box_iterator)
        batch["box_batch"]=super().preprocess_batch(box_batch)
        self.box_batches_seen+=1
        return batch

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        if hasattr(self, "box_loader"):
            self.box_loader.dataset.close_mosaic(hyp=copy(self.box_cfg))
            self.box_loader.reset()
            self.box_iterator=iter(self.box_loader)

    def validate(self):
        metrics=self.validator(self)
        fixed=self.validator.fixed_report
        # Detection-only validation cannot call the segmenter's mask loss on
        # unlabeled masks. A copied detector also avoids fusing the training EMA.
        plain=deepcopy(self.ema.ema).cpu().float()
        plain.__class__=SegmentationModel
        detector, _=detection_copy(plain)
        del plain
        cfg=copy(self.box_cfg)
        cfg.half=False
        cfg.batch=1
        cfg.workers=0
        cfg.plots=False
        cfg.save_json=False
        cfg.agnostic_nms=True
        detector.args=vars(cfg).copy()
        validator=FixedDetectionValidator(self.box_validation_loader,
                                          save_dir=self.save_dir/"current_boxes", args=cfg)
        validator(model=detector)
        current=validator.fixed_report
        del detector, validator
        gate=grouped_gate(fixed["image_rows"], self.validation_groups, current,
                          self.box_validation_loader.dataset.im_files,
                          self.mask_target, self.box_target)
        fitness=gate["fitness"]
        metrics.pop("fitness", None)
        metrics["joint/current_box_f1"]=current["boxes"]["f1"]
        for group, score in gate["sources"].items():
            for kind in ("boxes", "masks"):
                metrics[f"joint/{group}_{kind}_f1"]=score[kind]["f1"]
        if self.best_fitness is None or fitness>self.best_fitness:
            self.best_fitness=fitness
        self.stop=self.stop or gate["target_reached"]
        gate.update({"epoch":self.epoch+1, "box_batches_seen":self.box_batches_seen,
                     "validation_rect":{"masks":bool(self.test_loader.dataset.rect),
                                        "current_boxes":bool(self.box_validation_loader.dataset.rect)},
                     "selection":"Minimum ratio to each independent calibration-source target",
                     "limitation":"Native polygon masks and raw detection boxes are training controls; "
                                  "deployed exact-mask and box checks are still required."})
        write_json(self.save_dir/f"joint_validation_{self.epoch+1:03d}.json", gate)
        print(f"JOINT epoch={self.epoch+1} fitness={fitness:.6f} "
              f"current_box_f1={current['boxes']['f1']:.6f} reached={gate['target_reached']}", flush=True)
        return metrics, fitness


def run(args):
    torch.set_num_threads(4)
    if not 0<args.mask_target<=1 or not 0<args.box_target<=1:
        raise ValueError("Quality targets must be in (0, 1]")
    root=Path(args.data_root).resolve(strict=True)
    for name in ("segments.yaml", "current_boxes.yaml", "validation_groups.json", "provenance.json"):
        if not (root/name).is_file():
            raise FileNotFoundError(root/name)
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory: {output}")
    data_before=verify_joint_data(root)
    sampling=build_sampling_policy(root, getattr(args, "ai4va_positive_weight", 1.0),
                                   getattr(args, "rectangle_weight", 1.0),
                                   getattr(args, "rectangle_source", DEFAULT_RECTANGLE_SOURCE), args.seed)
    output.mkdir(parents=True, exist_ok=False)
    sampling_path=output/"sampling_policy.json"
    write_json(sampling_path, sampling)
    source=Path(args.model).resolve(strict=True)
    network=joint_model_from_segment(YOLO(str(source)).model.float(), args.box_weight)
    settings=dict(model=network.yaml["yaml_file"], data=str(root/"segments.yaml"),
                  project=str(output), name="joint", epochs=args.epochs,
                  imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                  device=args.device, seed=args.seed, deterministic=True,
                  optimizer="AdamW", lr0=args.lr, lrf=0.3, weight_decay=0.0005,
                  nbs=8, warmup_epochs=1.0, patience=args.patience, single_cls=True,
                  mosaic=0.25, close_mosaic=5, mixup=0.0, copy_paste=0.0,
                  fliplr=0.0, degrees=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
                  scale=0.2, translate=0.1, overlap_mask=False, mask_ratio=4,
                  conf=0.001, iou=0.5, max_det=300, agnostic_nms=True,
                  val=True, rect=False, plots=False, save=True, exist_ok=False, amp=False,
                  cache=False, verbose=False)
    trainer=JointTrainer(overrides=settings)
    trainer.model=network
    trainer.joint_root=root
    trainer.validation_groups=json.loads((root/"validation_groups.json").read_text(encoding="utf-8"))
    trainer.box_batch=args.box_batch
    trainer.box_workers=args.box_workers
    trainer.mask_target=args.mask_target
    trainer.box_target=args.box_target
    trainer.sampling_policy=sampling
    sources=[Path(__file__), ROOT/"training_scripts/joint_bubble_model.py",
             ROOT/"training_scripts/train_bubble_segment.py", ROOT/"training_scripts/bubble_metrics.py",
             ROOT/"training_scripts/verify_joint_bubble_data.py",
             ROOT/"training_scripts/joint_sampling.py", sampling_path,
             root/"provenance.json", root/"validation_groups.json"]
    provenance={"track":args.track, "status":"running", "arguments":vars(args),
                "initialization":{"source":str(source), "sha256":digest(source)},
                "source_sha256":{**{str(path):digest(path) for path in sources}, **sampling["source_sha256"]},
                "torch":torch.__version__, "ultralytics":__version__,
                "supervision":"Real mask batches plus separate real box-only batches; no rectangle masks",
                "data_verification":{"before":data_before, "after":None, "unchanged":None},
                "sampling":{"policy_path":str(sampling_path), "policy_sha256":digest(sampling_path),
                            "source_sha256":sampling["source_sha256"],
                            "groups":{name:{key:value for key, value in group.items() if key!="pages"}
                                      for name, group in sampling["groups"].items()}},
                "stages":{}}
    write_json(output/"run.json", provenance)
    started=time.perf_counter()
    trainer.train()
    try:
        data_after=verify_joint_data(root)
        provenance["data_verification"]["after"]=data_after
        unchanged=(data_before["verified_inventory_sha256"]==data_after["verified_inventory_sha256"]
                   and data_before["provenance_sha256"]==data_after["provenance_sha256"]
                   and data_before["checker"]["sha256"]==data_after["checker"]["sha256"])
        provenance["data_verification"]["unchanged"]=unchanged
        if not unchanged:
            raise RuntimeError("Verified training inputs changed during the run")
    except Exception as error:
        provenance["status"]="data_verification_failed"
        provenance["data_verification"]["error"]=str(error)
        write_json(output/"run.json", provenance)
        raise
    for path, expected in provenance["source_sha256"].items():
        if digest(path)!=expected:
            raise RuntimeError(f"Training source changed during run: {path}")
    if digest(source)!=provenance["initialization"]["sha256"]:
        raise RuntimeError("Starting checkpoint changed during training")
    reports=[json.loads(path.read_text(encoding="utf-8"))
             for path in trainer.save_dir.glob("joint_validation_*.json")]
    if not reports or not trainer.best.exists():
        raise RuntimeError("No validated joint checkpoint was saved")
    best=max(reports, key=lambda report:(report["fitness"], report["epoch"]))
    model=YOLO(str(trainer.best)).model.float().cpu()
    # Export a standard inference model without custom training-loss dependencies.
    model.__class__=SegmentationModel
    model.criterion=None
    if hasattr(model, "box_criterion"):
        delattr(model, "box_criterion")
    deployed=output/"segment.pt"
    save_network(model, deployed, "segment")
    provenance["stages"]["segments"]={"checkpoint":str(deployed), "sha256":digest(deployed),
                                        "training_checkpoint":str(trainer.best),
                                        "training_checkpoint_sha256":digest(trainer.best),
                                        "calibration":best, "target_reached":best["target_reached"],
                                        "seconds":time.perf_counter()-started}
    provenance["status"]="validation_target_reached" if best["target_reached"] else "needs_review"
    provenance["test_status"]="not_evaluated"
    write_json(output/"run.json", provenance)
    print(json.dumps({"status":provenance["status"], "checkpoint":str(deployed)}), flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=("small", "mayocream"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--box-batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--box-workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--ai4va-positive-weight", type=float, default=1.0)
    parser.add_argument("--rectangle-weight", type=float, default=1.0)
    parser.add_argument("--rectangle-source", default=str(DEFAULT_RECTANGLE_SOURCE))
    parser.add_argument("--mask-target", type=float, default=0.9)
    parser.add_argument("--box-target", type=float, default=0.965)
    parser.add_argument("--seed", type=int, default=20260927)
    run(parser.parse_args())
