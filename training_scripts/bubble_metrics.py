"""Fixed operating-point bubble metrics alongside unmodified Ultralytics AP.

Use these validators with ``YOLO.val(validator=FixedSegmentationValidator)``
or return an instance from a custom trainer's ``get_validator`` method.
AP uses predictions down to confidence 0.001. Fixed metrics independently
rematch predictions at confidence 0.35 and IoU 0.50 for boxes and masks.
"""

import math

import torch.distributed as distributed
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator


def count_metrics(tp, fp, fn):
    """Compute micro metrics from counts; an empty evaluation scores zero."""
    return {
        "tp":int(tp), "fp":int(fp), "fn":int(fn),
        "precision":tp/(tp+fp) if tp+fp else 0.0,
        "recall":tp/(tp+fn) if tp+fn else 0.0,
        "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
    }


class _FixedMetrics:
    fixed_iou=0.5
    fixed_kinds=(("boxes", "tp", "B"),)

    def __init__(self, dataloader=None, save_dir=None, args=None,
                 _callbacks=None, *, fixed_conf=0.35):
        if not math.isfinite(fixed_conf) or not 0.001<=fixed_conf<=1.0:
            raise ValueError("fixed_conf must be finite and between 0.001 and 1.0")
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.fixed_conf=float(fixed_conf)
        self.args.conf=0.001
        self.fixed_image_rows=[]

    def init_metrics(self, model):
        """Start a fresh epoch without retaining an earlier epoch's counts."""
        super().init_metrics(model)
        self.args.conf=0.001
        self.fixed_image_rows=[]
        if not math.isclose(float(self.iouv[0]), self.fixed_iou):
            raise ValueError("The first Ultralytics IoU threshold must be 0.50")
        self.metrics.fixed_report=self.fixed_report

    def _process_batch(self, preds, batch):
        # In the segmentation MRO, its super() goes directly to detection;
        # neither base call below re-enters this override.
        ap_matches=super()._process_batch(preds, batch)
        keep=preds["conf"]>=self.fixed_conf
        selected={key:value[keep] for key, value in preds.items()}
        fixed_matches=super()._process_batch(selected, batch)
        targets=int(batch["cls"].shape[0])
        predictions=int(selected["cls"].shape[0])
        row={"image":str(batch["im_file"]), "targets":targets,
             "predictions":predictions}
        for name, match_key, _ in self.fixed_kinds:
            tp=int(fixed_matches[match_key][:, 0].sum())
            row[name]=count_metrics(tp, predictions-tp, targets-tp)
        self.fixed_image_rows.append(row)
        return ap_matches

    def gather_stats(self):
        """Merge fixed rows when the base validator gathers distributed AP."""
        super().gather_stats()
        if distributed.is_available() and distributed.is_initialized():
            rank=distributed.get_rank()
            gathered=[None]*distributed.get_world_size() if rank==0 else None
            distributed.gather_object(self.fixed_image_rows, gathered, dst=0)
            if rank==0:
                self.fixed_image_rows=[row for rows in gathered for row in rows]
            else:
                self.fixed_image_rows=[]

    @property
    def fixed_report(self):
        """Return global counts and per-image rows in validation visit order."""
        targets=sum(row["targets"] for row in self.fixed_image_rows)
        report={"confidence_threshold":self.fixed_conf,
                "iou_threshold":self.fixed_iou,
                "images":len(self.fixed_image_rows), "targets":targets,
                "valid":targets>0, "image_rows":list(self.fixed_image_rows)}
        for name, _, _ in self.fixed_kinds:
            totals={key:sum(row[name][key] for row in self.fixed_image_rows)
                    for key in ("tp", "fp", "fn")}
            report[name]=count_metrics(**totals)
        return report

    def get_stats(self):
        """Keep standard AP and fitness while exposing fixed counts to training."""
        results=dict(super().get_stats())
        report=self.fixed_report
        # YOLO.val() returns this metrics object rather than the validator.
        self.metrics.fixed_report=report
        for name, _, suffix in self.fixed_kinds:
            for key, value in report[name].items():
                results[f"metrics/fixed_{key}({suffix})"]=value
        results["metrics/fixed_targets"]=report["targets"]
        results["metrics/fixed_valid"]=int(report["valid"])
        return results


class FixedDetectionValidator(_FixedMetrics, DetectionValidator):
    """Detection AP plus box F1 at a fixed confidence and IoU threshold."""


class FixedSegmentationValidator(_FixedMetrics, SegmentationValidator):
    """Segmentation AP plus independent fixed box and instance-mask F1."""

    fixed_kinds=(("boxes", "tp", "B"), ("masks", "tp_m", "M"))
