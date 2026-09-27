"""YOLO11 segmentation with additional, genuinely box-only supervision.

The two batches share one forward pass. Their native Ultralytics losses retain
per-image scaling: N_mask*L_segment+box_loss_weight*N_box*L_detect. Returned
logging items divide that vector by N_mask, as a normal segmentation trainer
expects. Box annotations never become masks. Concatenation also means that
BatchNorm running statistics see both image sources, including in mask branches.
"""

import math

import torch
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8DetectionLoss


def _raw_predictions(predictions):
    # Train forward returns a dict; ordinary validation forward returns
    # ((decoded_boxes_and_coefficients, prototypes), raw_predictions).
    raw=predictions[1] if isinstance(predictions, tuple) else predictions
    required={"boxes", "scores", "feats", "mask_coefficient", "proto"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise TypeError("Expected ordinary YOLO11 segmentation raw predictions")
    return raw


def _slice_predictions(predictions, start, end, total):
    """Slice every raw prediction tensor along its verified batch dimension."""
    if isinstance(predictions, torch.Tensor):
        if predictions.ndim==0 or predictions.shape[0]!=total:
            raise ValueError("Prediction tensor does not match the combined batch")
        return predictions[start:end]
    if isinstance(predictions, dict):
        return {key:_slice_predictions(value, start, end, total)
                for key, value in predictions.items()}
    if isinstance(predictions, (list, tuple)):
        values=[_slice_predictions(value, start, end, total) for value in predictions]
        return tuple(values) if isinstance(predictions, tuple) else values
    raise TypeError(f"Unsupported raw prediction value: {type(predictions).__name__}")


def _detection_predictions(predictions):
    # Deliberately exclude prototypes and mask coefficients from this objective.
    return {key:predictions[key] for key in ("boxes", "scores", "feats")}


def _segmentation_slots(detection_loss):
    zero=detection_loss.new_zeros(())
    return torch.stack((detection_loss[0], zero, detection_loss[1], detection_loss[2], zero))


class JointBubbleModel(SegmentationModel):
    """A normal segmenter whose loss optionally consumes batch['box_batch']."""

    box_loss_weight=1.0

    def loss(self, batch, preds=None):
        box_batch=batch.get("box_batch")
        if box_batch is None:
            return super().loss(batch, preds)
        if getattr(self, "end2end", False):
            raise ValueError("Joint supervision requires non-end-to-end YOLO11")
        if not isinstance(box_batch, dict) or "masks" in box_batch:
            raise ValueError("box_batch must contain box labels only, with no masks")
        if "masks" not in batch:
            raise ValueError("The primary batch must contain real segmentation masks")
        mask_images, box_images=batch["img"], box_batch["img"]
        if mask_images.ndim!=4 or box_images.ndim!=4:
            raise ValueError("Both image batches must be BCHW tensors")
        mask_count, box_count=mask_images.shape[0], box_images.shape[0]
        if not mask_count or not box_count:
            raise ValueError("Both image batches must be nonempty")
        if mask_images.shape[1:]!=box_images.shape[1:]:
            raise ValueError("Mask and box images must have identical channel/height/width")
        if mask_images.device!=box_images.device or mask_images.dtype!=box_images.dtype:
            raise ValueError("Mask and box images must share a device and dtype")
        weight=float(self.box_loss_weight)
        if not math.isfinite(weight) or weight<0:
            raise ValueError("box_loss_weight must be finite and nonnegative")
        if getattr(self, "criterion", None) is None:
            self.criterion=self.init_criterion()
        if getattr(self, "box_criterion", None) is None:
            self.box_criterion=v8DetectionLoss(self)
        if preds is None:
            preds=self.forward(torch.cat((mask_images, box_images), dim=0))
        raw=_raw_predictions(preds)
        total=mask_count+box_count
        mask_predictions=_slice_predictions(raw, 0, mask_count, total)
        box_predictions=_slice_predictions(raw, mask_count, total, total)
        mask_loss, _=self.criterion(mask_predictions, batch)
        box_loss, _=self.box_criterion(_detection_predictions(box_predictions), box_batch)
        combined=mask_loss+weight*_segmentation_slots(box_loss)
        return combined, combined.detach()/mask_count


def joint_model_from_segment(model, box_loss_weight=1.0):
    """Convert a loaded model in place, preserving every parameter and buffer.

    Set normal training arguments before calling loss. Existing criteria are
    cleared because they can retain a prior device or hyperparameter object.
    Checkpoints reference this module, so keep it importable when loading them.
    """
    if not isinstance(model, SegmentationModel):
        raise TypeError("Expected a loaded SegmentationModel")
    if getattr(model, "end2end", False) or model.model[-1].nc!=1:
        raise ValueError("Expected an ordinary one-class YOLO11 segmenter")
    weight=float(box_loss_weight)
    if not math.isfinite(weight) or weight<0:
        raise ValueError("box_loss_weight must be finite and nonnegative")
    if type(model) not in (SegmentationModel, JointBubbleModel):
        raise TypeError("Refusing to discard behavior from another model subclass")
    model.__class__=JointBubbleModel
    model.box_loss_weight=weight
    model.criterion=None
    model.box_criterion=None
    return model
