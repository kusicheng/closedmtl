"""Verified YOLO11 initialization and box-only adaptation for bubble models.

These functions return raw PyTorch networks. Training must inject the returned
network into the trainer: assigning YOLO(yaml).model alone loses loaded weights
in Ultralytics 8.4.61 because that wrapper has no checkpoint.
"""

from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, SegmentationModel


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SMALL=ROOT/"models/best/speech_bubble_yolo_s_gpu.pt"
DEFAULT_MAYOCREAM=ROOT/"models/segmentation_mayocreamVer/model.safetensors"
MASK_PREFIXES=("model.23.proto.", "model.23.cv4.")
CLASSIFIER_KEYS={f"model.23.cv3.{level}.2.{kind}"
                 for level in range(3) for kind in ("weight", "bias")}


def _is_mask(key):
    return key.startswith(MASK_PREFIXES)


def _fingerprint(path):
    path=Path(path).resolve(strict=True)
    with path.open("rb") as source:
        digest=hashlib.file_digest(source, "sha256").hexdigest()
    return {"path":str(path), "sha256":digest, "size_bytes":path.stat().st_size}


def _new_segment(scale, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=SegmentationModel(f"yolo11{scale}-seg.yaml", nc=1, verbose=False)
    model.names={0:"balloon"}
    return model


def _verify_equal(model, state, keys):
    actual=model.state_dict()
    changed=[key for key in keys if actual[key].dtype!=state[key].dtype
             or not torch.equal(actual[key], state[key])]
    if changed:
        raise RuntimeError(f"Transferred tensor values changed: {changed}")


def _transfer_report(destination, copied, initialized):
    state=destination.state_dict()
    return {
        "destination_parameters":sum(p.numel() for p in destination.parameters()),
        "destination_tensors":len(state),
        "copied_tensors":len(copied),
        "copied_elements":sum(state[key].numel() for key in copied),
        "copied_keys":sorted(copied),
        "reinitialized_tensors":len(initialized),
        "reinitialized_elements":sum(state[key].numel() for key in initialized),
        "reinitialized_keys":sorted(initialized),
        "copied_values_verified_exact":True,
    }


def _detector_into_segment(segment, detector, allow_new_classifier=False):
    """Preflight all tensor shapes before transferring any detector values."""
    if type(segment) is not SegmentationModel or type(detector) is not DetectionModel:
        raise TypeError("Expected a SegmentationModel and a DetectionModel")
    source=detector.state_dict()
    destination=segment.state_dict()
    extra=set(source)-set(destination)
    mismatches={key for key in source if key in destination
                and source[key].shape!=destination[key].shape}
    incompatible_dtypes={key for key in source if key in destination
                         and source[key].dtype!=destination[key].dtype}
    allowed=CLASSIFIER_KEYS if allow_new_classifier else set()
    if extra or mismatches-allowed or incompatible_dtypes:
        raise ValueError(f"Incompatible detector: extra={sorted(extra)}, "
                         f"mismatched={sorted(mismatches-allowed)}, "
                         f"dtypes={sorted(incompatible_dtypes)}; use FP32 networks")
    copied={key:value for key, value in source.items() if key not in mismatches}
    uncopied=set(destination)-set(copied)
    unexpected={key for key in uncopied if not _is_mask(key) and key not in mismatches}
    if unexpected:
        raise ValueError(f"Detector is missing shared tensors: {sorted(unexpected)}")
    segment.load_state_dict(copied, strict=False)
    _verify_equal(segment, source, copied)
    return copied, uncopied


def build_small_segment(path=DEFAULT_SMALL, seed=20260922):
    """Transfer the selected detector, leaving one-class logits and masks new."""
    source_info=_fingerprint(path)
    detector=YOLO(str(path), task="detect").model.float().cpu()
    if detector.yaml.get("scale")!="s":
        raise ValueError("The selected source must be a YOLO11s detector")
    segment=_new_segment("s", seed)
    copied, initialized=_detector_into_segment(segment, detector, allow_new_classifier=True)
    report=_transfer_report(segment, copied, initialized)
    report.update({
        "track":"selected_yolo11s_segment",
        "source":source_info,
        "source_classes":detector.names,
        "source_tensors":len(detector.state_dict()),
        "seed":seed,
        "architecture":"yolo11s-seg",
        "class_names":segment.names,
        "new_classifier_keys":sorted(set(initialized)&CLASSIFIER_KEYS),
        "new_mask_keys":sorted(key for key in initialized if _is_mask(key)),
        "class_initialization":"Fresh one-class logits; six-class logits are not averaged.",
    })
    return segment, report


def load_mayocream(path=DEFAULT_MAYOCREAM, seed=20260922):
    """Restore all SafeTensors values; only omitted BatchNorm counters start at zero."""
    source_info=_fingerprint(path)
    segment=_new_segment("n", seed)
    source=load_file(str(path), device="cpu")
    destination=segment.state_dict()
    extra=set(source)-set(destination)
    missing=set(destination)-set(source)
    invalid={key for key in missing if not key.endswith(".num_batches_tracked")}
    mismatched={key for key in source if key in destination
                and (source[key].shape!=destination[key].shape
                     or source[key].dtype!=destination[key].dtype)}
    if extra or invalid or mismatched:
        raise ValueError(f"Incompatible Mayocream state: extra={sorted(extra)}, "
                         f"missing={sorted(invalid)}, mismatched={sorted(mismatched)}")
    complete=dict(source)
    complete.update({key:torch.zeros_like(destination[key]) for key in missing})
    segment.load_state_dict(complete, strict=True)
    _verify_equal(segment, source, source)
    report=_transfer_report(segment, source, missing)
    report.update({
        "track":"mayocream_yolo11n_segment",
        "source":source_info,
        "source_tensors":len(source),
        "seed":seed,
        "architecture":"yolo11n-seg",
        "class_names":segment.names,
        "omitted_batchnorm_counters":sorted(missing),
        "counter_initialization":"Zero; upstream SafeTensors omitted training counters.",
    })
    return segment, report


def detection_copy(segment):
    """Make a one-class box trainer model without modifying mask tensors."""
    if type(segment) is not SegmentationModel or segment.model[-1].nc!=1:
        raise TypeError("Expected a one-class SegmentationModel")
    config=deepcopy(segment.yaml)
    config["head"][-1]=[config["head"][-1][0], 1, "Detect", [1]]
    detector=DetectionModel(config, nc=1, verbose=False)
    source=segment.state_dict()
    destination=detector.state_dict()
    expected={key for key in source if not _is_mask(key)}
    if set(destination)!=expected:
        raise ValueError("Detection architecture does not match the segment model")
    state={key:source[key] for key in destination}
    detector.load_state_dict(state, strict=True)
    detector.names=dict(segment.names)
    _verify_equal(detector, source, state)
    report=_transfer_report(detector, state, set())
    report["mask_tensors_retained_in_segment"]=len(set(source)-expected)
    return detector, report


def restore_detection_weights(segment, detector):
    """Update shared tensors after box training; preserve every mask tensor exactly."""
    masks={key:value.clone() for key, value in segment.state_dict().items() if _is_mask(key)}
    copied, untouched=_detector_into_segment(segment, detector)
    _verify_equal(segment, masks, masks)
    report=_transfer_report(segment, copied, set())
    report.update({
        "preserved_mask_tensors":len(untouched),
        "preserved_mask_keys":sorted(untouched),
        "preserved_mask_values_verified_exact":True,
        "limitation":"Box adaptation changes shared features; mask accuracy requires real-mask validation.",
    })
    return report
