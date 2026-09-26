"""Re-evaluate one unchanged checkpoint under explicit class-aware and merged rules."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"models/ultralytics_config"))

import torch
from training_scripts.bubble_metrics import FixedDetectionValidator


def main():
    torch.set_num_threads(4)
    checkpoint=ROOT/"models/best/speech_bubble_yolo_s_gpu.pt"
    data=ROOT/"training_data/speech-bubbles-detection-yolo/speech_bubbles.yaml"
    output=ROOT/"outputs/bubble_training_20260922/contamination_audit"
    output.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("rb") as stream:
        fingerprint=hashlib.file_digest(stream, "sha256").hexdigest()
    for split in ("val", "test"):
        for merged in (False, True):
            name=f"{split}_{'merged' if merged else 'six_class'}"
            destination=output/f"{name}.json"
            if destination.exists():
                raise FileExistsError(destination)
            args=dict(data=str(data), split=split, task="detect", imgsz=768,
                      batch=1, device="0", workers=0, half=False, rect=False,
                      conf=0.001, iou=0.5, single_cls=merged, agnostic_nms=merged,
                      max_det=300, plots=False, verbose=False, save_json=False)
            validator=FixedDetectionValidator(args=args, save_dir=output/name)
            result=validator(model=str(checkpoint))
            record={"checkpoint":str(checkpoint), "sha256":fingerprint,
                    "arguments":args, "metrics":result, "fixed":validator.fixed_report,
                    "interpretation":"Development benchmark; split independence is not established."}
            destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(name, json.dumps(validator.fixed_report["boxes"]), flush=True)


if __name__=="__main__":
    main()
