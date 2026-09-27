from __future__ import annotations

import json
import random
import secrets
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT / "models" / "best" / "root_best.pt"
DATA_YAML = ROOT / "training_data" / "speech-bubbles-detection-yolo" / "speech_bubbles.yaml"
VALID_IMAGES = ROOT / "training_data" / "speech-bubbles-detection-yolo" / "images" / "valid"
SUMMARY_PATH = ROOT / "runs" / "detect" / "best_pt_validation_summary.json"
VAL_PROJECT = ROOT / "runs" / "detect"
VAL_NAME = "best_pt_val"
EXAMPLE_NAME = "best_pt_val_examples"


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {DATA_YAML}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        plots=False,
        project=str(VAL_PROJECT),
        name=VAL_NAME,
        exist_ok=True,
        verbose=False,
    )
    results = {key: as_float(value) for key, value in metrics.results_dict.items()}
    precision = results.get("metrics/precision(B)", 0.0)
    recall = results.get("metrics/recall(B)", 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    images = sorted(
        path
        for path in VALID_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(images) < 5:
        raise ValueError(f"Need at least 5 validation images, found {len(images)}")

    seed = secrets.randbits(32)
    rng = random.Random(seed)
    sampled = rng.sample(images, 5)

    predictions = model.predict(
        source=[str(path) for path in sampled],
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(VAL_PROJECT),
        name=EXAMPLE_NAME,
        exist_ok=True,
        verbose=False,
    )

    examples_dir = VAL_PROJECT / EXAMPLE_NAME
    examples = []
    for path, result in zip(sampled, predictions):
        saved_path = examples_dir / Path(result.path).name
        examples.append(
            {
                "source": str(path.relative_to(ROOT)),
                "annotated": str(saved_path.relative_to(ROOT)),
                "detections": int(len(result.boxes) if result.boxes is not None else 0),
            }
        )

    summary = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "data": str(DATA_YAML.relative_to(ROOT)),
        "split": "val",
        "imgsz": 640,
        "batch": 8,
        "device": "cpu",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "results": results,
        "validation_run_dir": str((VAL_PROJECT / VAL_NAME).relative_to(ROOT)),
        "examples_dir": str(examples_dir.relative_to(ROOT)),
        "random_seed": seed,
        "examples": examples,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
