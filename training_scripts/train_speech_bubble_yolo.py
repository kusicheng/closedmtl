from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def f1_from_metrics(metrics) -> tuple[float, float, float, dict[str, float]]:
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    precision = float(results.get("metrics/precision(B)", 0.0))
    recall = float(results.get("metrics/recall(B)", 0.0))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, results


def latest_weight(run_dir: Path) -> Path | None:
    for name in ("best.pt", "last.pt"):
        path = run_dir / "weights" / name
        if path.exists():
            return path
    return None


def candidate_run_dirs(project: Path, name: str) -> list[Path]:
    dirs = [project / name]
    if not project.is_absolute():
        dirs.append(Path("runs") / "detect" / project / name)
    return dirs


def latest_weight_from_dirs(run_dirs: list[Path]) -> Path | None:
    candidates = [weight for run_dir in run_dirs if (weight := latest_weight(run_dir)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def train_until_target(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available from this Python environment. torch={torch.__version__}"
        )

    project = Path(args.project)
    run_dirs = candidate_run_dirs(project, args.name)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    initial = Path(args.weights)
    if not initial.exists():
        raise FileNotFoundError(f"Initial weights not found: {initial}")

    model_path = latest_weight_from_dirs(run_dirs) or initial
    total_epochs = 0
    best_f1 = -1.0
    best_summary: dict[str, object] = {}

    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"starting_weights={model_path}", flush=True)

    while total_epochs < args.max_epochs:
        chunk = min(args.epoch_chunk, args.max_epochs - total_epochs)
        print(f"training_chunk_epochs={chunk} completed_epochs={total_epochs}", flush=True)

        model = YOLO(str(model_path))
        model.train(
            data=args.data,
            epochs=chunk,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=args.name,
            exist_ok=True,
            patience=args.patience,
            plots=False,
            verbose=args.verbose,
        )

        candidate = latest_weight_from_dirs(run_dirs)
        if candidate is not None:
            model_path = candidate

        model = YOLO(str(model_path))
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=args.verbose,
        )
        precision, recall, f1, results = f1_from_metrics(metrics)
        total_epochs += chunk

        summary = {
            "weights": str(model_path),
            "completed_epochs": total_epochs,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "target_f1": args.target_f1,
            "target_reached": f1 > args.target_f1,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "results": results,
        }
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"validation_precision={precision:.6f} validation_recall={recall:.6f} "
            f"validation_f1={f1:.6f}",
            flush=True,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_summary = summary

        if f1 > args.target_f1:
            print(f"stop_condition_reached=f1>{args.target_f1}", flush=True)
            return 0

    if best_summary:
        metrics_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    print(f"stop_condition_not_reached=best_f1:{best_f1:.6f}", flush=True)
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO speech bubble detector on CUDA.")
    parser.add_argument("--data", default="training_data/speech-bubbles-detection-yolo/speech_bubbles.yaml")
    parser.add_argument("--weights", default="runs/detect/models/speech_bubble_yolo/weights/best.pt")
    parser.add_argument("--project", default="models")
    parser.add_argument("--name", default="speech_bubble_yolo_gpu")
    parser.add_argument("--metrics-output", default="models/speech_bubble_yolo_gpu_metrics.json")
    parser.add_argument("--target-f1", type=float, default=0.9)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--epoch-chunk", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train_until_target(parse_args()))
