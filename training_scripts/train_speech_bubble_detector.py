from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BACKGROUND_CLASS = 0


@dataclass(frozen=True)
class GroundTruth:
    bbox: tuple[int, int, int, int]
    category_id: int


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    width: int
    height: int
    boxes: tuple[GroundTruth, ...]


def load_coco_split(data_dir: Path, split: str) -> tuple[list[ImageRecord], dict[int, str]]:
    split_dir = data_dir / split
    annotations_path = split_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing required annotation file: {annotations_path}")

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {
        int(cat["id"]): str(cat["name"])
        for cat in data.get("categories", [])
        if int(cat.get("id", BACKGROUND_CLASS)) != BACKGROUND_CLASS
    }
    if not categories:
        raise ValueError(f"No object categories found in {annotations_path}")

    annotations_by_image: dict[int, list[GroundTruth]] = defaultdict(list)
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        annotations_by_image[int(ann["image_id"])].append(
            GroundTruth(
                bbox=(
                    int(round(x)),
                    int(round(y)),
                    int(round(x + w)),
                    int(round(y + h)),
                ),
                category_id=int(ann["category_id"]),
            )
        )

    records: list[ImageRecord] = []
    missing_images: list[Path] = []
    for image in data.get("images", []):
        path = split_dir / image["file_name"]
        if not path.exists():
            missing_images.append(path)
            continue
        records.append(
            ImageRecord(
                path=path,
                width=int(image["width"]),
                height=int(image["height"]),
                boxes=tuple(annotations_by_image.get(int(image["id"]), [])),
            )
        )

    if missing_images:
        preview = "\n".join(str(path) for path in missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} image files referenced by {annotations_path} are missing:\n{preview}"
        )
    if not records:
        raise ValueError(f"No image records found in {annotations_path}")
    if not any(record.boxes for record in records):
        raise ValueError(f"No bounding-box annotations found in {annotations_path}")

    return records, categories


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter)


def nms_boxes(
    detections: Iterable[tuple[tuple[int, int, int, int], float, int]],
    iou_threshold: float,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    kept: list[tuple[tuple[int, int, int, int], float, int]] = []
    for box, score, category_id in sorted(detections, key=lambda item: item[1], reverse=True):
        if all(box_iou(box, kept_box) < iou_threshold for kept_box, _, kept_class in kept if kept_class == category_id):
            kept.append((box, score, category_id))
    return kept


def add_box(
    boxes: set[tuple[int, int, int, int]],
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> None:
    x1, y1, x2, y2 = clip_box(box, image_width, image_height)
    w = x2 - x1
    h = y2 - y1
    area = w * h
    image_area = image_width * image_height
    if w < 18 or h < 18:
        return
    if area < image_area * 0.00012 or area > image_area * 0.45:
        return
    aspect = w / max(1, h)
    if aspect < 0.08 or aspect > 12.0:
        return
    boxes.add((x1, y1, x2, y2))


def anchor_boxes(image_width: int, image_height: int) -> list[tuple[int, int, int, int]]:
    aspects = (0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.2)
    height_fractions = (0.055, 0.085, 0.125, 0.18, 0.26, 0.38)
    boxes: list[tuple[int, int, int, int]] = []
    for height_fraction in height_fractions:
        box_height = max(20, int(image_height * height_fraction))
        for aspect in aspects:
            box_width = max(20, int(box_height * aspect))
            if box_width > image_width * 0.75 or box_height > image_height * 0.75:
                continue
            step_x = max(16, int(box_width * 0.5))
            step_y = max(16, int(box_height * 0.5))
            y = 0
            while y + box_height <= image_height:
                x = 0
                while x + box_width <= image_width:
                    boxes.append((x, y, x + box_width, y + box_height))
                    x += step_x
                y += step_y
    return boxes


def propose_boxes(image: np.ndarray, include_anchor_grid: bool = False) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes: set[tuple[int, int, int, int]] = set()

    for threshold in (170, 190, 210, 225, 238, 247):
        mask = cv2.inRange(gray, threshold, 255)
        for kernel_size in (3, 7, 13, 21):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                add_box(boxes, (x - 3, y - 3, x + w + 3, y + h + 3), width, height)

    edges = cv2.Canny(gray, 45, 130)
    for kernel_size in (5, 9, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            add_box(boxes, (x - 4, y - 4, x + w + 4, y + h + 4), width, height)

    candidates = [(box, 1.0, 1) for box in boxes]
    cv_boxes = [box for box, _, _ in nms_boxes(candidates, iou_threshold=0.92)]
    if not include_anchor_grid:
        return cv_boxes

    all_boxes = set(cv_boxes)
    all_boxes.update(anchor_boxes(width, height))
    return list(all_boxes)


def extract_features(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area = bw * bh
    image_area = width * height
    percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
    gray_small = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    edge_small = cv2.resize(edges, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    return np.array(
        [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
            bw / width,
            bh / height,
            area / image_area,
            bw / bh,
            float(gray.mean()) / 255.0,
            float(gray.std()) / 255.0,
            float(np.mean(gray > 170)),
            float(np.mean(gray > 210)),
            float(np.mean(gray > 235)),
            float(np.mean(gray < 70)),
            float(np.mean(edges > 0)),
            *(float(value) / 255.0 for value in percentiles),
            *gray_small,
            *edge_small,
        ],
        dtype=np.float32,
    )


def jitter_box(box: tuple[int, int, int, int], width: int, height: int) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    variants = [box]
    for dx, dy, scale in ((0.04, 0.04, 1.05), (-0.04, 0.03, 0.95), (0.02, -0.05, 1.10)):
        cx = (x1 + x2) / 2 + bw * dx
        cy = (y1 + y2) / 2 + bh * dy
        nw = bw * scale
        nh = bh * scale
        variants.append(
            clip_box(
                (
                    int(round(cx - nw / 2)),
                    int(round(cy - nh / 2)),
                    int(round(cx + nw / 2)),
                    int(round(cy + nh / 2)),
                ),
                width,
                height,
            )
        )
    return variants


def build_training_matrix(
    records: list[ImageRecord],
    negative_ratio: int,
    max_negative_proposals_per_image: int,
    include_anchor_grid: bool,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    features: list[np.ndarray] = []
    labels: list[int] = []
    positives = 0
    negatives: list[np.ndarray] = []

    for record_index, record in enumerate(records, start=1):
        if record_index == 1 or record_index % 100 == 0:
            print(f"Building samples: {record_index}/{len(records)} images", flush=True)
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {record.path}")

        gt_boxes = [gt.bbox for gt in record.boxes]
        for gt in record.boxes:
            for variant in jitter_box(gt.bbox, record.width, record.height):
                features.append(extract_features(image, variant))
                labels.append(gt.category_id)
                positives += 1

        negative_boxes: list[tuple[int, int, int, int]] = []
        for proposal in propose_boxes(image, include_anchor_grid=False):
            if gt_boxes:
                ious = np.array([box_iou(proposal, gt_box) for gt_box in gt_boxes], dtype=np.float32)
                best_index = int(np.argmax(ious))
                best_iou = float(ious[best_index])
            else:
                best_index = -1
                best_iou = 0.0
            if best_iou >= 0.55:
                features.append(extract_features(image, proposal))
                labels.append(record.boxes[best_index].category_id)
                positives += 1
            elif best_iou < 0.25:
                negative_boxes.append(proposal)

        if include_anchor_grid:
            anchors = anchor_boxes(record.width, record.height)
            sample_count = min(len(anchors), max_negative_proposals_per_image * 3)
            for index in rng.choice(len(anchors), size=sample_count, replace=False):
                anchor = anchors[int(index)]
                if not gt_boxes or max(box_iou(anchor, gt_box) for gt_box in gt_boxes) < 0.25:
                    negative_boxes.append(anchor)

        if negative_boxes:
            negative_count = min(len(negative_boxes), max_negative_proposals_per_image)
            chosen = rng.choice(len(negative_boxes), size=negative_count, replace=False)
            for index in chosen:
                negatives.append(extract_features(image, negative_boxes[int(index)]))

    max_negatives = min(len(negatives), max(positives * negative_ratio, 1))
    if max_negatives:
        chosen = rng.choice(len(negatives), size=max_negatives, replace=False)
        for index in chosen:
            features.append(negatives[int(index)])
            labels.append(BACKGROUND_CLASS)

    if not features:
        raise ValueError("No training samples could be generated.")
    return np.vstack(features), np.array(labels, dtype=np.int64)


def predict_records(
    classifier,
    records: list[ImageRecord],
    confidence_threshold: float,
    include_anchor_grid: bool,
    max_scored_candidates: int,
) -> dict[Path, list[tuple[tuple[int, int, int, int], float, int]]]:
    scored_predictions = score_records(classifier, records, include_anchor_grid, max_scored_candidates)
    return {
        path: threshold_detections(detections, confidence_threshold)
        for path, detections in scored_predictions.items()
    }


def score_records(
    classifier,
    records: list[ImageRecord],
    include_anchor_grid: bool,
    max_scored_candidates: int,
) -> dict[Path, list[tuple[tuple[int, int, int, int], float, int]]]:
    predictions: dict[Path, list[tuple[tuple[int, int, int, int], float, int]]] = {}
    class_order = get_classifier_classes(classifier)
    background_index = class_order.index(BACKGROUND_CLASS) if BACKGROUND_CLASS in class_order else -1

    for record_index, record in enumerate(records, start=1):
        if record_index == 1 or record_index % 50 == 0:
            print(f"Scoring validation: {record_index}/{len(records)} images", flush=True)
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {record.path}")
        boxes = propose_boxes(image, include_anchor_grid=include_anchor_grid)
        if not boxes:
            predictions[record.path] = []
            continue
        matrix = np.vstack([extract_features(image, box) for box in boxes])
        probabilities = classifier.predict_proba(matrix)
        detections: list[tuple[tuple[int, int, int, int], float, int]] = []
        for box, row in zip(boxes, probabilities):
            row = np.asarray(row, dtype=np.float32)
            if background_index >= 0:
                row[background_index] = 0.0
            best_index = int(np.argmax(row))
            score = float(row[best_index])
            category_id = int(class_order[best_index])
            if category_id != BACKGROUND_CLASS:
                detections.append((box, score, category_id))
        if len(detections) > max_scored_candidates:
            detections = sorted(detections, key=lambda item: item[1], reverse=True)[:max_scored_candidates]
        predictions[record.path] = detections
    return predictions


def threshold_detections(
    detections: list[tuple[tuple[int, int, int, int], float, int]],
    confidence_threshold: float,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    return nms_boxes(
        [
            (box, score, category_id)
            for box, score, category_id in detections
            if score >= confidence_threshold
        ],
        iou_threshold=0.45,
    )


def evaluate_predictions(
    records: list[ImageRecord],
    predictions: dict[Path, list[tuple[tuple[int, int, int, int], float, int]]],
    class_aware: bool,
) -> dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for record in records:
        matched_gt: set[int] = set()
        detections = predictions.get(record.path, [])
        for pred_box, _, pred_class in sorted(detections, key=lambda item: item[1], reverse=True):
            best_index = -1
            best_iou = 0.0
            for index, gt in enumerate(record.boxes):
                if index in matched_gt:
                    continue
                if class_aware and pred_class != gt.category_id:
                    continue
                iou = box_iou(pred_box, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_iou >= 0.5 and best_index >= 0:
                true_positive += 1
                matched_gt.add(best_index)
            else:
                false_positive += 1
        false_negative += len(record.boxes) - len(matched_gt)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def tune_threshold(
    classifier,
    records: list[ImageRecord],
    thresholds: list[float],
    include_anchor_grid: bool,
    max_scored_candidates: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    best_threshold = thresholds[0]
    best_detection = {"f1": -1.0}
    best_class_aware = {"f1": -1.0}
    scored_predictions = score_records(classifier, records, include_anchor_grid, max_scored_candidates)
    for threshold in thresholds:
        predictions = {
            path: threshold_detections(detections, threshold)
            for path, detections in scored_predictions.items()
        }
        detection_metrics = evaluate_predictions(records, predictions, class_aware=False)
        class_metrics = evaluate_predictions(records, predictions, class_aware=True)
        if detection_metrics["f1"] > best_detection["f1"]:
            best_threshold = threshold
            best_detection = detection_metrics
            best_class_aware = class_metrics
        print(
            f"threshold={threshold:.2f} "
            f"detection_f1={detection_metrics['f1']:.4f} "
            f"class_f1={class_metrics['f1']:.4f} "
            f"precision={detection_metrics['precision']:.4f} "
            f"recall={detection_metrics['recall']:.4f}",
            flush=True,
        )
    return best_threshold, best_detection, best_class_aware


def train(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    train_records, categories = load_coco_split(data_dir, "train")
    valid_records, valid_categories = load_coco_split(data_dir, "valid")
    if categories != valid_categories:
        raise ValueError("Train and validation category definitions do not match.")

    print(f"Loaded {len(train_records)} train images and {len(valid_records)} validation images.", flush=True)
    print(f"Classes: {categories}", flush=True)

    x_train, y_train = build_training_matrix(
        train_records,
        negative_ratio=args.negative_ratio,
        max_negative_proposals_per_image=args.max_negative_proposals_per_image,
        include_anchor_grid=args.include_anchor_grid,
        random_seed=args.random_seed,
    )
    print(f"Training samples: {x_train.shape[0]} with {x_train.shape[1]} features.", flush=True)

    if args.classifier == "random-forest":
        classifier = RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_seed,
        )
    else:
        classifier = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=args.alpha,
                class_weight="balanced",
                max_iter=args.max_iter,
                n_jobs=-1,
                random_state=args.random_seed,
                tol=1e-4,
            ),
        )
    classifier.fit(x_train, y_train)

    thresholds = [round(value, 2) for value in np.arange(args.min_threshold, args.max_threshold + 0.001, 0.05)]
    threshold, detection_metrics, class_metrics = tune_threshold(
        classifier,
        valid_records,
        thresholds,
        include_anchor_grid=args.include_anchor_grid,
        max_scored_candidates=args.max_scored_candidates,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_package = {
        "model_name": "speech_bubble_detection_model",
        "categories": categories,
        "max_training_image_width": max(record.width for record in train_records),
        "max_training_image_height": max(record.height for record in train_records),
        "confidence_threshold": threshold,
        "include_anchor_grid": args.include_anchor_grid,
        "max_scored_candidates": args.max_scored_candidates,
        "classifier": classifier,
        "feature_count": int(x_train.shape[1]),
        "validation_detection_metrics": detection_metrics,
        "validation_class_aware_metrics": class_metrics,
    }
    with output_path.open("wb") as file:
        pickle.dump(model_package, file)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model_path": str(output_path),
                "best_threshold": threshold,
                "validation_detection_metrics": detection_metrics,
                "validation_class_aware_metrics": class_metrics,
                "target_f1": args.target_f1,
                "target_reached": detection_metrics["f1"] > args.target_f1,
                "categories": categories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model: {output_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(f"Best validation detection F1: {detection_metrics['f1']:.4f}", flush=True)
    print(f"Best validation class-aware F1: {class_metrics['f1']:.4f}", flush=True)
    if detection_metrics["f1"] > args.target_f1:
        print(f"Stop condition reached: F1 {detection_metrics['f1']:.4f} exceeds {args.target_f1}.", flush=True)
        return 0
    print(f"Stop condition not reached: F1 {detection_metrics['f1']:.4f} does not exceed {args.target_f1}.", flush=True)
    return 2


def load_model(model_path: Path) -> dict:
    with model_path.open("rb") as file:
        return pickle.load(file)


def predict_image(
    model: dict,
    image_path: Path,
) -> list[dict[str, object]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    classifier = model["classifier"]
    threshold = float(model["confidence_threshold"])
    include_anchor_grid = bool(model.get("include_anchor_grid", False))
    max_scored_candidates = int(model.get("max_scored_candidates", 700))
    categories = {int(key): value for key, value in model["categories"].items()}
    class_order = get_classifier_classes(classifier)
    background_index = class_order.index(BACKGROUND_CLASS) if BACKGROUND_CLASS in class_order else -1

    boxes = propose_boxes(image, include_anchor_grid=include_anchor_grid)
    if not boxes:
        return []
    matrix = np.vstack([extract_features(image, box) for box in boxes])
    probabilities = classifier.predict_proba(matrix)
    detections: list[tuple[tuple[int, int, int, int], float, int]] = []
    for box, row in zip(boxes, probabilities):
        row = np.asarray(row, dtype=np.float32)
        if background_index >= 0:
            row[background_index] = 0.0
        best_index = int(np.argmax(row))
        score = float(row[best_index])
        category_id = int(class_order[best_index])
        if category_id != BACKGROUND_CLASS and score >= threshold:
            detections.append((box, score, category_id))
    if len(detections) > max_scored_candidates:
        detections = sorted(detections, key=lambda item: item[1], reverse=True)[:max_scored_candidates]
    return [
        {
            "bbox_xyxy": [int(value) for value in box],
            "confidence": float(score),
            "category_id": int(category_id),
            "category_name": categories.get(int(category_id), str(category_id)),
        }
        for box, score, category_id in nms_boxes(detections, iou_threshold=0.45)
    ]


def get_classifier_classes(classifier) -> list[int]:
    if hasattr(classifier, "classes_"):
        return [int(value) for value in classifier.classes_]
    if hasattr(classifier, "steps"):
        return [int(value) for value in classifier.steps[-1][1].classes_]
    raise AttributeError("Classifier does not expose classes_.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the speech bubble detection model.")
    parser.add_argument(
        "--data-dir",
        default="training_data/speech-bubbles-detection",
        help="Path to the Roboflow COCO speech-bubble dataset.",
    )
    parser.add_argument(
        "--output",
        default="models/speech_bubble_detection_model.pkl",
        help="Where to save the trained model package.",
    )
    parser.add_argument("--target-f1", type=float, default=0.9)
    parser.add_argument("--trees", type=int, default=220)
    parser.add_argument("--max-depth", type=int, default=22)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--classifier", choices=("sgd", "random-forest"), default="sgd")
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--include-anchor-grid", action="store_true", default=True)
    parser.add_argument("--max-scored-candidates", type=int, default=700)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--max-negative-proposals-per-image", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--min-threshold", type=float, default=0.20)
    parser.add_argument("--max-threshold", type=float, default=0.80)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train(parse_args()))
