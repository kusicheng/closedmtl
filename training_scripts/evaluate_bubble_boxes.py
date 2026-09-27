"""Measure deployed segmentation models on original box-only bubble labels.

This is a development regression check. It maps all six shape categories to
bubbles, preserves source box coordinates, and reports no mask quality score.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"models/ultralytics_config"))

from training_scripts.bubble_metrics import count_metrics
from training_scripts.evaluate_bubble_segments import (
    box_ious, digest, load_model, manifest_rows, match_instances, write_json,
)


def source_boxes(record):
    """Validate source xywh boxes, preserving documented one-pixel overhangs."""
    width, height=record["width"], record["height"]
    if any(type(value) is not int or value<=0 for value in (width, height)):
        raise ValueError("Image dimensions must be positive integers")
    annotations=record["annotations"]
    if not isinstance(annotations, list):
        raise ValueError("Expected an annotation list")
    ids=set()
    image_ids=set()
    boxes=[]
    overhangs=[]
    for annotation in annotations:
        if annotation["id"] in ids:
            raise ValueError("Duplicate annotation ID on one page")
        ids.add(annotation["id"])
        image_ids.add(annotation["image_id"])
        category=annotation["category_id"]
        if type(category) is not int or category not in range(1, 7) or annotation.get("iscrowd", 0):
            raise ValueError("Expected an individual bubble with source category 1 through 6")
        bbox=annotation["bbox"]
        if not isinstance(bbox, list) or len(bbox)!=4:
            raise ValueError("Expected a four-value COCO xywh box")
        if any(type(value) not in (int, float) for value in bbox) or not np.isfinite(bbox).all():
            raise ValueError("Source box coordinates must be finite numbers")
        x, y, w, h=bbox
        if min(x, y)<0 or min(w, h)<=0 or x+w>width+1 or y+h>height+1:
            raise ValueError("Source box exceeds the declared one-pixel tolerance")
        boxes.append([x, y, x+w, y+h])
        if x+w>width or y+h>height:
            overhangs.append({"annotation_id":annotation["id"],
                             "right_pixels":max(0, x+w-width), "bottom_pixels":max(0, y+h-height)})
    if len(image_ids)>1:
        raise ValueError("One page references multiple source image IDs")
    return boxes, overhangs


def inspect_manifest(path):
    images=set()
    sources=set()
    annotation_ids=set()
    ordered=hashlib.sha256()
    targets=0
    for record in manifest_rows(path):
        image=record["image_path"]
        source=record["source_image"]
        if not image or not source or image in images or source in sources:
            raise ValueError("Duplicate or missing manifest image/source identity")
        if not isinstance(record["source_sha256"], str) or len(record["source_sha256"])!=64:
            raise ValueError("Missing image SHA256")
        boxes, _=source_boxes(record)
        for annotation in record["annotations"]:
            if annotation["id"] in annotation_ids:
                raise ValueError("Duplicate annotation ID across manifest pages")
            annotation_ids.add(annotation["id"])
        images.add(image)
        sources.add(source)
        ordered.update((source+"\n").encode())
        targets+=len(boxes)
    if not images:
        raise ValueError("Empty box evaluation manifest")
    return {"path":str(Path(path).resolve()), "sha256":digest(path), "images":len(images),
            "targets":targets, "ordered_source_images_sha256":ordered.hexdigest()}


def deployed_boxes(model, image, settings):
    """Read boxes after the segmentation predictor has removed empty masks."""
    result=model.predict(image, **settings)[0]
    boxes=result.boxes.xyxy.cpu().numpy()
    scores=result.boxes.conf.cpu().numpy()
    if len(boxes)!=len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid deployed box predictions")
    if len(boxes):
        if result.masks is None or len(result.masks.data)!=len(boxes):
            raise ValueError("Deployed segmentation boxes lack corresponding instance masks")
        if tuple(result.masks.data.shape[1:])!=tuple(image.shape[:2]):
            raise ValueError("Expected original-resolution deployed masks")
    return [{"bbox":box.tolist(), "confidence":float(score)} for box, score in zip(boxes, scores)
            if np.float32(score)>=np.float32(0.35)]


def score_image(record, predictions):
    boxes, overhangs=source_boxes(record)
    matches=match_instances(box_ious(boxes, [prediction["bbox"] for prediction in predictions]), 0.5)
    for pair in matches:
        pair["annotation_id"]=record["annotations"][pair["target_index"]]["id"]
    tp=len(matches)
    return {"image_id":record["source_image"], "image_path":record["image_path"],
            "source_sha256":record["source_sha256"], "targets":len(boxes),
            "boxes":count_metrics(tp, len(predictions)-tp, len(boxes)-tp),
            "predictions":predictions, "matches":matches, "source_box_overhangs":overhangs}


def evaluate(args, model=None):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    started=time.perf_counter()
    manifest=Path(args.manifest).resolve(strict=True)
    model_path=Path(args.model).resolve(strict=True)
    output=Path(args.output).resolve()
    journal=output.with_suffix(".predictions.jsonl")
    identity_path=output.with_suffix(".identity.json")
    temporary=output.with_suffix(".json.tmp")
    paths=(output, journal, identity_path, temporary)
    if len(set(paths))!=4 or any(path.exists() for path in paths):
        raise FileExistsError("Use fresh, distinct output report and evidence paths")
    if args.limit is not None and args.limit<=0:
        raise ValueError("--limit must be positive")
    imgsz=getattr(args, "imgsz", 768)
    if type(imgsz) is not int or imgsz<=0 or imgsz%32:
        raise ValueError("Image size must be a positive multiple of32")
    settings={"imgsz":imgsz, "device":args.device, "half":False, "rect":False,
              "retina_masks":True, "conf":0.35, "iou":0.5, "agnostic_nms":True,
              "max_det":300, "verbose":False, "save":False}
    manifest_identity=inspect_manifest(manifest)
    scripts=[Path(__file__)]+[ROOT/"training_scripts"/name for name in
                             ("evaluate_bubble_segments.py", "bubble_metrics.py", "bubble_models.py",
                              "prepare_bubble_segments.py")]
    import ultralytics
    identity={"manifest":manifest_identity,
              "model":{"path":str(model_path), "sha256":digest(model_path), "mayocream":args.mayocream},
              "source_sha256":{str(path.resolve()):digest(path) for path in scripts},
              "settings":settings, "match_iou":0.5, "limit":args.limit,
              "versions":{"ultralytics":ultralytics.__version__, "torch":torch.__version__, "numpy":np.__version__},
              "class_mapping":"Source categories 1 through 6 all map to bubble.",
              "box_policy":"Original COCO xywh converted to half-open xyxy without clipping. "
                           "Right/bottom overhang up to one pixel is preserved and individually reported.",
              "prediction_policy":"Actual segmentation predictor outputs, including its empty-mask filtering."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(identity_path, identity)
    identity_hash=digest(identity_path)
    model=model if model is not None else load_model(model_path, args.mayocream)
    totals={"tp":0, "fp":0, "fn":0}
    images=0
    targets=0
    negatives=0
    overhangs=0
    with journal.open("x", encoding="utf-8", buffering=1) as stream:
        for index, record in enumerate(manifest_rows(manifest)):
            if args.limit is not None and index>=args.limit:
                break
            image_path=Path(record["image_path"])
            if not image_path.is_absolute():
                image_path=manifest.parent/image_path
            raw=image_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=record["source_sha256"]:
                raise ValueError(f"Image SHA256 differs: {image_path}")
            image=cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2]!=(record["height"], record["width"]):
                raise ValueError(f"Image dimensions differ: {image_path}")
            row=score_image(record, deployed_boxes(model, image, settings))
            row.update({"index":index, "identity_sha256":identity_hash})
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            for key in totals:
                totals[key]+=row["boxes"][key]
            images+=1
            targets+=row["targets"]
            negatives+=int(row["targets"]==0)
            overhangs+=len(row["source_box_overhangs"])
            if images%20==0:
                os.fsync(stream.fileno())
                print(f"EVALUATED BOXES {images}/{manifest_identity['images']}", flush=True)
        os.fsync(stream.fileno())
    if digest(manifest)!=manifest_identity["sha256"] or digest(model_path)!=identity["model"]["sha256"]:
        raise ValueError("Manifest or model changed during evaluation")
    if any(digest(path)!=expected for path, expected in identity["source_sha256"].items()):
        raise ValueError("Evaluator source changed during evaluation")
    complete=args.limit is None
    if complete and (images!=manifest_identity["images"] or targets!=manifest_identity["targets"]):
        raise ValueError("Evaluation did not cover every manifest page and target")
    report={"complete":complete, "status":"complete" if complete else "smoke_incomplete",
            "usage":"development_regression_only", "acceptance_eligible":False,
            "acceptance_reason":"Current dataset test does not establish independent model or book exposure.",
            "images":images, "targets":targets, "negative_images":negatives, "valid":targets>0,
            "boxes":count_metrics(**totals), "source_boxes_with_overhang":overhangs,
            "mask_evaluation":"Unavailable: this manifest contains box-only ground truth.",
            "identity":identity, "identity_sha256":identity_hash,
            "predictions_jsonl":str(journal), "predictions_sha256":digest(journal),
            "seconds":time.perf_counter()-started}
    write_json(temporary, report)
    temporary.rename(output)
    print(json.dumps({"output":str(output), "complete":complete, "boxes":report["boxes"]}), flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"training_data/bubble_joint_20260922/current_test_boxes.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mayocream", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--limit", type=int, help="Smoke only; incomplete coverage is explicitly marked")
    evaluate(parser.parse_args())
