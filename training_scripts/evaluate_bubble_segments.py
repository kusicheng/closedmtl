"""Evaluate deployed bubble masks and boxes against original, exact COCO RLE.

The manifest includes every labeled evaluation page, including pages rejected
by polygon conversion. Empty annotations are evaluated as supplied negatives.
Output paths must be fresh; partial journals are evidence, never resumable runs.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"models/ultralytics_config"))

from training_scripts.bubble_metrics import count_metrics
from training_scripts.prepare_bubble_segments import cropped_mask, digest


def crop_mask(mask, origin=(0, 0)):
    """Keep one tight binary crop rather than a separate full page per mask."""
    mask=np.asarray(mask)>0.5
    if mask.ndim!=2:
        raise ValueError("Expected a two-dimensional instance mask")
    ys=np.flatnonzero(mask.any(axis=1))
    xs=np.flatnonzero(mask.any(axis=0))
    if not len(xs):
        return {"bbox":[0, 0, 0, 0], "pixels":np.zeros((0, 0), dtype=bool), "area":0}
    x1, x2=int(xs[0]), int(xs[-1])+1
    y1, y2=int(ys[0]), int(ys[-1])+1
    pixels=mask[y1:y2, x1:x2].copy()
    return {"bbox":[x1+origin[0], y1+origin[1], x2+origin[0], y2+origin[1]],
            "pixels":pixels, "area":int(np.count_nonzero(pixels))}


def box_ious(targets, predictions):
    targets=np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    predictions=np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    intersection=np.maximum(0, np.minimum(targets[:, None, 2:], predictions[None, :, 2:])-
                            np.maximum(targets[:, None, :2], predictions[None, :, :2])).prod(axis=2)
    target_area=np.maximum(0, targets[:, 2:]-targets[:, :2]).prod(axis=1)
    prediction_area=np.maximum(0, predictions[:, 2:]-predictions[:, :2]).prod(axis=1)
    union=target_area[:, None]+prediction_area[None, :]-intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union>0)


def mask_ious(targets, predictions):
    """Intersect only overlapping cropped rectangles, one mask pair at a time."""
    result=np.zeros((len(targets), len(predictions)), dtype=np.float64)
    for ti, target in enumerate(targets):
        tx1, ty1, tx2, ty2=target["bbox"]
        for pi, prediction in enumerate(predictions):
            px1, py1, px2, py2=prediction["bbox"]
            x1, y1=max(tx1, px1), max(ty1, py1)
            x2, y2=min(tx2, px2), min(ty2, py2)
            if x2<=x1 or y2<=y1:
                continue
            left=target["pixels"][y1-ty1:y2-ty1, x1-tx1:x2-tx1]
            right=prediction["pixels"][y1-py1:y2-py1, x1-px1:x2-px1]
            intersection=int(np.count_nonzero(left&right))
            union=target["area"]+prediction["area"]-intersection
            result[ti, pi]=intersection/union if union else 0.0
    return result


def match_instances(ious, threshold):
    """Use Ultralytics' IoU-sort, unique-prediction, unique-target matching."""
    pairs=np.array(np.nonzero(ious>=threshold)).T
    if len(pairs)>1:
        pairs=pairs[ious[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs=pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs=pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return [{"target_index":int(ti), "prediction_index":int(pi), "iou":float(ious[ti, pi])}
            for ti, pi in pairs]


def score_image(record, predictions, thresholds=(0.5,)):
    """Score one supplied page, preserving negative pages and mask/box differences."""
    annotations=record["annotations"]
    if len({annotation["id"] for annotation in annotations})!=len(annotations):
        raise ValueError("Repeated annotation ID on one image")
    targets=[]
    known_indices=[]
    missing_indices=set()
    boxes=[]
    for index, annotation in enumerate(annotations):
        if annotation["category_id"]!=5 or annotation.get("iscrowd", 0):
            raise ValueError("Expected individual category-5 balloon annotations")
        if annotation.get("image_id", record["image_id"])!=record["image_id"]:
            raise ValueError("Annotation refers to another image")
        x, y, w, h=annotation["bbox"]
        if not np.isfinite([x, y, w, h]).all() or min(x, y)<0 or min(w, h)<=0:
            raise ValueError("Invalid annotation bounding box")
        if x+w>record["width"] or y+h>record["height"]:
            raise ValueError("Annotation box exceeds image bounds")
        boxes.append([x, y, x+w, y+h])
        missing=annotation.get("mask_status")=="missing_source_polygon"
        if annotation.get("segmentation") is None:
            if not missing:
                raise ValueError("Absent mask requires explicit missing_source_polygon status")
            missing_indices.add(index)
            continue
        if missing:
            raise ValueError("Missing-source status conflicts with supplied segmentation")
        mask=cropped_mask(annotation, record)
        known_indices.append(index)
        targets.append(crop_mask(mask, (x, y)))
    predictions=[prediction for prediction in predictions
                 if np.float32(prediction["confidence"])>=np.float32(0.35)]
    ious={"boxes":box_ious(boxes, [prediction["bbox"] for prediction in predictions]),
          "masks":mask_ious(targets, [prediction["mask"] for prediction in predictions])}
    counts={}
    matches={}
    for threshold in thresholds:
        key=f"{threshold:.2f}"
        box_matches=match_instances(ious["boxes"], threshold)
        mask_matches=match_instances(ious["masks"], threshold)
        for pair in mask_matches:
            pair["target_index"]=known_indices[pair["target_index"]]
        for matched in (box_matches, mask_matches):
            for pair in matched:
                pair["annotation_id"]=annotations[pair["target_index"]]["id"]
        known_predictions={pair["prediction_index"] for pair in mask_matches}
        ignored=[pair for pair in box_matches if pair["target_index"] in missing_indices
                 and pair["prediction_index"] not in known_predictions]
        box_tp=len(box_matches)
        mask_tp=len(mask_matches)
        conservative=count_metrics(mask_tp, len(predictions)-mask_tp, len(annotations)-mask_tp)
        counts[key]={"boxes":count_metrics(box_tp, len(predictions)-box_tp, len(annotations)-box_tp),
                     "masks":count_metrics(mask_tp, len(predictions)-mask_tp-len(ignored), len(targets)-mask_tp),
                     "masks_conservative":conservative, "mask_f1_lower_bound":conservative["f1"],
                     "known_mask_targets":len(targets), "missing_mask_targets":len(missing_indices),
                     "ignored_mask_predictions":len(ignored)}
        matches[key]={"boxes":box_matches, "masks":mask_matches, "ignored_masks":ignored}
    return {"image_id":record["image_id"], "book":record["book"],
            "image_path":record["image_path"], "source_sha256":record["source_sha256"],
            "targets":len(annotations), "known_mask_targets":len(targets),
            "missing_mask_targets":len(missing_indices),
            "ignored_mask_predictions":counts["0.50"]["ignored_mask_predictions"],
            "mask_f1_lower_bound":counts["0.50"]["mask_f1_lower_bound"],
            "counts":counts, "matches":matches,
            "training_polygon_rejected":record.get("training_polygon_rejected", False),
            "predictions":[{"bbox":prediction["bbox"], "confidence":prediction["confidence"],
                            "mask_bbox":prediction["mask"]["bbox"],
                            "mask_area":prediction["mask"]["area"],
                            "mask_crop_sha256":hashlib.sha256(prediction["mask"]["pixels"].tobytes()).hexdigest()}
                           for prediction in predictions]}


def aggregate_rows(rows, thresholds=(0.5,)):
    report={"images":len(rows), "targets":sum(row["targets"] for row in rows),
            "known_mask_targets":sum(row["known_mask_targets"] for row in rows),
            "missing_mask_targets":sum(row["missing_mask_targets"] for row in rows),
            "negative_images":sum(row["targets"]==0 for row in rows),
            "polygon_rejected_images":sum(row["training_polygon_rejected"] for row in rows),
            "by_iou":{}}
    for threshold in thresholds:
        key=f"{threshold:.2f}"
        report["by_iou"][key]={}
        for kind in ("boxes", "masks", "masks_conservative"):
            counts={name:sum(row["counts"][key][kind][name] for row in rows)
                    for name in ("tp", "fp", "fn")}
            report["by_iou"][key][kind]=count_metrics(**counts)
        for name in ("known_mask_targets", "missing_mask_targets", "ignored_mask_predictions"):
            report["by_iou"][key][name]=sum(row["counts"][key][name] for row in rows)
        report["by_iou"][key]["mask_f1_lower_bound"]=report["by_iou"][key]["masks_conservative"]["f1"]
    report.update(report["by_iou"]["0.50"])
    report["valid"]=report["targets"]>0
    report["mask_valid"]=report["known_mask_targets"]>0
    return report


def manifest_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def inspect_manifest(path):
    identities=set()
    ordered=hashlib.sha256()
    for row in manifest_rows(path):
        key=str(row["image_id"])
        if key in identities:
            raise ValueError(f"Repeated manifest image_id: {key}")
        identities.add(key)
        if min(row["width"], row["height"])<=0 or not isinstance(row["annotations"], list):
            raise ValueError("Invalid manifest dimensions or annotations")
        for name in ("book", "image_path", "source_sha256"):
            if not row.get(name):
                raise ValueError(f"Missing manifest field: {name}")
        ordered.update((key+"\n").encode())
    if not identities:
        raise ValueError("Empty evaluation manifest")
    return {"path":str(Path(path).resolve()), "sha256":digest(path),
            "images":len(identities), "ordered_image_ids_sha256":ordered.hexdigest()}


def load_model(path, mayocream=False):
    from ultralytics import YOLO
    if mayocream:
        from training_scripts.bubble_models import load_mayocream
        network, _=load_mayocream(path)
        model=YOLO("yolo11n-seg.yaml", task="segment")
        model.model=network.eval()
        model.overrides.update({"task":"segment"})
    else:
        model=YOLO(str(path), task="segment")
    if model.task!="segment" or len(model.names)!=1:
        raise ValueError("Expected a one-class bubble segmentation model")
    return model


def deployed_predictions(model, image, settings):
    """Retain exactly the boxes/masks returned by deployed segmentation inference."""
    result=model.predict(image, **settings)[0]
    boxes=result.boxes.xyxy.cpu().numpy()
    scores=result.boxes.conf.cpu().numpy()
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite model prediction")
    if len(boxes) and (result.masks is None or len(result.masks.data)!=len(boxes)):
        raise ValueError("Missing predicted instance masks")
    predictions=[]
    for index, (box, score) in enumerate(zip(boxes, scores)):
        mask=result.masks.data[index].cpu().numpy()
        if tuple(mask.shape)!=tuple(image.shape[:2]):
            raise ValueError("retina_masks did not produce original-size masks")
        predictions.append({"bbox":box.tolist(), "confidence":float(score), "mask":crop_mask(mask)})
    return predictions


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(args, model=None):
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    started=time.perf_counter()
    manifest=Path(args.manifest).resolve(strict=True)
    model_path=Path(args.model).resolve(strict=True)
    output=Path(args.output).resolve()
    journal=output.with_suffix(".predictions.jsonl")
    identity_path=output.with_suffix(".identity.json")
    temporary=output.with_suffix(".json.tmp")
    if len({output, journal, identity_path, temporary})!=4:
        raise ValueError("Output must have a distinct .json report filename")
    if any(path.exists() for path in (output, journal, identity_path, temporary)):
        raise FileExistsError("Use fresh report, identity and prediction paths")
    if args.limit is not None and args.limit<=0:
        raise ValueError("--limit must be positive")
    thresholds=(0.5, 0.75) if args.iou75 else (0.5,)
    settings={"imgsz":args.imgsz, "device":args.device, "half":False, "rect":False,
              "retina_masks":True, "conf":0.35, "iou":0.5, "agnostic_nms":True,
              "max_det":300, "verbose":False, "save":False}
    manifest_identity=inspect_manifest(manifest)
    source_paths=[Path(__file__), ROOT/"training_scripts/prepare_bubble_segments.py",
                  ROOT/"training_scripts/bubble_models.py", ROOT/"training_scripts/bubble_metrics.py"]
    import ultralytics
    identity={"manifest":manifest_identity,
              "model":{"path":str(model_path), "sha256":digest(model_path), "mayocream":args.mayocream},
              "source_sha256":{str(path.resolve()):digest(path) for path in source_paths},
              "settings":settings, "match_iou":list(thresholds), "limit":args.limit,
              "versions":{"ultralytics":ultralytics.__version__, "torch":torch.__version__, "numpy":np.__version__},
              "method":"Original RLE pixels; half-open COCO boxes; independent IoU-sorted one-to-one matches. "
                       "Actual deployed segmentation outputs include Ultralytics empty-mask filtering.",
              "missing_mask_policy":"Only explicit missing_source_polygon annotations may omit segmentation. "
                                    "All boxes remain targets. Known mask matches take priority; ignore at most one "
                                    "otherwise-unmatched mask prediction per unknown target using all-GT box assignment. "
                                    "Conservative mask counts restore ignored predictions as FP and missing targets as FN."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(identity_path, identity)
    identity_sha256=digest(identity_path)
    model=model if model is not None else load_model(model_path, args.mayocream)
    summaries=[]
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
            predictions=deployed_predictions(model, image, settings)
            row=score_image(record, predictions, thresholds)
            row.update({"index":index, "identity_sha256":identity_sha256})
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            summaries.append({key:row[key] for key in ("book", "targets", "known_mask_targets",
                                                       "missing_mask_targets", "counts", "training_polygon_rejected")})
            if (index+1)%20==0:
                os.fsync(stream.fileno())
                print(f"EVALUATED {index+1}/{manifest_identity['images']}", flush=True)
        os.fsync(stream.fileno())
    if digest(manifest)!=manifest_identity["sha256"] or digest(model_path)!=identity["model"]["sha256"]:
        raise ValueError("Manifest or model changed during evaluation")
    if any(digest(path)!=expected for path, expected in identity["source_sha256"].items()):
        raise ValueError("Evaluator source changed during evaluation")
    if args.limit is None and len(summaries)!=manifest_identity["images"]:
        raise ValueError("Evaluation did not cover the complete manifest")
    report=aggregate_rows(summaries, thresholds)
    groups=defaultdict(list)
    for row in summaries:
        groups[row["book"]].append(row)
    complete=args.limit is None
    report.update({"complete":complete, "status":"complete" if complete else "smoke_incomplete",
                   "acceptance_eligible":False,
                   "acceptance_reason":"Independent dataset provenance and human label verification must be assessed separately",
                   "target_f1":0.9,
                   "development_target_reached":complete and report["mask_valid"] and
                   min(report["boxes"]["f1"], report["mask_f1_lower_bound"])>=0.9,
                   "identity":identity, "identity_sha256":identity_sha256,
                   "predictions_jsonl":str(journal), "predictions_sha256":digest(journal),
                   "books":[{"book":book, **aggregate_rows(rows, thresholds)} for book, rows in sorted(groups.items())],
                   "seconds":time.perf_counter()-started})
    write_json(temporary, report)
    temporary.rename(output)
    print(json.dumps({"output":str(output), "complete":complete,
                      "boxes":report["boxes"], "masks":report["masks"]}), flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mayocream", action="store_true", help="Load verified YOLO11n SafeTensors")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou75", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke only; always disqualifies acceptance")
    evaluate(parser.parse_args())


if __name__=="__main__":
    main()
