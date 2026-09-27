"""Audit completed calibration journals by source category and fixed GT geometry."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_scripts.compare_bubble_segments import (
    aggregate, assert_fields, box_iou, digest, float32, metrics, read_json, read_rows,
    require, validate_matches, verify_row,
)

DEFAULT_CATEGORIES=ROOT/"training_data/speech-bubbles-detection-yolo/speech_bubbles.yaml"
SIZE_BINS=("lt16", "16_to_lt24", "24_to_lt32", "ge32")
ASPECT_BINS=("lt1", "1_to_3", "gt3")
PAGE_BINS=("le1024", "1025_to_2048", "gt2048")


def artifact_path(value, base):
    path=Path(value)
    return (path if path.is_absolute() else base/path).resolve(strict=True)


def category_names(segmentation, records, category_yaml):
    if segmentation:
        require(all(str(row["image_id"]).startswith("ai4va:") for row in records),
                "Segmentation analysis expects published AI4VA bubble labels")
        return {5:"26 Comic Bubble"}, None
    path=Path(category_yaml).resolve(strict=True)
    names=yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    if isinstance(names, list):
        names=dict(enumerate(names))
    require(set(names)==set(range(6)), "Expected six zero-based source YAML category names")
    return {key+1:f"{key+1} {name}" for key, name in names.items()}, {"path":str(path), "sha256":digest(path)}


def geometry_groups(annotation, record, names):
    width, height=record["width"], record["height"]
    require(type(width) is int and type(height) is int and min(width, height)>0, "Invalid source dimensions")
    category=annotation["category_id"]
    require(category in names and not annotation.get("iscrowd", 0), "Unsupported source category or crowd")
    bbox=annotation["bbox"]
    require(isinstance(bbox, list) and len(bbox)==4 and all(type(v) in (int, float) and math.isfinite(v) for v in bbox),
            "Invalid ground-truth box")
    x, y, w, h=bbox
    require(min(x, y)>=0 and min(w, h)>0 and x+w<=width+1 and y+h<=height+1, "Invalid ground-truth bounds")
    short=min(w, h)*768/max(width, height)
    aspect=w/h
    longside=max(width, height)
    return {"category":names[category],
            "short_side_at_768":SIZE_BINS[0 if short<16 else 1 if short<24 else 2 if short<32 else 3],
            "width_height_ratio":ASPECT_BINS[0 if aspect<1 else 1 if aspect<=3 else 2],
            "source_page_long_side":PAGE_BINS[0 if longside<=1024 else 1 if longside<=2048 else 2]}


def recompute_box_matches(annotations, predictions, threshold):
    """Repeat the evaluator's global IoU-sort/unique-prediction/unique-target rule."""
    matrix=np.array([[box_iou(annotation, prediction) for prediction in predictions]
                     for annotation in annotations], dtype=np.float64).reshape(len(annotations), len(predictions))
    pairs=np.array(np.nonzero(matrix>=threshold)).T
    if len(pairs)>1:
        pairs=pairs[matrix[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs=pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs=pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return [{"target_index":int(ti), "prediction_index":int(pi),
             "annotation_id":annotations[ti]["id"], "iou":float(matrix[ti, pi])} for ti, pi in pairs]


def check_box_assignment(saved, annotations, predictions, threshold):
    expected=recompute_box_matches(annotations, predictions, threshold)
    require(len(saved)==len(expected), "Journal does not contain the global box assignment")
    observed=sorted(saved, key=lambda pair:(pair["target_index"], pair["prediction_index"]))
    expected=sorted(expected, key=lambda pair:(pair["target_index"], pair["prediction_index"]))
    for left, right in zip(observed, expected):
        assert_fields(left, right, "global_box_match")


def verify_box_row(row, record, index, identity_hash):
    for key in ("image_path", "source_sha256"):
        require(row[key]==record[key], f"Journal coverage differs at image {index}: {key}")
    require(row["image_id"]==record["source_image"], "Box journal source identity differs")
    require(row["index"]==index and row["identity_sha256"]==identity_hash, "Journal index/identity differs")
    annotations=record["annotations"]
    require(len({annotation["id"] for annotation in annotations})==len(annotations), "Duplicate annotation ID")
    predictions=row["predictions"]
    for prediction in predictions:
        confidence=prediction["confidence"]
        require(type(confidence) in (int, float) and math.isfinite(confidence) and float32(0.35)<=confidence<=1,
                "Journal prediction violates fixed confidence")
        require(len(prediction["bbox"])==4 and all(math.isfinite(value) for value in prediction["bbox"]),
                "Invalid predicted box")
    validate_matches(row["matches"], annotations, predictions, 0.5, "boxes")
    check_box_assignment(row["matches"], annotations, predictions, 0.5)
    tp=len(row["matches"])
    expected=metrics(tp, len(predictions)-tp, len(annotations)-tp)
    assert_fields(row, {"targets":len(annotations), "boxes":expected}, "journal")
    return {"targets":len(annotations), "boxes":expected}


def verified_inputs(report_path, manifest_path):
    report_path=Path(report_path).resolve(strict=True)
    manifest_path=Path(manifest_path).resolve(strict=True)
    report=read_json(report_path)
    require(report.get("complete") is True and report.get("status")=="complete", "Incomplete evaluation report")
    identity_path=report_path.with_suffix(".identity.json")
    require(digest(identity_path)==report["identity_sha256"], "Identity sidecar hash mismatch")
    identity=read_json(identity_path)
    require(identity==report["identity"] and identity.get("limit") is None, "Identity differs or evaluation is limited")
    recorded_manifest=identity["manifest"]
    require(artifact_path(recorded_manifest["path"], ROOT)==manifest_path, "Manifest path differs from report identity")
    require(digest(manifest_path)==recorded_manifest["sha256"], "Manifest hash mismatch")
    journal=artifact_path(report["predictions_jsonl"], report_path.parent)
    require(digest(journal)==report["predictions_sha256"], "Prediction journal hash mismatch")
    records, rows=read_rows(manifest_path), read_rows(journal)
    require(bool(records) and len(records)==len(rows)==recorded_manifest["images"]==report["images"],
            "Journal does not cover every manifest page")
    segmentation=isinstance(identity["match_iou"], list)
    thresholds=identity["match_iou"] if segmentation else [identity["match_iou"]]
    require(thresholds and 0.5 in thresholds and len(set(thresholds))==len(thresholds)
            and all(value in (0.5, 0.75) for value in thresholds), "Unsupported matching thresholds")
    require(segmentation or thresholds==[0.5], "Box journals currently provide only IoU0.50")
    settings=identity["settings"]
    assert_fields(settings, {"conf":0.35, "iou":0.5, "half":False, "rect":False,
                             "retina_masks":True, "agnostic_nms":True, "max_det":300}, "settings")
    require(type(settings["imgsz"]) is int and settings["imgsz"]>0, "Invalid inference resolution")
    require(re.fullmatch(r"[0-9a-f]{64}", identity["model"]["sha256"]) is not None, "Missing model identity hash")
    identities=[str(record["image_id"] if segmentation else record["source_image"]) for record in records]
    require(len(set(identities))==len(identities), "Duplicate manifest image identity")
    ordered=hashlib.sha256("".join(value+"\n" for value in identities).encode()).hexdigest()
    ordered_key="ordered_image_ids_sha256" if segmentation else "ordered_source_images_sha256"
    require(recorded_manifest[ordered_key]==ordered, "Ordered manifest identity hash differs")
    summaries=[]
    for index, (row, record) in enumerate(zip(rows, records)):
        require(re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"]) is not None, "Invalid source image hash")
        if segmentation:
            summaries.append(verify_row(row, record, index, report["identity_sha256"], thresholds))
            for threshold in thresholds:
                check_box_assignment(row["matches"][f"{threshold:.2f}"]["boxes"], record["annotations"], row["predictions"], threshold)
        else:
            summaries.append(verify_box_row(row, record, index, report["identity_sha256"]))
    if segmentation:
        assert_fields(report, aggregate(summaries, thresholds), "report")
    else:
        counts=metrics(*(sum(row["boxes"][key] for row in summaries) for key in ("tp", "fp", "fn")))
        assert_fields(report, {"targets":sum(row["targets"] for row in summaries), "boxes":counts}, "report")
    evidence={"report":{"path":str(report_path), "sha256":digest(report_path)},
              "identity":{"path":str(identity_path), "sha256":report["identity_sha256"]},
              "manifest":{"path":str(manifest_path), "sha256":recorded_manifest["sha256"]},
              "journal":{"path":str(journal), "sha256":report["predictions_sha256"]}}
    return report, records, rows, segmentation, thresholds, evidence


def summarize_group(values):
    support=len(values)
    matched=[value for value in values if value is not None]
    return {"support":support, "tp":len(matched), "fn":support-len(matched),
            "recall":len(matched)/support if support else None,
            "mean_matched_iou":sum(matched)/len(matched) if matched else None}


def analyze(report_path, manifest_path, output=None, category_yaml=DEFAULT_CATEGORIES):
    report, records, rows, segmentation, thresholds, evidence=verified_inputs(report_path, manifest_path)
    names, category_source=category_names(segmentation, records, category_yaml)
    memberships=[]
    for record in records:
        ident=record["image_id"] if segmentation else record["source_image"]
        for annotation in record["annotations"]:
            memberships.append({"image_id":ident, "annotation_id":annotation["id"],
                                **geometry_groups(annotation, record, names)})
    membership_hash=hashlib.sha256(json.dumps(memberships, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    labels={"category":list(names.values()), "short_side_at_768":SIZE_BINS,
            "width_height_ratio":ASPECT_BINS, "source_page_long_side":PAGE_BINS}
    by_iou={}
    for threshold in thresholds:
        key=f"{threshold:.2f}"
        by_iou[key]={}
        for kind in (("boxes", "masks") if segmentation else ("boxes",)):
            grouped={axis:{name:[] for name in choices} for axis, choices in labels.items()}
            overall=[]
            fp=ignored=missing=predictions=0
            for record, row in zip(records, rows):
                pairs=row["matches"][key][kind] if segmentation else row["matches"]
                matched={pair["target_index"]:pair["iou"] for pair in pairs}
                counts=row["counts"][key][kind] if segmentation else row["boxes"]
                fp+=counts["fp"]
                predictions+=len(row["predictions"])
                ignored+=len(row["matches"][key]["ignored_masks"]) if kind=="masks" else 0
                for index, annotation in enumerate(record["annotations"]):
                    if kind=="masks" and annotation.get("mask_status")=="missing_source_polygon":
                        missing+=1
                        continue
                    value=matched.get(index)
                    overall.append(value)
                    for axis, label in geometry_groups(annotation, record, names).items():
                        grouped[axis][label].append(value)
            axes={}
            for axis, groups in grouped.items():
                results={name:summarize_group(values) for name, values in groups.items()}
                recalls=[result["recall"] for result in results.values() if result["support"]]
                axes[axis]={"groups":results, "macro_recall":sum(recalls)/len(recalls) if recalls else None,
                            "supported_groups":len(recalls)}
            by_iou[key][kind]={"global":{**summarize_group(overall), "predictions":predictions,
                                         "unmatched_false_positives":fp, "ignored_unknown_mask_predictions":ignored,
                                         "excluded_unknown_mask_targets":missing}, "by_group":axes}
    result={"status":"verified", "usage":"calibration_diagnostics_only", "images":len(records),
            "targets":len(memberships), "inference_imgsz":report["identity"]["settings"]["imgsz"],
            "model":report["identity"]["model"], "settings":report["identity"]["settings"],
            "evidence":evidence, "category_source":category_source,
            "membership_sha256":membership_hash, "by_iou":by_iou,
            "group_definitions":{"category":"Original ground-truth labels; all model outputs remain single-class bubbles.",
                                 "short_side_at_768":"min(bbox_width,bbox_height)*768/max(source_width,source_height); fixed across inference sizes",
                                 "width_height_ratio":"bbox_width/bbox_height; intervals <1, [1,3], >3",
                                 "source_page_long_side":"max(source_width,source_height); intervals <=1024, (1024,2048], >2048",
                                 "macro_recall":"Unweighted mean of recalls for groups with positive support; empty groups excluded"},
            "limitations":["These are calibration diagnostics, not independent final-test acceptance.",
                           "GT group memberships never depend on actual inference resolution or predicted box size.",
                           "All matches are globally one-to-one on each page before grouping; categories are never matched separately.",
                           "Single-class predictions cannot assign unmatched false positives to true shape categories; no per-category precision or F1 is reported.",
                           "AI4VA has one published bubble class and no supplied semantic bubble-subtype labels.",
                           "Mean matched IoU describes detected targets only; always read it with support and recall.",
                           "Mask pairs/counts are verified from hash-bound journals; prediction mask pixels are unavailable for independent mask rematching.",
                           "Image bytes and historical evaluator source files are not rehashed here; original completed evaluations recorded their verification.",
                           "Box-only journals supply IoU0.50 only; IoU0.75 is shown only when present in segmentation journals."]}
    if output is not None:
        path=Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--category-yaml", default=str(DEFAULT_CATEGORIES))
    args=parser.parse_args()
    result=analyze(args.report, args.manifest, args.output, args.category_yaml)
    print(json.dumps({"status":result["status"], "images":result["images"],
                      "targets":result["targets"], "membership_sha256":result["membership_sha256"]}, indent=2))
