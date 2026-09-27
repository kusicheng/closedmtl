"""Verify complete evaluation evidence and compare the frozen AI4VA operating point."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL=ROOT/"outputs/bubble_training_20260922/external_evaluation_protocol.json"
KINDS=("boxes", "masks", "masks_conservative")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def path_from(value, base=ROOT):
    path=Path(value)
    return (path if path.is_absolute() else base/path).resolve(strict=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(tp, fp, fn):
    require(all(type(value) is int and value>=0 for value in (tp, fp, fn)), "Invalid instance counts")
    return {"tp":tp, "fp":fp, "fn":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.0,
            "recall":tp/(tp+fn) if tp+fn else 0.0,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def assert_fields(actual, expected, label):
    for key, value in expected.items():
        require(key in actual, f"Missing {label}.{key}")
        observed=actual[key]
        if isinstance(value, dict):
            require(isinstance(observed, dict), f"Invalid {label}.{key}")
            assert_fields(observed, value, f"{label}.{key}")
        elif type(value) is float:
            require(type(observed) in (int, float) and math.isfinite(observed) and
                    math.isclose(observed, value, rel_tol=1e-12, abs_tol=1e-12), f"Mismatch {label}.{key}")
        else:
            require(type(observed) is type(value) and observed==value, f"Mismatch {label}.{key}")


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def box_iou(annotation, prediction):
    x, y, w, h=annotation["bbox"]
    px1, py1, px2, py2=prediction["bbox"]
    overlap=max(0, min(x+w, px2)-max(x, px1))*max(0, min(y+h, py2)-max(y, py1))
    union=w*h+max(0, px2-px1)*max(0, py2-py1)-overlap
    return overlap/union if union else 0.0


def validate_matches(matches, annotations, predictions, threshold, kind):
    targets=set()
    selected=set()
    for pair in matches:
        target, prediction=pair["target_index"], pair["prediction_index"]
        require(type(target) is int and 0<=target<len(annotations), "Invalid matched target index")
        require(type(prediction) is int and 0<=prediction<len(predictions), "Invalid matched prediction index")
        require(target not in targets and prediction not in selected, "Repeated instance match")
        targets.add(target)
        selected.add(prediction)
        annotation=annotations[target]
        require(pair["annotation_id"]==annotation["id"], "Matched annotation ID differs")
        iou=pair["iou"]
        require(type(iou) in (int, float) and math.isfinite(iou) and threshold<=iou<=1, "Invalid matched IoU")
        missing=annotation.get("mask_status")=="missing_source_polygon"
        if kind=="masks":
            require(not missing, "Known-mask match points to an unknown mask")
        else:
            require(math.isclose(iou, box_iou(annotation, predictions[prediction]), abs_tol=1e-9),
                    "Journal box-match IoU differs from manifest and prediction geometry")
            if kind=="ignored_masks":
                require(missing, "Ignored prediction does not match an unknown-mask target")
    return selected


def verify_row(row, record, index, identity_hash, thresholds):
    for key in ("image_id", "book", "image_path", "source_sha256"):
        require(row[key]==record[key], f"Journal coverage or source differs at image {index}: {key}")
    require(row["index"]==index and row["identity_sha256"]==identity_hash, "Journal index/identity differs")
    annotations=record["annotations"]
    require(len({ann["id"] for ann in annotations})==len(annotations), "Duplicate GT annotation ID")
    for ann in annotations:
        require(ann["category_id"]==5 and not ann.get("iscrowd", 0), "Unsupported ground-truth category/crowd")
        require(ann.get("image_id", record["image_id"])==record["image_id"], "Ground truth refers to another page")
        require((ann.get("segmentation") is None)==(ann.get("mask_status")=="missing_source_polygon"),
                "Invalid missing-mask annotation status")
    total=len(annotations)
    unknown=sum(ann.get("mask_status")=="missing_source_polygon" for ann in annotations)
    known=total-unknown
    predictions=row["predictions"]
    for prediction in predictions:
        score=prediction["confidence"]
        require(type(score) in (int, float) and math.isfinite(score) and float32(0.35)<=score<=1,
                "Journal prediction violates fixed confidence threshold")
        require(len(prediction["bbox"])==4 and all(math.isfinite(v) for v in prediction["bbox"]),
                "Invalid prediction box")
        require(re.fullmatch(r"[0-9a-f]{64}", prediction["mask_crop_sha256"]) is not None,
                "Invalid predicted mask hash")
    expected={"targets":total, "known_mask_targets":known, "missing_mask_targets":unknown,
              "training_polygon_rejected":record.get("training_polygon_rejected", False), "counts":{}}
    for threshold in thresholds:
        key=f"{threshold:.2f}"
        matched=row["matches"][key]
        box_predictions=validate_matches(matched["boxes"], annotations, predictions, threshold, "boxes")
        mask_predictions=validate_matches(matched["masks"], annotations, predictions, threshold, "masks")
        ignored=validate_matches(matched["ignored_masks"], annotations, predictions, threshold, "ignored_masks")
        require(not ignored&mask_predictions, "Ignored mask prediction also matches a known mask")
        box_pairs={(pair["target_index"], pair["prediction_index"]) for pair in matched["boxes"]}
        expected_ignored={(pair["target_index"], pair["prediction_index"]) for pair in matched["boxes"]
                          if annotations[pair["target_index"]].get("mask_status")=="missing_source_polygon"
                          and pair["prediction_index"] not in mask_predictions}
        actual_ignored={(pair["target_index"], pair["prediction_index"]) for pair in matched["ignored_masks"]}
        require(actual_ignored==expected_ignored and actual_ignored<=box_pairs, "Unknown-mask ignore policy differs")
        n=len(predictions)
        bt, mt=len(box_predictions), len(mask_predictions)
        conservative=metrics(mt, n-mt, total-mt)
        expected["counts"][key]={"boxes":metrics(bt, n-bt, total-bt),
                                  "masks":metrics(mt, n-mt-len(ignored), known-mt),
                                  "masks_conservative":conservative,
                                  "known_mask_targets":known, "missing_mask_targets":unknown,
                                  "ignored_mask_predictions":len(ignored),
                                  "mask_f1_lower_bound":conservative["f1"]}
    expected["ignored_mask_predictions"]=expected["counts"]["0.50"]["ignored_mask_predictions"]
    expected["mask_f1_lower_bound"]=expected["counts"]["0.50"]["mask_f1_lower_bound"]
    assert_fields(row, expected, f"journal[{index}]")
    return {**expected, "book":record["book"]}


def aggregate(rows, thresholds):
    result={"images":len(rows), "targets":sum(row["targets"] for row in rows),
            "known_mask_targets":sum(row["known_mask_targets"] for row in rows),
            "missing_mask_targets":sum(row["missing_mask_targets"] for row in rows),
            "negative_images":sum(row["targets"]==0 for row in rows),
            "polygon_rejected_images":sum(row["training_polygon_rejected"] for row in rows), "by_iou":{}}
    for threshold in thresholds:
        key=f"{threshold:.2f}"
        current={kind:metrics(*(sum(row["counts"][key][kind][name] for row in rows)
                               for name in ("tp", "fp", "fn"))) for kind in KINDS}
        for name in ("known_mask_targets", "missing_mask_targets", "ignored_mask_predictions"):
            current[name]=sum(row["counts"][key][name] for row in rows)
        current["mask_f1_lower_bound"]=current["masks_conservative"]["f1"]
        result["by_iou"][key]=current
    result.update(result["by_iou"]["0.50"])
    result.update({"valid":result["targets"]>0, "mask_valid":result["known_mask_targets"]>0})
    return result


def verify_report(path, protocol, records):
    path=Path(path).resolve(strict=True)
    report=read_json(path)
    require(report["complete"] is True and report["status"]=="complete", "Incomplete evaluation report")
    identity_path=path.with_suffix(".identity.json")
    require(digest(identity_path)==report["identity_sha256"], "Identity sidecar hash mismatch")
    identity=read_json(identity_path)
    require(identity==report["identity"], "Report identity differs from sidecar")
    require(identity["limit"] is None, "Limited evaluation cannot enter comparison")
    manifest=identity["manifest"]
    expected_order=hashlib.sha256("".join(str(row["image_id"])+"\n" for row in records).encode()).hexdigest()
    assert_fields(manifest, {"sha256":protocol["manifest_sha256"], "images":len(records),
                             "ordered_image_ids_sha256":expected_order}, "identity.manifest")
    require(digest(path_from(manifest["path"]))==protocol["manifest_sha256"], "Manifest identity hash differs")
    assert_fields(identity["settings"], {"conf":protocol["confidence"], "iou":protocol["nms_iou"],
                                         "imgsz":protocol["imgsz"], "half":False, "rect":False,
                                         "retina_masks":True, "agnostic_nms":True, "max_det":300}, "settings")
    thresholds=identity["match_iou"]
    require(thresholds and len(set(thresholds))==len(thresholds) and 0.5 in thresholds and
            all(type(v) in (int, float) and math.isfinite(v) and 0<v<=1 for v in thresholds), "Invalid matching thresholds")
    require(identity["source_sha256"], "Missing evaluator source identity")
    for source, expected in identity["source_sha256"].items():
        require(digest(path_from(source))==expected, "Evaluator source hash mismatch")
    require(digest(path_from(identity["model"]["path"]))==identity["model"]["sha256"], "Model hash mismatch")
    journal=path_from(report["predictions_jsonl"], path.parent)
    require(digest(journal)==report["predictions_sha256"], "Prediction journal hash mismatch")
    rows=read_rows(journal)
    require(len(rows)==len(records), "Journal does not cover every manifest image")
    summaries=[verify_row(row, record, i, report["identity_sha256"], thresholds)
               for i, (row, record) in enumerate(zip(rows, records))]
    recomputed=aggregate(summaries, thresholds)
    assert_fields(report, recomputed, "report")
    groups=defaultdict(list)
    for row in summaries:
        groups[row["book"]].append(row)
    expected_books=[{"book":book, **aggregate(values, thresholds)} for book, values in sorted(groups.items())]
    require(len(report["books"])==len(expected_books), "Issue-level report coverage differs")
    for observed, expected in zip(report["books"], expected_books):
        assert_fields(observed, expected, "report.books")
    gate=recomputed["mask_valid"] and min(recomputed["boxes"]["f1"], recomputed["mask_f1_lower_bound"])>=0.9
    require(report["target_f1"]==0.9 and report["development_target_reached"] is gate, "Reported gate differs from verified counts")
    return {"path":str(path), "sha256":digest(path), "identity":identity,
            "identity_sha256":report["identity_sha256"], "predictions_sha256":report["predictions_sha256"],
            "recomputed":recomputed}


def verify_lineage(checked, small_run, mayocream_run):
    require((small_run is None)==(mayocream_run is None), "Supply both candidate run metadata files together")
    if small_run is None:
        return {"verified":False, "status":"separate_run_lineage_verification_required",
                "scope":"Distinct checkpoint hashes and loader roles alone do not prove retraining or architecture identity."}
    evidence={}
    for role, value in (("small", small_run), ("mayocream", mayocream_run)):
        path=Path(value).resolve(strict=True)
        run=read_json(path)
        require(run["track"]==role and run.get("arguments", {}).get("track")==role,
                f"Run track does not match {role} role")
        require(run["status"] in ("validation_target_reached", "needs_review"),
                f"Run metadata for {role} does not record completed training")
        selected=run["stages"]["segments"]
        model_hash=checked[role]["identity"]["model"]["sha256"]
        require(selected["sha256"]==model_hash, f"Selected run checkpoint hash differs for {role}")
        require(digest(path_from(selected["checkpoint"]))==model_hash,
                f"Run checkpoint artifact hash differs for {role}")
        evidence[role]={"path":str(path), "sha256":digest(path), "track":role,
                        "run_status":run["status"], "selected_checkpoint_sha256":model_hash}
    return {"verified":True, "status":"run_metadata_and_selected_checkpoint_hashes_verified",
            "runs":evidence,
            "scope":"Binds completed run records and declared tracks to evaluated artifacts; upstream lineage and external selection discipline require separate provenance review."}


def compare(baseline, small, mayocream, protocol_path=DEFAULT_PROTOCOL, output=None,
            small_run=None, mayocream_run=None):
    protocol_path=Path(protocol_path).resolve(strict=True)
    protocol_hash=digest(protocol_path)
    protocol=read_json(protocol_path)
    assert_fields(protocol, {"confidence":0.35, "iou":0.5, "nms_iou":0.5, "agnostic_nms":True,
                             "precision":"FP32", "mask_resolution":"original_image", "no_tuning_on_external_data":True,
                             "target":{"box_f1":0.9, "mask_f1_conservative":0.9},
                             "comparability":{"max_absolute_box_f1_difference":0.01,
                                              "max_absolute_mask_f1_difference":0.01}}, "protocol")
    manifest=path_from(protocol["manifest"])
    require(digest(manifest)==protocol["manifest_sha256"], "Frozen manifest hash mismatch")
    records=read_rows(manifest)
    require(len(records)==protocol["images"] and len({str(row["image_id"]) for row in records})==len(records),
            "Frozen manifest image count/identity mismatch")
    checked={name:verify_report(path, protocol, records)
             for name, path in (("baseline", baseline), ("small", small), ("mayocream", mayocream))}
    require(len({item["identity"]["model"]["sha256"] for item in checked.values()})==3,
            "Baseline and candidate roles must reference three distinct model hashes")
    require(checked["baseline"]["identity"]["model"]["mayocream"] is True,
            "Baseline must use the Mayocream SafeTensors loader")
    for role in ("small", "mayocream"):
        require(checked[role]["identity"]["model"]["mayocream"] is False,
                f"Candidate {role} must use the trained checkpoint loader")
    lineage=verify_lineage(checked, small_run, mayocream_run)
    reference=checked["baseline"]["identity"]
    for name, item in checked.items():
        for field in ("settings", "match_iou", "source_sha256", "versions", "method", "missing_mask_policy"):
            require(item["identity"][field]==reference[field], f"Incompatible {field} in {name}")
    scores={name:{"box_f1":item["recomputed"]["boxes"]["f1"],
                  "known_mask_f1":item["recomputed"]["masks"]["f1"],
                  "mask_f1_conservative":item["recomputed"]["mask_f1_lower_bound"]}
            for name, item in checked.items()}
    deltas={key:scores["mayocream"][key]-scores["small"][key] for key in scores["small"]}
    comparable=abs(deltas["box_f1"])<=0.01+1e-12 and abs(deltas["mask_f1_conservative"])<=0.01+1e-12
    tolerance=protocol.get("mayocream_noninferiority_tolerance")
    if tolerance is not None:
        require(type(tolerance) in (int, float) and tolerance==0.01,
                "Declared Mayocream noninferiority tolerance must be0.01")
    at_least_comparable=(all(deltas[key]>=-tolerance-1e-12
                            for key in ("box_f1", "mask_f1_conservative"))
                         if tolerance is not None else comparable)
    gates={name:item["recomputed"]["mask_valid"] and min(values["box_f1"], values["mask_f1_conservative"])>=0.9
           for name, values in scores.items() for item in [checked[name]]}
    relevant=[deltas["box_f1"], deltas["mask_f1_conservative"]]
    preference=("tie" if all(value==0 for value in relevant) else
                "mayocream" if all(value>=0 for value in relevant) else
                "small" if all(value<=0 for value in relevant) else "tradeoff")
    result={"status":"verified", "protocol":{"path":str(protocol_path), "sha256":protocol_hash},
            "manifest":{"path":str(manifest), "sha256":protocol["manifest_sha256"], "images":len(records)},
            "scores":scores, "gains_vs_baseline":{
                name:{key:value-scores["baseline"][key] for key, value in scores[name].items()}
                for name in ("small", "mayocream")},
            "mayocream_minus_small":deltas, "practically_comparable":comparable,
            "comparability_definition":protocol["comparability"],
            "ai4va_subset_target_reached":gates,
            "both_candidate_models_pass_and_comparable":gates["small"] and gates["mayocream"] and comparable,
            "both_retrained_models_pass_and_comparable":gates["small"] and gates["mayocream"] and comparable and lineage["verified"],
            "mayocream_meets_declared_comparison":at_least_comparable,
            "comparison_allows_mayocream_to_outperform":tolerance is not None,
            "both_retrained_models_meet_declared_quality_requirement":gates["small"] and gates["mayocream"] and at_least_comparable and lineage["verified"],
            "training_lineage":lineage,
            "observed_joint_quality_preference":preference,
            "architecture_superiority_established":False,
            "requirements":{"box_f1":0.9, "mask_f1_conservative":0.9, "complete_identical_coverage":True,
                            "prediction_journals_and_identities_verified":True, "aggregate_counts_recomputed":True},
            "evidence":{name:{key:value for key, value in item.items() if key!="identity"} for name, item in checked.items()},
            "limitations":["Scores and gates apply only to this AI4VA subset and frozen operating point; no universal generalization claim.",
                           "Publication issue groups are not independent story series. The paper describes two central series; per-page membership is unknown and the released pages visibly contain additional strips.",
                           "Absolute differences within0.01 are descriptive practical comparability, not statistical equivalence.",
                           "Both candidates belong to YOLO11; capacity and pretraining differ, so these results cannot isolate architecture superiority.",
                           "Unknown-mask conservative counts include missing targets as false negatives and restore ignored predictions as false positives.",
                           "The legacy mask_f1_lower_bound field is a conservative known-mask score under the documented missing-mask policy, not a rigorous universal bound on fully annotated greedy-matcher F1.",
                           "Journal counts, match indices and matched box IoUs are checked. Mask IoUs cannot be independently recalculated because journals retain mask hashes rather than pixels.",
                           "No inference about upstream training exposure or complete human label correctness follows from artifact hash verification."]}
    require(digest(protocol_path)==protocol_hash, "Protocol changed during comparison")
    if output is not None:
        path=Path(output)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "small", "mayocream", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--small-run", help="Completed small-track run.json; supply together with --mayocream-run")
    parser.add_argument("--mayocream-run", help="Completed Mayocream-track run.json")
    args=parser.parse_args()
    result=compare(args.baseline, args.small, args.mayocream, args.protocol, args.output,
                   args.small_run, args.mayocream_run)
    print(json.dumps({key:result[key] for key in ("scores", "practically_comparable", "ai4va_subset_target_reached")}, indent=2))
