"""Freeze annotation-only issue splits and prepare real-mask AI4VA adaptation data."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training_scripts.prepare_ai4va_evaluation import checked_image, convert_annotation, digest
from training_scripts.prepare_bubble_segments import box_line, cropped_mask, link_image, polygon_from_mask


DATASET=ROOT/"training_data/ai4va_additional"
AUDIT=ROOT/"outputs/bubble_training_20260922/ai4va_audit/prospective_train_audit.json"
EXCLUDED={154:"missing_source_polygon", 272:"crowd_bubble"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def assign_issues(stats, seed=20260923, minimum_masks=200, attempts=10000):
    """Stratify issue groups using only known annotation counts, never predictions."""
    if len(stats)<5 or minimum_masks<1:
        raise ValueError("Need at least five issues and a positive heldout mask minimum")
    holdout=round(len(stats)*0.2)
    positive=sorted(issue for issue, row in stats.items() if row["masks"]>0)
    negative=sorted(set(stats)-set(positive))
    negative_count=min(round(len(negative)*0.2), holdout-1, len(negative)//2)
    positive_count=holdout-negative_count
    if len(positive)<2*positive_count+1 or sum(row["masks"] for row in stats.values())<2*minimum_masks:
        raise ValueError("Insufficient annotated capacity for requested heldout issue/mask counts")
    rng=random.Random(seed)
    for attempt in range(1, attempts+1):
        positives=positive.copy()
        negatives=negative.copy()
        rng.shuffle(positives)
        rng.shuffle(negatives)
        calibration=positives[:positive_count]+negatives[:negative_count]
        test=positives[positive_count:2*positive_count]+negatives[negative_count:2*negative_count]
        if min(sum(stats[issue]["masks"] for issue in group) for group in (calibration, test))<minimum_masks:
            continue
        selected=set(calibration+test)
        groups={"train":sorted(set(stats)-selected), "calibration":sorted(calibration), "test":sorted(test)}
        return {"seed":seed, "method":"Sorted issue IDs; seeded stratified shuffle; first assignment meeting count-only minima",
                "selection_inputs":"Eligible source annotation counts and issue IDs only; no model outputs or scores",
                "attempts_used":attempt, "minimum_heldout_known_masks":minimum_masks,
                "negative_only_issues_per_heldout":negative_count,
                "issue_splits":groups,
                "issue_to_split":{issue:split for split, issues in groups.items() for issue in issues},
                "counts":{split:{"issues":len(issues),
                                 **{key:sum(stats[issue][key] for issue in issues)
                                    for key in ("pages", "masks", "negative_pages")}}
                          for split, issues in groups.items()},
                "minimum_heldout_mask_requirement_met":True}
    raise ValueError("No qualifying annotation-only split found within search budget; infeasibility is not established")


def inspect_sources(dataset, audit_path):
    dataset=Path(dataset).resolve(strict=True)
    audit=load_json(audit_path)
    mapping_path=dataset/"mapping_manifest.json"
    mapping=load_json(mapping_path)
    if mapping.get("status")!="complete":
        raise ValueError("Additional image organization is incomplete")
    source_path=dataset/"annotations/source_train.json"
    if digest(source_path)!=audit["source_sha256"]:
        raise ValueError("Official source annotations differ from prospective audit")
    source=load_json(source_path)
    categories={category["id"]:category for category in source["categories"]}
    if categories.get(26, {}).get("name")!="Comic Bubble":
        raise ValueError("Expected category26 Comic Bubble")
    images={row["id"]:row for row in source["images"]}
    if len(images)!=len(source["images"]):
        raise ValueError("Duplicate source image IDs")
    annotations=defaultdict(list)
    annotation_ids=set()
    for annotation in source["annotations"]:
        if annotation["id"] in annotation_ids or annotation["image_id"] not in images:
            raise ValueError("Duplicate annotation ID or invalid image reference")
        if annotation["category_id"] not in categories:
            raise ValueError("Annotation references an unknown source category")
        annotation_ids.add(annotation["id"])
        annotations[annotation["image_id"]].append(annotation)
    frozen_path=Path(audit["frozen_mapping_path"]).resolve(strict=True)
    if digest(frozen_path)!=audit["frozen_mapping_sha256"]:
        raise ValueError("Frozen external issue mapping changed after prospective audit")
    frozen=load_json(frozen_path)
    frozen_issues={row["issue_id"] for row in frozen["images"] if row.get("sha256")}
    frozen_hashes={row["sha256"] for row in frozen["images"] if row.get("sha256")}
    prospective={row["image_id"]:row for row in audit["pages"] if not row["overlaps_frozen_evaluation_issue"]}
    organized={row["image_id"]:row for row in mapping["images"]}
    if len(organized)!=len(mapping["images"]) or set(organized)!=set(prospective):
        raise ValueError("Organized images do not exactly cover prospective nonoverlap pages")
    rows=[]
    exclusions=[]
    stats=defaultdict(lambda:{"pages":0, "masks":0, "negative_pages":0})
    for ident in sorted(organized):
        item=organized[ident]
        image=images[ident]
        if item["source_split"]!="train" or item["source_file_name"]!=image["file_name"]:
            raise ValueError("Organized source split/filename differs")
        if item["issue_id"]!=prospective[ident]["issue_id"] or item["issue_id"] in frozen_issues:
            raise ValueError("Issue identity differs or overlaps frozen external evaluation")
        path=checked_image(dataset, item, image)
        if item["sha256"] in frozen_hashes:
            raise ValueError("Image bytes overlap frozen external evaluation")
        all_annotations=annotations[ident]
        if not all_annotations:
            raise ValueError("Unannotated page cannot become an adaptation negative")
        bubbles=[annotation for annotation in all_annotations if annotation["category_id"]==26]
        missing=[annotation["id"] for annotation in bubbles if not annotation.get("segmentation")]
        crowd=[annotation["id"] for annotation in bubbles if annotation.get("iscrowd", 0)]
        if ident in EXCLUDED:
            if not ((ident==154 and missing) or (ident==272 and crowd)):
                raise ValueError("Prospective exclusion no longer matches source annotation issue")
            exclusions.append({"source_image_id":ident, "issue_id":item["issue_id"],
                               "source_file_name":image["file_name"], "image_path":str(path),
                               "source_sha256":item["sha256"], "status":"needs_annotation_review",
                               "reason":EXCLUDED[ident], "excluded_from":["train", "calibration", "test"],
                               "bubble_boxes":len(bubbles), "known_nonempty_masks":len(bubbles)-len(missing),
                               "missing_mask_annotation_ids":missing, "crowd_annotation_ids":crowd,
                               "excluded_annotation_ids":[annotation["id"] for annotation in bubbles]})
            continue
        if missing or crowd:
            raise ValueError(f"Unexpected unresolved bubble annotation issue on page {ident}")
        canonical=[convert_annotation(annotation, image, "train") for annotation in bubbles]
        row={"image_id":f"ai4va:train:{ident}", "source_image_id":ident, "source_split":"train",
             "book":item["issue_id"], "issue_id":item["issue_id"], "group_type":"publication_issue",
             "story_series":None, "publication_date":item["publication_date"], "page_number":item["page_number"],
             "image_path":str(path), "source_image":str(path), "source_sha256":item["sha256"],
             "source_file_name":image["file_name"], "width":image["width"], "height":image["height"],
             "annotations":canonical, "source_annotation_count_all_categories":len(all_annotations),
             "annotation_status":"supplied_bubble_positive" if bubbles else "supplied_negative_other_classes_annotated",
             "source_ref":{"annotations":str(source_path), "image_id":ident}}
        rows.append(row)
        current=stats[item["issue_id"]]
        current["pages"]+=1
        current["masks"]+=len(bubbles)
        current["negative_pages"]+=int(not bubbles)
    hashes={str(path.resolve()):digest(path) for path in (source_path, mapping_path, Path(audit_path), frozen_path,
             Path(__file__), ROOT/"training_scripts/prepare_ai4va_evaluation.py",
             ROOT/"training_scripts/prepare_bubble_segments.py")}
    return rows, exclusions, dict(stats), hashes


def write_yaml(path, output, kind):
    text=f"path: {output.as_posix()}\ntrain: {kind}_train.txt\nval: {kind}_calibration.txt\nnames:\n  0: balloon\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def prepare(dataset_root=DATASET, audit_path=AUDIT, output=None, seed=20260923,
            minimum_masks=200, minimum_iou=0.98, polygon_converter=polygon_from_mask):
    dataset=Path(dataset_root).resolve(strict=True)
    output=Path(output).resolve() if output is not None else dataset/"prepared"
    if output.exists():
        raise FileExistsError(f"Use a fresh prepared directory: {output}")
    if not 0.98<=minimum_iou<=1.0:
        raise ValueError("Minimum polygon raster IoU must be at least0.98")
    rows, excluded, stats, source_hashes=inspect_sources(dataset, audit_path)
    plan=assign_issues(stats, seed, minimum_masks)
    seen_hashes={}
    for row in rows:
        row["split"]=plan["issue_to_split"][row["issue_id"]]
        previous=seen_hashes.get(row["source_sha256"])
        if previous is not None and previous!=row["split"]:
            raise ValueError("Duplicate image bytes would cross prospective split boundaries")
        seen_hashes[row["source_sha256"]]=row["split"]
    output.mkdir(parents=True, exist_ok=False)
    plan.update({"frozen_at_utc":datetime.now(timezone.utc).isoformat(), "source_hashes":source_hashes,
                 "group_type":"publication_issue", "excluded_page_ids":[row["source_image_id"] for row in excluded],
                 "test_policy":"No model inference or threshold/checkpoint selection on this test set before final selection."})
    write_json(output/"split_plan.json", plan)
    write_json(output/"excluded_pages.json", excluded)
    exact={split:[] for split in ("train", "calibration", "test")}
    lists=defaultdict(list)
    counts=defaultdict(Counter)
    rejected=[]
    quality=[]
    links=Counter()
    for row in rows:
        split=row["split"]
        lines=[]
        failures=[]
        if split!="test":
            for annotation in row["annotations"]:
                try:
                    mask=cropped_mask(annotation, row)
                    line, geometry=polygon_converter(mask, annotation["bbox"], row, minimum_iou)
                except ValueError as error:
                    line, geometry=None, {"iou":0.0, "error":str(error)}
                if line is None:
                    failures.append(annotation["id"])
                else:
                    lines.append(line)
                quality.append({"image_id":row["image_id"], "annotation_id":annotation["id"],
                                "split":split, **geometry})
        row["training_polygon_rejected"]=bool(failures)
        row["polygon_conversion_status"]="not_required_for_untouched_test" if split=="test" else "rejected" if failures else "accepted"
        exact[split].append(row)
        counts[split]["exact_pages"]+=1
        counts[split]["exact_masks"]+=len(row["annotations"])
        counts[split]["exact_negative_pages"]+=int(not row["annotations"])
        if failures:
            rejected.append({"image_id":row["image_id"], "source_image_id":row["source_image_id"],
                             "issue_id":row["issue_id"], "split":split,
                             "reason":"polygon_raster_iou_below_requirement_or_conversion_error",
                             "failed_annotation_ids":failures, "minimum_iou":minimum_iou,
                             "whole_page_excluded_from_yolo":True, "exact_manifest_retained":True,
                             "bubble_masks":len(row["annotations"])})
            continue
        if split=="test":
            continue
        stem=f"ai4va_train_{row['source_image_id']:04d}_{Path(row['source_file_name']).stem}"
        for kind in ("segments", "boxes"):
            target=output/kind/"images"/split/(stem+Path(row["image_path"]).suffix.lower())
            links[link_image(Path(row["image_path"]), target)]+=1
            label=output/kind/"labels"/split/(stem+".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            labels=lines if kind=="segments" else [box_line(ann["bbox"], row["width"], row["height"]) for ann in row["annotations"]]
            with label.open("x", encoding="utf-8") as stream:
                stream.write("\n".join(labels)+("\n" if labels else ""))
            lists[f"{kind}_{split}"].append(target.as_posix())
        counts[split]["yolo_pages"]+=1
        counts[split]["yolo_masks"]+=len(row["annotations"])
        counts[split]["yolo_negative_pages"]+=int(not row["annotations"])
    for split, records in exact.items():
        with (output/f"full_{split}_masks.jsonl").open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    for kind in ("segments", "boxes"):
        for split in ("train", "calibration"):
            with (output/f"{kind}_{split}.txt").open("x", encoding="utf-8") as stream:
                stream.write("\n".join(lists[f"{kind}_{split}"])+"\n")
        write_yaml(output/f"{kind}.yaml", output, kind)
    with (output/"polygon_quality.jsonl").open("x", encoding="utf-8") as stream:
        for row in quality:
            stream.write(json.dumps(row)+"\n")
    write_json(output/"polygon_rejected_pages.json", rejected)
    if any(digest(path)!=value for path, value in source_hashes.items()):
        raise ValueError("Source data or preparation code changed during conversion")
    report={"status":"complete", "seed":seed, "counts":dict(counts),
            "source_hashes":source_hashes, "issue_splits":plan["issue_splits"],
            "split_plan_sha256":digest(output/"split_plan.json"), "source_issue_stats":stats,
            "prospectively_excluded_pages":excluded, "polygon_rejected_pages":rejected,
            "minimum_polygon_iou":minimum_iou, "image_link_methods":dict(links),
            "artifact_sha256":{str(path.relative_to(output)):digest(path)
                               for path in sorted(output.rglob("*")) if path.is_file()},
            "versions":{name:importlib.metadata.version(name) for name in ("pycocotools", "ultralytics", "numpy")},
            "test_untouched_by_model_inference":True, "acceptance_established":False,
            "yolo_calibration_has_positive_masks":counts["calibration"]["yolo_masks"]>0,
            "limitations":["Splits were frozen from source annotation counts and issue IDs before polygon filtering or model inference.",
                           "Issue independence does not establish story-series or artist independence. Upstream model exposure remains unknown.",
                           "Use complete exact-RLE calibration/test manifests for authoritative evaluation; YOLO calibration can omit polygon-rejected pages.",
                           "Training polygons are accepted only after serialized/resampled raster IoU passes0.98; boxes derive from canonical masks.",
                           "Test images have no YOLO list or YAML test entry and no training-polygon conversion; they remain reserved for final evaluation.",
                           "Frozen external evaluation issues and exact image hashes are excluded; its files and protocol were not modified."]}
    write_json(output/"provenance.json", report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DATASET))
    parser.add_argument("--audit", default=str(AUDIT))
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--minimum-heldout-masks", type=int, default=200)
    args=parser.parse_args()
    result=prepare(args.dataset_root, args.audit, args.output, args.seed, args.minimum_heldout_masks)
    print(json.dumps({"counts":result["counts"], "issue_splits":result["issue_splits"],
                      "excluded_pages":len(result["prospectively_excluded_pages"]),
                      "polygon_rejected_pages":len(result["polygon_rejected_pages"])}, indent=2))
