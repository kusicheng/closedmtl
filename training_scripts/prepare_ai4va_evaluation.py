"""Prepare evaluation-only AI4VA manifests using canonical COCO polygon rasterization.

Retain all available annotated pages, including supplied negative pages and the
one bubble whose source polygon is missing. Never fabricate a mask or a negative
label for a completely unannotated source page.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATASET=ROOT/"training_data/evaluation/ai4va"
SPLITS={"val":"validation", "test":"test"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_id(value, kind):
    if type(value) is not int or value<0:
        raise ValueError(f"Invalid COCO {kind} ID: {value}")
    return value


def qualified_id(split, value, kind="image"):
    return f"ai4va:{split}:{'annotation:' if kind=='annotation' else ''}{value}"


def finite_values(values, length=None):
    if not isinstance(values, list) or (length is not None and len(values)!=length):
        raise ValueError("Invalid coordinate list")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Coordinates must be finite numbers")


def convert_annotation(annotation, image, split):
    """Preserve source fields and derive known-mask bounds from canonical RLE."""
    source_id(annotation["id"], "annotation")
    if annotation["category_id"]!=26 or annotation.get("iscrowd", 0)!=0:
        raise ValueError("Expected an individual category-26 Comic Bubble")
    width, height=image["width"], image["height"]
    box=annotation["bbox"]
    finite_values(box, 4)
    x, y, w, h=box
    if min(x, y)<0 or min(w, h)<=0 or x+w>width+1e-6 or y+h>height+1e-6:
        raise ValueError(f"Invalid source box for annotation {annotation['id']}")
    area=annotation.get("area")
    if type(area) not in (int, float) or not math.isfinite(area) or area<=0:
        raise ValueError("Invalid source area")
    result={
        "id":qualified_id(split, annotation["id"], "annotation"),
        "image_id":qualified_id(split, annotation["image_id"]),
        "category_id":5, "iscrowd":0,
        "source_annotation_id":annotation["id"],
        "source_image_id":annotation["image_id"], "source_category_id":26,
        "source_bbox":deepcopy(box), "source_area":area,
        "source_attributes":deepcopy(annotation.get("attributes", {})),
    }
    polygons=annotation.get("segmentation")
    if polygons is None or polygons==[]:
        result.update({"bbox":deepcopy(box), "area":area, "segmentation":None,
                       "mask_status":"missing_source_polygon"})
        return result
    if not isinstance(polygons, list):
        raise ValueError("Expected COCO polygon lists")
    for polygon in polygons:
        finite_values(polygon)
        if len(polygon)<6 or len(polygon)%2:
            raise ValueError("Each polygon needs at least three coordinate pairs")
        points=np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(np.unique(points, axis=0))<3:
            raise ValueError("A polygon needs three distinct vertices")
        if np.any(points<0) or np.any(points[:, 0]>width) or np.any(points[:, 1]>height):
            raise ValueError(f"Polygon outside image for annotation {annotation['id']}")
    rle=coco_mask.merge(coco_mask.frPyObjects(polygons, height, width), intersect=False)
    area=int(coco_mask.area(rle))
    bbox=coco_mask.toBbox(rle).tolist()
    if area<=0 or any(value!=int(value) for value in bbox):
        raise ValueError("Polygon union is empty or has noninteger mask bounds")
    result.update({"bbox":[int(value) for value in bbox], "area":area,
                   "segmentation":{"size":[height, width], "counts":rle["counts"].decode("ascii")},
                   "mask_status":"known", "source_polygon_count":len(polygons)})
    return result


def source_box_annotation(annotation):
    result=deepcopy(annotation)
    result["bbox"]=deepcopy(result["source_bbox"])
    result["area"]=result["source_area"]
    result.pop("segmentation", None)
    result["evaluation_target"]="source_bbox_only"
    return result


def checked_image(dataset, mapping, image):
    path=Path(mapping["organized_path"]).resolve()
    if not path.is_relative_to(dataset):
        raise ValueError(f"Organized image escapes dataset root: {path}")
    if not path.is_file() or digest(path)!=mapping["sha256"]:
        raise ValueError(f"Organized image SHA256 mismatch or missing file: {path}")
    if [mapping["width"], mapping["height"]]!=[image["width"], image["height"]]:
        raise ValueError("Mapping and COCO image dimensions differ")
    with Image.open(path) as opened:
        opened.load()
        if opened.size!=(image["width"], image["height"]):
            raise ValueError("Actual image dimensions differ from COCO metadata")
    return path


def load_split(dataset, split, mappings):
    path=dataset/"annotations"/f"source_{split}.json"
    data=json.loads(path.read_text(encoding="utf-8"))
    categories={item["id"]:item for item in data["categories"]}
    if len(categories)!=len(data["categories"]):
        raise ValueError("Duplicate COCO category IDs")
    category=categories.get(26)
    if category is None or re.sub(r"[^a-z]", "", category["name"].lower())!="comicbubble":
        raise ValueError("Category26 is not Comic Bubble")
    images={}
    for image in data["images"]:
        ident=source_id(image["id"], "image")
        if ident in images:
            raise ValueError("Duplicate source image ID within one split")
        if any(type(image[name]) is not int or image[name]<=0 for name in ("width", "height")):
            raise ValueError("Invalid image dimensions")
        images[ident]=image
    if set(images)!={ident for source_split, ident in mappings if source_split==split}:
        raise ValueError("Mapping image IDs differ from source COCO image IDs")
    grouped=defaultdict(list)
    ids=set()
    for annotation in data["annotations"]:
        ident=source_id(annotation["id"], "annotation")
        source_id(annotation["image_id"], "image")
        if ident in ids or annotation["image_id"] not in images:
            raise ValueError("Duplicate annotation ID or unknown image reference")
        if annotation["category_id"] not in categories:
            raise ValueError("Annotation references unknown category")
        ids.add(ident)
        grouped[annotation["image_id"]].append(annotation)
    counts=Counter(source_images=len(images), source_annotations=len(ids))
    counts["source_bubble_boxes"]=sum(ann["category_id"]==26 for ann in data["annotations"])
    joint=[]
    boxes=[]
    exclusions=[]
    missing=[]
    for ident, image in images.items():
        mapping=mappings[(split, ident)]
        if mapping["source_file_name"]!=image["file_name"]:
            raise ValueError("Mapping source filename differs from COCO filename")
        annotations=grouped[ident]
        if not annotations:
            exclusions.append({"source_split":split, "source_image_id":ident,
                               "source_file_name":image["file_name"],
                               "reason":"completely_unannotated_page_is_not_a_supplied_negative",
                               "image_availability":mapping.get("availability"),
                               "excluded_bubble_boxes":0})
            counts["excluded_unannotated_pages"]+=1
            continue
        source=checked_image(dataset, mapping, image)
        bubbles=[convert_annotation(ann, image, split) for ann in annotations if ann["category_id"]==26]
        missing_here=[ann for ann in bubbles if ann["mask_status"]!="known"]
        for ann in missing_here:
            missing.append({"source_split":split, "source_image_id":ident,
                            "source_annotation_id":ann["source_annotation_id"],
                            "image_id":qualified_id(split, ident), "annotation_id":ann["id"],
                            "mask_status":ann["mask_status"], "page_retained":True})
        issue=mapping["issue_id"]
        if not issue:
            raise ValueError("Missing publication issue identity")
        record={"image_id":qualified_id(split, ident), "source_image_id":ident,
                "source_split":split, "split":SPLITS[split],
                "book":issue, "group_type":"publication_issue", "issue_id":issue,
                "publication_date":mapping["publication_date"], "page_number":mapping["page_number"],
                "story_series":None, "image_path":str(source), "source_sha256":mapping["sha256"],
                "width":image["width"], "height":image["height"],
                "source_file_name":image["file_name"],
                "source_ref":{"annotations":str(path), "image_id":ident},
                "annotations":bubbles, "training_polygon_rejected":False,
                "annotation_status":"supplied_bubble_positive" if bubbles else "supplied_negative_other_classes_annotated",
                "evaluation_only":True, "manifest_kind":"joint_mask_and_box"}
        joint.append(record)
        box_record=deepcopy(record)
        box_record["manifest_kind"]="source_box_only"
        box_record["annotations"]=[source_box_annotation(ann) for ann in bubbles]
        boxes.append(box_record)
        counts["available_annotated_pages"]+=1
        counts["bubble_boxes"]+=len(bubbles)
        counts["known_bubble_masks"]+=len(bubbles)-len(missing_here)
        counts["missing_bubble_masks"]+=len(missing_here)
        counts["supplied_negative_pages"]+=int(not bubbles)
        counts["pages_with_missing_bubble_masks"]+=int(bool(missing_here))
        counts["complete_mask_pages"]+=int(not missing_here)
    return joint, boxes, dict(counts), exclusions, missing, {str(path):digest(path)}


def prepare(dataset_root=DEFAULT_DATASET):
    dataset=Path(dataset_root).resolve()
    output=dataset/"evaluation"
    if output.exists():
        raise FileExistsError(f"Use a fresh evaluation directory: {output}")
    mapping_path=dataset/"mapping_manifest.json"
    mapping=json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("status")!="complete":
        raise ValueError("Dataset organization is not complete")
    mappings={}
    for row in mapping["images"]:
        key=(row["source_split"], source_id(row["image_id"], "image"))
        if key in mappings or key[0] not in SPLITS:
            raise ValueError("Duplicate or unsupported mapping split/image ID")
        mappings[key]=row
    counts={}
    exclusions=[]
    missing=[]
    records={}
    sources={str(mapping_path):digest(mapping_path), str(Path(__file__).resolve()):digest(__file__)}
    hashes={}
    for split, name in SPLITS.items():
        joint, boxes, totals, omitted, unknown, source_hashes=load_split(dataset, split, mappings)
        counts[name]=totals
        exclusions.extend(omitted)
        missing.extend(unknown)
        sources.update(source_hashes)
        for row in joint:
            if row["source_sha256"] in hashes:
                previous=hashes[row["source_sha256"]]
                if previous[0]!=split:
                    raise ValueError(f"Cross-split image hash overlap: {previous[1]} and {row['image_id']}")
            hashes[row["source_sha256"]]=(split, row["image_id"])
        records[f"{name}_joint_masks.jsonl"]=joint
        records[f"{name}_source_boxes.jsonl"]=boxes
    report={"status":"complete", "usage":"evaluation_only_no_training_or_calibration",
            "counts":counts, "exclusions":exclusions, "missing_masks":missing,
            "source_hashes":sources, "category_mapping":{"26 Comic Bubble":5},
            "pycocotools_version":importlib.metadata.version("pycocotools"),
            "rasterization":"pycocotools.mask.frPyObjects then union merge; tight integer bounds and area from RLE",
            "source_box_policy":"Separate source-box manifests retain original floating COCO bounds and area for every bubble.",
            "missing_mask_policy":"Retain the page, every box, and all known masks. A missing polygon has segmentation=null and mask_status=missing_source_polygon; no rectangle pseudomask.",
            "negative_policy":"Pages without bubbles are retained only when other source categories annotate the page. Entirely unannotated pages are excluded.",
            "group_type":"publication_issue", "series_from_official_paper":mapping.get("series_from_official_paper"),
            "series_limitation":"Issue groups are not independent books or story series. Per-page story-series membership is unavailable; the official source describes only two series.",
            "cross_split_exact_image_hash_overlaps":0,
            "acceptance_eligible":False,
            "acceptance_reason":"External evaluation preparation only; source coverage, missing-mask bounds, model exposure and label calibration must be considered with measured results.",
            "manifest_sha256":{}}
    output.mkdir(parents=False, exist_ok=False)
    for name, rows in records.items():
        path=output/name
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        report["manifest_sha256"][name]=digest(path)
    with (output/"provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    args=parser.parse_args()
    result=prepare(args.dataset_root)
    print(json.dumps({"counts":result["counts"], "exclusions":result["exclusions"],
                      "missing_masks":result["missing_masks"]}, indent=2))
