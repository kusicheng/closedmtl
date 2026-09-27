"""Compose real-mask adaptation/replay data while keeping all test pages reserved."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANGA_POLYGON_PRODUCER=ROOT/"training_scripts/prepare_bubble_segments.py"
DEFAULT_AI4VA=ROOT/"training_data/ai4va_additional/prepared"
DEFAULT_MANGA=ROOT/"training_data/bubble_joint_20260922"
DEFAULT_FROZEN=ROOT/"training_data/evaluation/ai4va/mapping_manifest.json"
DEFAULT_OUTPUT=ROOT/"training_data/bubble_joint_adaptation_20260927"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def paths_from_list(path):
    source=Path(path).resolve(strict=True)
    paths=[Path(line).resolve(strict=True) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths)!=len(set(paths)):
        raise ValueError(f"Duplicate image paths: {source}")
    return paths


def label_path(image):
    parts=list(Path(image).parts)
    if "images" not in parts:
        raise ValueError(f"Image has no YOLO images directory: {image}")
    index=len(parts)-1-parts[::-1].index("images")
    parts[index]="labels"
    return Path(*parts).with_suffix(".txt")


def label_count(path, segmentation):
    lines=[line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for values in lines:
        expected=(len(values)>=7 and len(values)%2==1) if segmentation else len(values)==5
        if not expected or values[0]!="0":
            raise ValueError(f"Invalid one-class {'polygon' if segmentation else 'box'} label: {path}")
        if any(not math.isfinite(float(value)) or not 0<=float(value)<=1 for value in values[1:]):
            raise ValueError(f"Invalid normalized coordinates: {path}")
    return len(lines)


def deterministic_selection(paths, count, seed):
    ordered=sorted(paths, key=lambda path:path.as_posix())
    if count<1 or len(ordered)<count:
        raise ValueError("Insufficient Manga replay pages for the requested positive count")
    return sorted(random.Random(seed).sample(ordered, count), key=lambda path:path.as_posix())


def verify_hash(path, expected, sources):
    path=Path(path).resolve(strict=True)
    actual=digest(path)
    if actual!=expected:
        raise ValueError(f"Source hash changed: {path}")
    sources[path.as_posix()]=actual


def disjoint_groups(groups, description):
    seen=set()
    for values in groups.values():
        if seen.intersection(values):
            raise ValueError(f"Overlapping {description} across source splits")
        seen.update(values)


def inspect_ai4va(folder, frozen_path, sources):
    provenance=load_json(folder/"provenance.json")
    artifacts={name.replace("\\", "/"):value for name, value in provenance["artifact_sha256"].items()}
    plan=load_json(folder/"split_plan.json")
    if provenance.get("status")!="complete":
        raise ValueError("AI4VA preparation is incomplete")
    verify_hash(folder/"split_plan.json", provenance["split_plan_sha256"], sources)
    if provenance["issue_splits"]!=plan["issue_splits"]:
        raise ValueError("AI4VA issue split provenance differs")
    disjoint_groups(plan["issue_splits"], "AI4VA publication issues")
    if plan["issue_to_split"]!={issue:split for split, issues in plan["issue_splits"].items() for issue in issues}:
        raise ValueError("AI4VA issue lookup differs from frozen split groups")
    frozen=load_json(frozen_path)
    # Missing, entirely unannotated source image94 was excluded from the old62-page evaluation.
    frozen_issues={row["issue_id"] for row in frozen["images"] if row.get("sha256")}
    frozen_hashes={row["sha256"] for row in frozen["images"] if row.get("sha256")}
    expected_frozen=provenance["source_hashes"].get(str(frozen_path))
    if expected_frozen is None:
        expected_frozen=provenance["source_hashes"].get(frozen_path.as_posix())
    if expected_frozen is None:
        raise ValueError("Frozen external evaluation mapping is absent from AI4VA provenance")
    verify_hash(frozen_path, expected_frozen, sources)
    if any(frozen_issues.intersection(issues) for issues in plan["issue_splits"].values()):
        raise ValueError("AI4VA adaptation issues overlap frozen external evaluation")
    rows={}
    reserved_hashes=set(frozen_hashes)
    for split in ("train", "calibration", "test"):
        manifest=folder/f"full_{split}_masks.jsonl"
        verify_hash(manifest, artifacts[manifest.name], sources)
        metadata={}
        for row in json_rows(manifest):
            if row["split"]!=split or plan["issue_to_split"].get(row["issue_id"])!=split:
                raise ValueError("AI4VA exact manifest page disagrees with frozen issue split")
            if row["source_split"]!="train" or row["source_sha256"] in frozen_hashes:
                raise ValueError("AI4VA page overlaps frozen evaluation or is not official train data")
            if split=="test":
                reserved_hashes.add(row["source_sha256"])
                continue
            if row["polygon_conversion_status"]!="accepted":
                continue
            name=f"ai4va_train_{row['source_image_id']:04d}_{Path(row['source_file_name']).stem}"
            path=(folder/"segments/images"/split/(name+Path(row["image_path"]).suffix.lower())).resolve()
            if path in metadata:
                raise ValueError("Duplicate AI4VA prepared page identity")
            metadata[path]=row
        if split!="test":
            list_path=folder/f"segments_{split}.txt"
            verify_hash(list_path, artifacts[list_path.name], sources)
            paths=paths_from_list(list_path)
            if set(paths)!=set(metadata):
                raise ValueError("AI4VA list differs from accepted pages of its source split")
            for path in paths:
                for artifact in (path, label_path(path)):
                    key=artifact.relative_to(folder).as_posix()
                    verify_hash(artifact, artifacts[key], sources)
            rows[split]=[(path, metadata[path]) for path in paths]
    return rows, reserved_hashes, provenance, plan


def verify_manga_labels(path, row, minimum_iou):
    """Recreate each serialized polygon from its audited RLE with the original producer."""
    from training_scripts.prepare_bubble_segments import cropped_mask, polygon_from_mask

    stored=path.read_text(encoding="utf-8")
    lines=[]
    ious=[]
    for annotation in row["annotations"]:
        if annotation.get("iscrowd", 0):
            raise ValueError(f"Crowd annotation cannot supply Manga replay polygons: {path}")
        mask=cropped_mask(annotation, row)
        line, geometry=polygon_from_mask(mask, annotation["bbox"], row, minimum_iou)
        if line is None:
            raise ValueError(f"Manga replay polygon reconstruction failed its raster IoU gate: {path}")
        lines.append(line)
        ious.append(geometry["iou"])
    expected="\n".join(lines)+"\n"
    if stored!=expected:
        raise ValueError(f"Manga polygon geometry differs from audited RLE reconstruction: {path}")
    return {"method":"Exact serialized-label reconstruction from audited source RLE with original producer",
            "masks":len(lines), "minimum_raster_iou":min(ious) if ious else None,
            "required_raster_iou":minimum_iou}


def inspect_manga(folder, replay_count, seed, sources):
    provenance=load_json(folder/"provenance.json")
    if provenance.get("status")!="complete":
        raise ValueError("Manga preparation is incomplete")
    expected_producer=provenance["source_hashes"].get(str(MANGA_POLYGON_PRODUCER))
    if expected_producer is None:
        expected_producer=provenance["source_hashes"].get(MANGA_POLYGON_PRODUCER.as_posix())
    if expected_producer is None:
        raise ValueError("Original Manga polygon producer is absent from source provenance")
    verify_hash(MANGA_POLYGON_PRODUCER, expected_producer, sources)
    minimum_iou=provenance["polygon_quality"]["minimum_required_iou"]
    if not 0.98<=minimum_iou<=1:
        raise ValueError("Original Manga polygon gate is below the required0.98 raster IoU")
    disjoint_groups(provenance["split_books"], "Manga books")
    rows={}
    for split in ("train", "validation"):
        list_path=folder/f"segments_{split}.txt"
        verify_hash(list_path, provenance["manifest_sha256"][list_path.name], sources)
        selected=deterministic_selection(paths_from_list(list_path), replay_count, seed)
        manifest=folder/f"{split}_masks.jsonl"
        verify_hash(manifest, provenance["exact_mask_sha256"][manifest.name], sources)
        metadata={}
        selected_set=set(selected)
        for row in json_rows(manifest):
            path=Path(row["image_path"]).resolve()
            if path in selected_set:
                if path in metadata or row["book"] not in provenance["split_books"][split]:
                    raise ValueError("Manga replay identity or source book split differs")
                metadata[path]=row
        if selected_set!=set(metadata):
            raise ValueError("Manga replay page missing from exact source manifest")
        for path, row in metadata.items():
            label=label_path(path).resolve(strict=True)
            sources[label.as_posix()]=digest(label)
            row["geometry_verification"]=verify_manga_labels(label, row, minimum_iou)
        rows[split]=[(path, metadata[path]) for path in selected]
    return rows, provenance


def checked_record(path, split, dataset, metadata, segmentation=True):
    label=label_path(path).resolve(strict=True)
    count=label_count(label, segmentation)
    image_hash=digest(path)
    if metadata.get("source_sha256", image_hash)!=image_hash:
        raise ValueError(f"Prepared image differs from annotated source: {path}")
    if "annotations" in metadata and count!=len(metadata["annotations"]):
        raise ValueError(f"Label count differs from source annotations: {path}")
    return {"image_path":path.as_posix(), "label_path":label.as_posix(), "split":split,
            "dataset":dataset, "group":metadata.get("issue_id", metadata.get("book")),
            "group_type":"publication_issue" if dataset=="ai4va" else "book" if dataset=="manga" else None,
            "source_image_id":metadata.get("source_image_id", metadata.get("image_id")),
            "source_split":metadata.get("source_split", split), "image_sha256":image_hash,
            "label_sha256":digest(label), "targets":count, "label_kind":"polygon" if segmentation else "box",
            **({"geometry_verification":metadata["geometry_verification"]} if dataset=="manga" else {})}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def prepare(ai4va=DEFAULT_AI4VA, manga=DEFAULT_MANGA, frozen=DEFAULT_FROZEN,
            output=DEFAULT_OUTPUT, replay_count=128, seed=20260927):
    ai4va, manga, frozen=(Path(path).resolve(strict=True) for path in (ai4va, manga, frozen))
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    sources={str(path):digest(path) for path in (Path(__file__), ai4va/"provenance.json", manga/"provenance.json")}
    ai_rows, reserved, ai_provenance, plan=inspect_ai4va(ai4va, frozen, sources)
    manga_rows, manga_provenance=inspect_manga(manga, replay_count, seed, sources)
    records=[]
    for split, ai_split in (("train", "train"), ("validation", "calibration")):
        records.extend(checked_record(path, split, "ai4va", row) for path, row in ai_rows[ai_split])
        records.extend(checked_record(path, split, "manga", row) for path, row in manga_rows[split])
    current_config_path=manga/"current_boxes.yaml"
    current_config=yaml.safe_load(current_config_path.read_text(encoding="utf-8"))
    sources[str(current_config_path)]=digest(current_config_path)
    box_records=[]
    box_lists={}
    source_root=Path(current_config["path"]).resolve(strict=True)
    for split, key in (("train", "train"), ("validation", "val")):
        list_path=(source_root/current_config[key]).resolve(strict=True)
        if list_path!=manga/f"current_{split}.txt":
            raise ValueError("Current box configuration points outside its original split lists")
        verify_hash(list_path, manga_provenance["manifest_sha256"][list_path.name], sources)
        paths=paths_from_list(list_path)
        if len(paths)!=manga_provenance["current_boxes"][split]["images"]:
            raise ValueError("Current box image count differs from source provenance")
        box_lists[key]=list_path.as_posix()
        box_records.extend(checked_record(path, split, "current_boxes", {}, False) for path in paths)
    seen_paths={}
    seen_hashes={}
    for row in records+box_records:
        if row["image_sha256"] in reserved:
            raise ValueError("Training/calibration image overlaps reserved AI4VA test or frozen external evaluation")
        for field, seen in (("image_path", seen_paths), ("image_sha256", seen_hashes)):
            old=seen.setdefault(row[field], row["split"])
            if old!=row["split"]:
                raise ValueError(f"Training/validation overlap by {field}")
    if any(digest(path)!=expected for path, expected in sources.items()):
        raise ValueError("Input data changed during composition")
    output.mkdir(parents=True, exist_ok=False)
    for split, name in (("train", "train.txt"), ("validation", "val.txt")):
        paths=[row["image_path"] for row in records if row["split"]==split]
        with (output/name).open("x", encoding="utf-8") as stream:
            stream.write("\n".join(paths)+"\n")
    for name, config in (("segments.yaml", {"path":output.as_posix(), "train":"train.txt", "val":"val.txt", "names":{0:"balloon"}}),
                         ("current_boxes.yaml", {"path":source_root.as_posix(), **box_lists, "names":current_config["names"]})):
        with (output/name).open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    write_json(output/"validation_groups.json", {row["image_path"]:row["dataset"] for row in records if row["split"]=="validation"})
    for name, data in (("image_label_manifest.jsonl", records), ("current_boxes_image_label_manifest.jsonl", box_records)):
        with (output/name).open("x", encoding="utf-8") as stream:
            for row in data:
                stream.write(json.dumps(row)+"\n")
    counts={split:{dataset:{"images":sum(row["split"]==split and row["dataset"]==dataset for row in records+box_records),
                           "targets":sum(row["targets"] for row in records+box_records if row["split"]==split and row["dataset"]==dataset)}
                   for dataset in ("ai4va", "manga", "current_boxes")} for split in ("train", "validation")}
    report={"status":"complete", "created_at_utc":datetime.now(timezone.utc).isoformat(), "seed":seed,
            "replay_pages_per_split":replay_count, "selection":"Seeded sample of sorted original Manga lists; independent of model predictions",
            "counts":counts, "source_hashes":sources, "ai4va_issue_splits":plan["issue_splits"],
            "ai4va_split_plan_sha256":ai_provenance["split_plan_sha256"],
            "ai4va_original_provenance":ai_provenance, "manga_original_provenance":manga_provenance,
            "selected_manga_books":{split:sorted({row["group"] for row in records if row["dataset"]=="manga" and row["split"]==split})
                                    for split in ("train", "validation")},
            "reserved_ai4va_image_hashes_checked":len(reserved), "cross_split_path_and_hash_overlap":0,
            "manga_polygon_geometry_verified":True,
            "manga_polygon_producer_sha256":sources[MANGA_POLYGON_PRODUCER.as_posix()],
            "artifact_sha256":{path.name:digest(path) for path in sorted(output.iterdir()) if path.is_file()},
            "limitations":["Calibration is for checkpoint selection; no test list or test YAML entry is exposed.",
                           "Authoritative calibration/test scoring uses complete exact masks, including polygon-rejected pages.",
                           "Current box-only data is provided separately; no rectangle or predicted masks were created.",
                           "Publication-issue and local Manga-book disjointness do not prove upstream pretraining independence."]}
    write_json(output/"provenance.json", report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai4va", default=str(DEFAULT_AI4VA))
    parser.add_argument("--manga", default=str(DEFAULT_MANGA))
    parser.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--replay-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260927)
    args=parser.parse_args()
    result=prepare(args.ai4va, args.manga, args.frozen, args.output, args.replay_count, args.seed)
    print(json.dumps(result["counts"], indent=2))
