"""Optional deterministic sampling of verified training pages, without relabeling."""

import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker

from training_scripts.verify_joint_bubble_data import path_key, sha256


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RECTANGLE_SOURCE=ROOT/"training_data/bubble_joint_20260922/current_train_boxes.jsonl"


def _multiplier(value):
    value=float(value)
    if not math.isfinite(value) or value<1:
        raise ValueError("Sampling multipliers must be finite and at least one")
    return value


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _train_rows(path, datasets):
    result={}
    for row in _rows(path):
        if row["split"]!="train":
            continue
        key=path_key(row["image_path"])
        if key in result or row["dataset"] not in datasets or row["targets"]<0:
            raise ValueError("Invalid or repeated training page in sampling manifest")
        result[key]=row
    if not result:
        raise ValueError("Sampling requires a nonempty training manifest")
    return result


def rectangle_training_pages(source, intended):
    """Verify original categories against the actual one-class training labels."""
    source_rows={}
    rectangle_pages=set()
    for row in _rows(source):
        key=path_key(row["image_path"])
        if key in source_rows or key not in intended or row.get("split", "train")!="train":
            raise ValueError("Rectangle source coverage differs from current training pages")
        expected=intended[key]
        if row["source_sha256"]!=expected["image_sha256"] or sha256(row["image_path"])!=expected["image_sha256"]:
            raise ValueError("Rectangle source image hash differs from composed training manifest")
        label=Path(expected["label_path"])
        if sha256(label)!=expected["label_sha256"]:
            raise ValueError("Current training label hash differs from composed manifest")
        annotations=row["annotations"]
        labels=[line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(annotations)!=expected["targets"] or len(labels)!=len(annotations):
            raise ValueError("Rectangle source target count differs from current training labels")
        width, height=float(row["width"]), float(row["height"])
        if not math.isfinite(width) or not math.isfinite(height) or min(width, height)<=0:
            raise ValueError("Invalid original image dimensions")
        for annotation, values in zip(annotations, labels):
            if annotation["category_id"] not in range(1, 7):
                raise ValueError("Unknown original bubble category")
            x, y, w, h=map(float, annotation["bbox"])
            if not all(math.isfinite(value) for value in (x, y, w, h)) or min(w, h)<=0:
                raise ValueError("Invalid original box geometry")
            normalized=((x+w/2)/width, (y+h/2)/height, w/width, h/height)
            if len(values)!=5 or values[0]!="0" or any(
                    not math.isfinite(float(value)) or abs(float(value)-target)>1e-8
                    for value, target in zip(values[1:], normalized)):
                raise ValueError("Original categories do not map to actual one-class box labels")
            if annotation["category_id"]==4:
                rectangle_pages.add(key)
        source_rows[key]=row
    if set(source_rows)!=set(intended):
        raise ValueError("Rectangle source coverage differs from current training pages")
    return rectangle_pages


def build_sampling_policy(data_root, ai4va_positive_weight=1.0, rectangle_weight=1.0,
                          rectangle_source=DEFAULT_RECTANGLE_SOURCE, seed=20260927):
    root=Path(data_root).resolve(strict=True)
    ai_weight, box_weight=_multiplier(ai4va_positive_weight), _multiplier(rectangle_weight)
    mask_path=root/"image_label_manifest.jsonl"
    box_path=root/"current_boxes_image_label_manifest.jsonl"
    sources={str(path):sha256(path) for path in (mask_path, box_path)}
    masks=_train_rows(mask_path, {"ai4va", "manga"})
    boxes=_train_rows(box_path, {"current_boxes"})
    rectangle_pages=set()
    if box_weight>1:
        rectangle_source=Path(rectangle_source).resolve(strict=True)
        sources[str(rectangle_source)]=sha256(rectangle_source)
        rectangle_pages=rectangle_training_pages(rectangle_source, boxes)
    groups={}
    for name, records, multiplier, offset in (("masks", masks, ai_weight, 101),
                                               ("current_boxes", boxes, box_weight, 102)):
        pages=[]
        for key, row in sorted(records.items()):
            eligible=(row["dataset"]=="ai4va" and row["targets"]>0) if name=="masks" else key in rectangle_pages
            weight=multiplier if eligible else 1.0
            pages.append({"image_path":row["image_path"], "image_sha256":row["image_sha256"],
                          "targets":row["targets"], "weight":weight,
                          "reason":"ai4va_positive" if name=="masks" and eligible else
                                   "source_category_4_rectangle" if eligible else "default"})
        groups[name]={"enabled":multiplier>1, "multiplier":multiplier, "seed":int(seed)+offset,
                      "replacement":True, "num_samples":len(pages),
                      "boosted_pages":sum(page["weight"]>1 for page in pages), "pages":pages}
    if any(sha256(path)!=expected for path, expected in sources.items()):
        raise ValueError("Sampling source changed while constructing policy")
    return {"version":1, "training_only":True, "source_sha256":sources, "groups":groups,
            "method":"WeightedRandomSampler with replacement; no duplicated dataset files or labels; "
                     "one dataset-length draw sequence per loader cycle; separate private generators."}


def weights_for_dataset(dataset, group):
    paths=[path_key(path) for path in dataset.im_files]
    policy={path_key(page["image_path"]):page["weight"] for page in group["pages"]}
    if len(policy)!=len(group["pages"]) or len(paths)!=len(set(paths)) or set(paths)!=set(policy):
        raise ValueError("Actual loader image coverage differs from sampling policy")
    if len(dataset)!=group["num_samples"]:
        raise ValueError("Actual loader length differs from sampling policy")
    return [policy[path] for path in paths]


def make_weighted_loader(dataset, batch, workers, group, pin_memory=False):
    """Build the sampler before InfiniteDataLoader creates its cached iterator."""
    weights=weights_for_dataset(dataset, group)
    if not weights or batch<1 or workers<0:
        raise ValueError("Invalid weighted loader dimensions")
    if any(not math.isfinite(weight) or weight<=0 for weight in weights):
        raise ValueError("Sampling weights must be finite and positive")
    sampler_generator=torch.Generator().manual_seed(group["seed"])
    worker_generator=torch.Generator().manual_seed(group["seed"]+10000)
    sampler=WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True,
                                  generator=sampler_generator)
    worker_count=min(os.cpu_count() or 1, workers)
    return InfiniteDataLoader(dataset, batch_size=min(batch, len(dataset)), sampler=sampler,
                              shuffle=False, num_workers=worker_count,
                              prefetch_factor=4 if worker_count else None,
                              pin_memory=pin_memory, collate_fn=getattr(dataset, "collate_fn", None),
                              worker_init_fn=seed_worker, generator=worker_generator, drop_last=False)
