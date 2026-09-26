"""Move OpenMantra into an excluded reference collection, preserving its source."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"training_data/open-mantra-dataset"
DESTINATION=ROOT/"training_data/reference_only/openmantra"
ARCHIVE=ROOT/"ARCHIVE/dataset_sources/open-mantra-dataset"
AUDIT=ROOT/"outputs/bubble_training_20260922/openmantra_audit.json"


def checked(path):
    resolved=Path(path).resolve()
    if resolved==ROOT or not resolved.is_relative_to(ROOT):
        raise ValueError(f"Path escapes the intended workspace: {resolved}")
    return resolved


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode("utf-8")


def preflight():
    source=checked(SOURCE)
    destination=checked(DESTINATION)
    archive=checked(ARCHIVE)
    audit_path=checked(AUDIT)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not (source/".git").is_dir():
        raise ValueError("The original cloned Git repository must be preserved")
    for target in (destination, archive):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing collection: {target}")
    temporary=checked(audit_path.with_suffix(".relocation.tmp"))
    if temporary.exists():
        raise FileExistsError(temporary)
    data=json.loads((source/"annotation.json").read_text(encoding="utf-8"))
    normalized=deepcopy(data)
    commit=subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    entries=[]
    sources=set()
    targets=set()
    for book in normalized:
        title=book["book_title"]
        if not re.fullmatch(r"[a-z0-9_]+", title):
            raise ValueError(f"Unsafe book title: {title}")
        for page in book["pages"]:
            index=page["page_index"]
            if type(index) is not int or index<0 or set(page["image_paths"])!={"ja"}:
                raise ValueError(f"Unexpected page identity: {title}/{index}")
            old_relative=page["image_paths"]["ja"]
            old=checked(source/old_relative)
            relative=Path("images")/title/f"{title}_page_{index:04d}.jpg"
            new=checked(destination/relative)
            if not old.is_relative_to(source/"images") or old.suffix.lower()!=".jpg":
                raise ValueError(f"Unexpected source image: {old}")
            if not old.is_file():
                raise FileNotFoundError(old)
            if old in sources or new in targets or new.exists():
                raise FileExistsError(f"Duplicate or occupied image path: {old} -> {new}")
            sources.add(old)
            targets.add(new)
            entries.append({
                "book_title":title, "original_page_index":index,
                "original_annotation_path":old_relative,
                "original_path":str(old.relative_to(ROOT)),
                "new_annotation_path":relative.as_posix(),
                "new_path":str(new.relative_to(ROOT)),
                "sha256":digest(old), "bytes":old.stat().st_size,
            })
            page["image_paths"]["ja"]=relative.as_posix()
    actual={checked(path) for path in (source/"images").rglob("*") if path.is_file()}
    if len(entries)!=214 or actual!=sources:
        raise ValueError("Expected exactly the214 referenced images and no additional image files")
    original_files={name:digest(source/name) for name in ("annotation.json", "README.md", "LICENSE.md")}
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    audit_hashes={row["sha256"] for row in audit["image_records"]}
    if audit_hashes!={entry["sha256"] for entry in entries}:
        raise ValueError("The existing audit does not describe these exact source images")
    metadata={
        "annotations.json":json_bytes(normalized),
        "SOURCE_README.txt":(source/"README.md").read_bytes(),
        "LICENSE.txt":(source/"LICENSE.md").read_bytes(),
    }
    for name in (*metadata, "manifest.json"):
        target=checked(destination/name)
        if target.exists():
            raise FileExistsError(target)
    manifest={
        "status":"excluded_by_user",
        "purpose":"Reference collection only; excluded from training, calibration and evaluation.",
        "source_url":"https://github.com/mantra-inc/open-mantra-dataset",
        "source_commit":commit,
        "organized_at_utc":datetime.now(timezone.utc).isoformat(),
        "image_count":len(entries), "book_count":len(normalized),
        "original_repository":str(source.relative_to(ROOT)),
        "current_collection":str(destination.relative_to(ROOT)),
        "archived_repository":str(archive.relative_to(ROOT)),
        "original_metadata_sha256":original_files,
        "normalized_metadata_sha256":{
            name:hashlib.sha256(content).hexdigest() for name, content in metadata.items()
        },
        "mapping":entries,
    }
    return manifest, metadata, audit


def verify_images(manifest):
    for entry in manifest["mapping"]:
        path=checked(ROOT/entry["new_path"])
        if path.stat().st_size!=entry["bytes"] or digest(path)!=entry["sha256"]:
            raise RuntimeError(f"Image integrity verification failed: {path}")
    images=[path for path in (DESTINATION/"images").rglob("*") if path.is_file()]
    if len(images)!=manifest["image_count"]:
        raise RuntimeError("Reference image count changed during relocation")


def apply(manifest, metadata, audit):
    moved=[]
    archive_moved=False
    try:
        checked(DESTINATION).mkdir(parents=True, exist_ok=False)
        for entry in manifest["mapping"]:
            source=checked(ROOT/entry["original_path"])
            target=checked(ROOT/entry["new_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
            moved.append((source, target))
        verify_images(manifest)
        if any(path.is_file() for path in (SOURCE/"images").rglob("*")):
            raise RuntimeError("Source repository still contains image files")
        for name, content in metadata.items():
            with checked(DESTINATION/name).open("xb") as stream:
                stream.write(content)
        with checked(DESTINATION/"manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
        checked(ARCHIVE).parent.mkdir(parents=True, exist_ok=True)
        if checked(ARCHIVE).exists():
            raise FileExistsError(ARCHIVE)
        checked(SOURCE).rename(checked(ARCHIVE))
        archive_moved=True
        for name, expected in manifest["original_metadata_sha256"].items():
            if digest(checked(ARCHIVE/name))!=expected:
                raise RuntimeError(f"Archived source metadata changed: {name}")
        if SOURCE.exists() or not (ARCHIVE/".git").is_dir():
            raise RuntimeError("Raw clone relocation or Git history preservation failed")
        path_map={entry["sha256"]:entry for entry in manifest["mapping"]}
        for row in audit["image_records"]:
            entry=path_map[row["sha256"]]
            row["original_image_path"]=row["image"]
            row["current_image_path"]=entry["new_path"]
        audit["relocation"]={
            "status":"excluded_by_user",
            "organized_at_utc":manifest["organized_at_utc"],
            "current_collection":str(DESTINATION),
            "archived_repository":str(ARCHIVE),
            "mapping_manifest":str(DESTINATION/"manifest.json"),
            "image_count":manifest["image_count"],
            "all_image_hashes_verified":True,
            "note":"Historical paths, counts and measurements remain unchanged. "
                   "Use current_image_path or the manifest mapping to find relocated images. "
                   "The user excluded this collection from training and evaluation.",
        }
        temporary=checked(AUDIT.with_suffix(".relocation.tmp"))
        with temporary.open("xb") as stream:
            stream.write(json_bytes(audit))
        temporary.replace(checked(AUDIT))
    except Exception:
        # Restore original image paths where possible without deleting any file.
        if archive_moved and ARCHIVE.exists() and not SOURCE.exists():
            checked(ARCHIVE).rename(checked(SOURCE))
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                checked(source).parent.mkdir(parents=True, exist_ok=True)
                checked(target).rename(checked(source))
        raise
    verify_images(manifest)
    print(json.dumps({"status":manifest["status"], "images_verified":manifest["image_count"],
                      "collection":str(DESTINATION), "archive":str(ARCHIVE),
                      "raw_clone_exists":SOURCE.exists()}, indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args=parser.parse_args()
    planned, metadata, audit=preflight()
    if args.apply:
        apply(planned, metadata, audit)
    else:
        print(json.dumps({"status":"preflight_passed", "images":planned["image_count"],
                          "destination":str(DESTINATION), "archive":str(ARCHIVE)}, indent=2))
