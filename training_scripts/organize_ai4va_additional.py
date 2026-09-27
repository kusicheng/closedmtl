"""Organize the downloaded official AI4VA train pages outside frozen eval issues."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import zipfile

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_scripts.prepare_ai4va_evaluation import digest

DATASET=ROOT/"training_data/ai4va_additional"
AUDIT=ROOT/"outputs/bubble_training_20260922/ai4va_audit/prospective_train_audit.json"
ARCHIVE_SHA256="a53ebb55fdd0c8fa2ab0648e30dd13a7aff1a771bf4d7e6e9b843facbecb1874"
ARCHIVE_BYTES=1230669651
NEEDS_REVIEW={154:"missing_source_polygon", 272:"crowd_bubble"}
PAGE_PATTERN=re.compile(r"Vaillant_(\d{4})_(\d{4})_(\d{2})_(\d{2})-(\d+)\.png")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_members(archive):
    members={}
    for member in archive.infolist():
        name=member.filename
        path=PurePosixPath(name)
        mode=member.external_attr>>16
        if ("\\" in name or ":" in name or path.is_absolute() or ".." in path.parts
                or stat.S_ISLNK(mode) or name in members):
            raise ValueError(f"Unsafe or duplicate ZIP member: {name}")
        members[name]=member
    return members


def save_bytes(path, data):
    expected=hashlib.sha256(data).hexdigest()
    if path.exists():
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f"Existing organized file differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    if digest(path)!=expected:
        raise ValueError(f"Organized write verification failed: {path}")
    return expected


def organize(dataset_root=DATASET, audit_path=AUDIT):
    dataset=Path(dataset_root).resolve(strict=True)
    mapping_path=dataset/"mapping_manifest.json"
    if mapping_path.exists():
        raise FileExistsError(f"Completed mapping exists; do not overwrite: {mapping_path}")
    audit=load_json(audit_path)
    source_path=Path(audit["source_path"]).resolve(strict=True)
    frozen_path=Path(audit["frozen_mapping_path"]).resolve(strict=True)
    sources={source_path:audit["source_sha256"], frozen_path:audit["frozen_mapping_sha256"]}
    if any(digest(path)!=expected for path, expected in sources.items()):
        raise ValueError("Source metadata changed after prospective audit")
    source=load_json(source_path)
    frozen=load_json(frozen_path)
    frozen_issues={row["issue_id"] for row in frozen["images"] if row.get("sha256")}
    frozen_hashes={row["sha256"] for row in frozen["images"] if row.get("sha256")}
    prospective=[row for row in audit["pages"] if not row["overlaps_frozen_evaluation_issue"]]
    images={row["id"]:row for row in source["images"]}
    if len(images)!=len(source["images"]) or len({row["image_id"] for row in prospective})!=len(prospective):
        raise ValueError("Duplicate source image IDs")
    archive_path=dataset/"_downloads/train-images-comics.zip"
    if archive_path.stat().st_size!=ARCHIVE_BYTES or digest(archive_path)!=ARCHIVE_SHA256:
        raise ValueError("Downloaded official image archive size or SHA256 differs")
    download=load_json(Path(str(archive_path)+".source.json"))
    if download["bytes"]!=ARCHIVE_BYTES or download["sha256"]!=ARCHIVE_SHA256:
        raise ValueError("Download metadata differs from checked archive")
    rows=[]
    with zipfile.ZipFile(archive_path) as archive:
        members=safe_members(archive)
        for audited in sorted(prospective, key=lambda row:row["image_id"]):
            ident=audited["image_id"]
            image=images[ident]
            filename=image["file_name"]
            match=PAGE_PATTERN.fullmatch(filename)
            if match is None or filename!=audited["file_name"]:
                raise ValueError(f"Unexpected page filename for source ID {ident}")
            issue, year, month, day, page=match.groups()
            issue_id=f"vaillant_{issue}_{year}_{month}_{day}"
            if issue_id!=audited["issue_id"] or issue_id in frozen_issues:
                raise ValueError(f"Publication issue differs or overlaps frozen evaluation: {ident}")
            member_name=f"train/{filename}"
            if member_name not in members or members[member_name].is_dir():
                raise ValueError(f"Prospective image absent from ZIP: {member_name}")
            data=archive.read(members[member_name])
            with Image.open(BytesIO(data)) as opened:
                opened.load()
                if opened.format!="PNG" or opened.size!=(image["width"], image["height"]):
                    raise ValueError(f"Image dimensions/format differ from source annotations: {ident}")
            sha256=hashlib.sha256(data).hexdigest()
            if sha256 in frozen_hashes:
                raise ValueError("Image bytes overlap frozen evaluation")
            relative=Path("images")/issue_id/filename
            target=(dataset/relative).resolve()
            if not target.is_relative_to(dataset):
                raise ValueError("Organized image escapes dataset root")
            save_bytes(target, data)
            rows.append({"source_split":"train", "image_id":ident, "source_file_name":filename,
                         "issue_id":issue_id, "issue_number":issue, "publication_date":f"{year}-{month}-{day}",
                         "page_number":page, "width":image["width"], "height":image["height"],
                         "organized_path":target.as_posix(), "relative_path":relative.as_posix(), "sha256":sha256,
                         "availability":"verified", "annotation_count":audited["annotation_count"],
                         "bubble_count":audited["bubble_count"], "bubble_mask_count":audited["known_nonempty_masks"],
                         "archive_member":member_name, "archive_crc32":members[member_name].CRC,
                         "annotation_review_status":NEEDS_REVIEW.get(ident, "eligible_for_prospective_split"),
                         "joint_split_eligible":ident not in NEEDS_REVIEW})
    target_source=dataset/"annotations/source_train.json"
    if target_source.exists():
        if digest(target_source)!=audit["source_sha256"]:
            raise ValueError("Existing copied annotations differ")
    else:
        target_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_source)
    if digest(target_source)!=audit["source_sha256"] or any(digest(p)!=h for p, h in sources.items()):
        raise ValueError("Source metadata changed during organization")
    report={"status":"complete", "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "usage":"Additional issue-disjoint adaptation/calibration/prospective test pool; splits not assigned here",
            "official_repository":"https://github.com/IVRL/AI4VA", "dataset_root":dataset.as_posix(),
            "organization":"images/<publication_issue>/<original_descriptive_filename>; unchanged image bytes",
            "source_archive":{"path":archive_path.as_posix(), "bytes":ARCHIVE_BYTES, "sha256":ARCHIVE_SHA256,
                              "public_url":download["public_url"], "file_id":download["file_id"]},
            "source_annotations":{"original_path":source_path.as_posix(), "organized_path":target_source.as_posix(),
                                  "sha256":audit["source_sha256"]},
            "frozen_evaluation_mapping":{"path":frozen_path.as_posix(), "sha256":audit["frozen_mapping_sha256"]},
            "prospective_audit":{"path":str(Path(audit_path).resolve()), "sha256":digest(audit_path)},
            "counts":{"images":len(rows), "issues":len({row["issue_id"] for row in rows}),
                      "needs_annotation_review":sum(not row["joint_split_eligible"] for row in rows)},
            "categories":source["categories"], "images":rows,
            "issue_page_counts":dict(sorted(Counter(row["issue_id"] for row in rows).items())),
            "source_license_note":frozen.get("license_note"), "license_source":frozen.get("license_source"),
            "limitation":"Publication-issue independence does not establish independent story series or artists."}
    source_text=("AI4VA additional images\nOfficial repository: https://github.com/IVRL/AI4VA\n"
                 "Only official train pages outside all frozen 62-page evaluation publication issues are organized here.\n"
                 "Images keep descriptive publication/date/page filenames and unchanged bytes.\n"
                 "Source image IDs 154 and 272 require annotation review and are excluded before joint split assignment.\n"
                 "mapping_manifest.json records archive members, hashes, dimensions, source IDs and attribution.\n"
                 f"License source: {frozen.get('license_source')}\nLicense note: {frozen.get('license_note')}\n")
    save_bytes(dataset/"SOURCE.txt", source_text.encode("utf-8"))
    with mapping_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DATASET))
    parser.add_argument("--audit", default=str(AUDIT))
    args=parser.parse_args()
    result=organize(args.dataset_root, args.audit)
    print(json.dumps(result["counts"], indent=2))
