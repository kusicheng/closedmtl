"""Audit local split duplicates without treating different JPEG hashes as independence."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(path, split):
    content=path.read_bytes()
    image=cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    small=cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    square=cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    spectrum=cv2.dct(square)[:8, :8].reshape(-1)[1:]
    phash=spectrum>np.median(spectrum)
    return {"path":str(path), "split":split, "shape":list(image.shape),
            "file_sha256":sha256(content),
            "pixel_sha256":sha256(str(image.shape).encode()+image.tobytes()),
            "source_stem":path.name.split(".rf.")[0],
            "phash":np.packbits(phash).tobytes().hex()}, small, phash


def exact_groups(rows, key):
    groups=defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [[r["path"] for r in group] for group in groups.values()
            if len({r["split"] for r in group})>1]


def main(args):
    root=Path(args.root)
    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    rows=[]
    thumbnails=[]
    hashes=[]
    for split in ("train", "valid", "test"):
        paths=sorted((root/"images"/split).glob("*.jpg"))
        for path in paths:
            row, small, phash=fingerprint(path, split)
            rows.append(row)
            thumbnails.append(small)
            hashes.append(phash)
        print(f"hashed {split}: {len(paths)}", flush=True)
    hashes=np.array(hashes)
    candidates=[]
    for i, row in enumerate(rows):
        distance=np.count_nonzero(hashes[i+1:]!=hashes[i], axis=1)
        for offset in np.flatnonzero(distance<=args.hamming):
            j=i+1+int(offset)
            if rows[j]["split"]==row["split"]:
                continue
            a=thumbnails[i].astype(np.float32)
            b=thumbnails[j].astype(np.float32)
            correlation=float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
            candidates.append({"a":row["path"], "b":rows[j]["path"],
                               "splits":[row["split"], rows[j]["split"]],
                               "phash_hamming":int(distance[offset]),
                               "normalized_mae":float(np.abs(a-b).mean()/255),
                               "gray_correlation":correlation if np.isfinite(correlation) else None})
    candidates.sort(key=lambda pair:(pair["phash_hamming"], pair["normalized_mae"]))
    report={"root":str(root.resolve()), "counts":{split:sum(r["split"]==split for r in rows)
             for split in ("train", "valid", "test")},
            "cross_split_file_hash_groups":exact_groups(rows, "file_sha256"),
            "cross_split_decoded_pixel_groups":exact_groups(rows, "pixel_sha256"),
            "cross_split_source_stem_groups":exact_groups(rows, "source_stem"),
            "phash_hamming_limit":args.hamming, "near_duplicate_candidates":candidates,
            "images":rows,
            "limits":["Perceptual candidates need visual confirmation.",
                      "No duplicate match does not establish book independence or exclude crops.",
                      "Source filenames do not contain authoritative book or chapter IDs."]}
    (output/"split_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key:len(report[key]) for key in
                      ("cross_split_file_hash_groups", "cross_split_decoded_pixel_groups",
                       "cross_split_source_stem_groups", "near_duplicate_candidates")}), flush=True)
    for i, pair in enumerate(candidates[:12]):
        panels=[]
        for key in ("a", "b"):
            image=cv2.imread(pair[key])
            width=max(1, round(image.shape[1]*500/image.shape[0]))
            image=cv2.resize(image, (width, 500), interpolation=cv2.INTER_AREA)
            panel=np.full((550, max(400, width), 3), 255, np.uint8)
            panel[50:, :width]=image
            cv2.putText(panel, Path(pair[key]).parent.name+": "+Path(pair[key]).name[:30],
                        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            panels.append(panel)
        cv2.imwrite(str(output/f"candidate_{i:02d}.jpg"), np.concatenate(panels, axis=1))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="training_data/speech-bubbles-detection-yolo")
    parser.add_argument("--output", default="outputs/bubble_training_20260922/contamination_audit")
    parser.add_argument("--hamming", type=int, default=6)
    main(parser.parse_args())
