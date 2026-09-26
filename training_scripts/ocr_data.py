"""Prepare local Manga109-s manifests without copying or publishing images."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def crop(row):
    with Image.open(row["image"]) as image:
        return image.crop(row["box"]).convert("L").convert("RGB")


def prepare(root, output, annotation_version="annotations", seed=20260914):
    root=Path(root).resolve()
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    titles=root.joinpath("books.txt").read_text(encoding="utf-8-sig").splitlines()
    titles=[title.strip() for title in titles if title.strip()]
    if len(titles)!=len(set(titles)):
        raise ValueError("Duplicate book titles")
    random.Random(seed).shuffle(titles)
    heldout=max(1, round(len(titles)*0.1))
    splits={"test":titles[:heldout], "validation":titles[heldout:2*heldout],
            "train":titles[2*heldout:]}
    report={"root":str(root), "seed":seed, "annotation_version":annotation_version,
            "readme_sha256":digest(root/"readme.txt"), "splits":{},
            "source":"User-supplied Manga109-s, released 2026-05-21",
            "license":"Local research use. No dataset redistribution. Attribute Manga109-s in published results/models.",
            "teacher_overlap":"Unknown: original teacher used an unseeded crop split; these are student-heldout books.",
            "padding_pixels":10, "annotation_hashes":{}, "missing_unlabeled_pages":[]}
    for split, books in splits.items():
        rows=[]
        pages=0
        skipped=0
        for title in sorted(books):
            xml=root/annotation_version/(title+".xml")
            report["annotation_hashes"][title]=digest(xml)
            for page in ET.parse(xml).findall(".//page"):
                image=root/"images"/title/(f"{int(page.attrib['index']):03d}.jpg")
                if not image.exists() and not page.findall("text"):
                    report["missing_unlabeled_pages"].append(str(image))
                    continue
                with Image.open(image) as source:
                    width, height=source.size
                if (width, height)!=(int(page.attrib["width"]), int(page.attrib["height"])):
                    raise ValueError(f"Image dimensions disagree with XML: {image}")
                pages+=1
                for text in page.findall("text"):
                    label="".join(text.itertext()).strip()
                    if not label:
                        skipped+=1
                        continue
                    a=text.attrib
                    box=[max(0, int(a["xmin"])-10), max(0, int(a["ymin"])-10),
                         min(width, int(a["xmax"])+10), min(height, int(a["ymax"])+10)]
                    if box[0]>=box[2] or box[1]>=box[3]:
                        raise ValueError(f"Invalid crop {title}/{a['id']}")
                    rows.append({"id":title+"/"+a["id"], "book":title,
                                 "image":str(image), "box":box, "text":label})
        random.Random(seed).shuffle(rows)
        path=output/(split+".jsonl")
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        report["splits"][split]={"books":sorted(books), "pages":pages, "crops":len(rows),
                                 "empty_labels_skipped":skipped, "sha256":digest(path)}
    (output/"provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="training_data/manga109_ocr")
    parser.add_argument("--annotation-version", default="annotations")
    args=parser.parse_args()
    prepare(args.root, args.output, args.annotation_version)
