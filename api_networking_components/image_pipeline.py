"""Japanese manga OCR, structured translation, and verified image export."""

import argparse
import json
from pathlib import Path
import tempfile
import time
import shutil
import statistics
import zipfile

from .ocr_regions import extract_regions, load_models
from .text_replacement import replace_and_zip
from .translation_API import translate_regions


ROOT=Path(__file__).resolve().parents[1]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_images(paths, output_dir, *, detector, ocr):
    """Prepare local OCR evidence without sending any text to an external provider."""
    output=Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries=[]
    if (output/"ocr.json").exists():
        entries=json.loads((output/"ocr.json").read_text(encoding="utf-8"))["images"]
        if [e["path"] for e in entries]!=[str(Path(p).resolve()) for p in paths[:len(entries)]]:
            raise ValueError("Saved OCR sources differ from the requested images.")
    for index, path in enumerate(paths[len(entries):], len(entries)+1):
        image_id=f"image_{index:04d}"
        regions=extract_regions(path, detector, ocr, image_id)
        entries.append({"path": str(Path(path).resolve()), "image_id": image_id, "regions": regions})
        save_json(output/"ocr.json", {"images": entries, "complete": False})
        print(f"OCR {index}/{len(paths)}: {len(regions)} regions", flush=True)
    save_json(output/"ocr.json", {"images": entries, "complete": True})
    return entries


def process_images(paths, output_dir, *, detector, ocr, font_path, key=None, target="en", prepared=False):
    """Translate prepared local OCR and keep the audit files plus a verified ZIP."""
    output=Path(output_dir)
    if prepared:
        entries=json.loads((output/"ocr.json").read_text(encoding="utf-8"))["images"]
        if [e["path"] for e in entries]!=[str(Path(p).resolve()) for p in paths]:
            raise ValueError("Prepared OCR is incomplete or belongs to different images.")
    else:
        entries=prepare_images(paths, output, detector=detector, ocr=ocr)
    groups=[]
    group=[]
    for entry in entries:
        rows=entry["regions"]
        if group and len(group)+len(rows)>20:
            groups.append(group)
            group=[]
        group.extend(rows)
    if group:
        groups.append(group)
    translations={}
    responses=[]
    cache=output/"translation_responses.json"
    if prepared and cache.exists():
        responses=json.loads(cache.read_text(encoding="utf-8"))
        for response in responses:
            translations.update({row["id"]: row["text"] for row in response["regions"]})
    for index, rows in enumerate(groups, 1):
        if all(row["id"] in translations for row in rows):
            continue
        for attempt in range(3):
            try:
                payload=[{name: row[name] for name in ("id", "image_id", "text", "box") if name in row} for row in rows]
                response=translate_regions(payload, target_language=target, key=key,
                                           model="minimax/minimax-m3:free")
                break
            except RuntimeError as error:
                rate_limited="HTTP 429" in str(error)
                invalid_json="invalid textbox JSON" in str(error)
                if not (rate_limited or invalid_json) or attempt==2:
                    raise
                delay=30 if rate_limited else 1
                print(f"Provider {'rate limited' if rate_limited else 'returned invalid JSON'}; retry {attempt+1}/2 after {delay} seconds", flush=True)
                time.sleep(delay)
        responses.append(response)
        translations.update({row["id"]: row["text"] for row in response["regions"]})
        save_json(output/"translation_responses.json", responses)
        print(f"Translation {index}/{len(groups)} complete", flush=True)
        if index<len(groups):
            time.sleep(4)
    for entry in entries:
        for region in entry["regions"]:
            region["source_text"]=region["text"]
            region["text"]=translations[region["id"]]
            if "detector_box" in region:
                l, t, r, b=region["detector_box"]
                inset_x, inset_y=round((r-l)*0.17), round((b-t)*0.17)
                x1, y1, x2, y2=region["box"]
                region["layout_box"]=[min(x1, l+inset_x), min(y1, t+inset_y),
                                      max(x2, r-inset_x), max(y2, b-inset_y)]
            glyph_heights=[rect[3]-rect[1] for rect in region.get("wipe_rects", []) if rect[3]-rect[1]>5]
            if glyph_heights:
                region["max_font_size"]=max(12, min(32, round(statistics.median(glyph_heights)*1.5)))
    save_json(output/"translated_regions.json", {"images": entries})
    archive=replace_and_zip(entries, output/"translated.zip", font_path=font_path)
    summary={"images": len(entries), "regions": len(translations), "target": target,
             "model": "minimax/minimax-m3:free", "usage": [r["usage"] for r in responses],
             "zip": str(archive), "status": "complete",
             "limitations": "Japanese OCR; heuristic bubble/text boxes and reading order; manual semantic review required."}
    save_json(output/"summary.json", summary)
    return summary


def extract_zip(zip_path, destination):
    """Extract only image entries under a caller-owned directory; never delete uploads."""
    root=Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    paths=[]
    with zipfile.ZipFile(zip_path) as archive:
        infos=archive.infolist()
        if len(infos)>1000 or sum(i.file_size for i in infos)>1024*1024*1024:
            raise ValueError("ZIP exceeds 1000 entries or 1 GiB uncompressed.")
        for index, info in enumerate(infos, 1):
            if info.is_dir() or Path(info.filename).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            # Generated names avoid traversal, duplicate names, and platform-specific paths.
            path=root/f"{index:06d}{Path(info.filename).suffix.lower()}"
            with archive.open(info) as source, path.open("xb") as output:
                shutil.copyfileobj(source, output)
            paths.append(path)
    if not paths:
        raise ValueError("ZIP has no supported images.")
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="en")
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT/"models/best/speech_bubble_yolo_s_gpu.pt")
    args=parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=".upload-", dir=args.output.parent)).resolve()
    try:
        paths=extract_zip(args.zip, staging/"images")
        detector, ocr=load_models(args.weights)
        result=process_images(paths, args.output, detector=detector, ocr=ocr,
                              font_path=args.font, target=args.target)
    except Exception:
        print(f"Upload recovery folder retained: {staging}", flush=True)
        raise
    if staging.is_symlink() or staging.parent!=args.output.parent.resolve():
        raise RuntimeError("Unexpected upload cleanup path.")
    shutil.rmtree(staging)
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
