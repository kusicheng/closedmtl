"""Run live, randomized Japanese OCR/translation batches for manual review."""

import argparse
from datetime import datetime
import getpass
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT/"models/huggingface"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"models/ultralytics_config"))

from api_networking_components.image_pipeline import prepare_images, process_images, save_json
from api_networking_components.ocr_regions import load_models


def make_plan(dataset, batches, seed):
    rng=random.Random(seed)
    sizes=rng.sample(range(1, 21), batches)
    groups={}
    seen=set()
    for path in sorted(Path(dataset).glob("*/ja_*.jpg")):
        identity=path.name.split(".rf.")[0]
        if identity in seen:
            continue
        seen.add(identity)
        group=re.sub(r"_\d+_png_jpg$", "", identity)
        groups.setdefault(group, []).append(path)
    plan=[]
    for size in sizes:
        available=[name for name, paths in groups.items() if paths]
        if len(available)<size:
            raise ValueError("Not enough distinct Japanese chapter groups for unrelated batch sampling.")
        paths=[]
        for name in rng.sample(available, size):
            path=rng.choice(groups[name])
            groups[name].remove(path)
            paths.append(str(path.resolve()))
        plan.append({"size": size, "sources": paths})
    return plan


def review_files(batch, paths):
    from PIL import Image

    result=batch/"result"
    rendered=batch/"translated"
    rendered.mkdir(exist_ok=True)
    with zipfile.ZipFile(result/"translated.zip") as archive:
        for name in archive.namelist():
            if name.endswith(".png") and Path(name).name==name:
                (rendered/name).write_bytes(archive.read(name))
    entries=json.loads((result/"translated_regions.json").read_text(encoding="utf-8"))["images"]
    document=['<!doctype html><meta charset="utf-8"><title>OCR translation review</title>',
              '<style>body{font-family:sans-serif}img{max-width:46%;vertical-align:top}pre{white-space:pre-wrap}</style>',
              '<h1>Japanese to English — manual review</h1>']
    for index, (source, entry) in enumerate(zip(paths, entries), 1):
        output=next(rendered.glob(f"{index:06d}_*.png"))
        source_relative=Path(os.path.relpath(source, batch)).as_posix()
        output_relative=output.relative_to(batch).as_posix()
        document.append(f'<h2>Image {index}</h2><img src="{html.escape(source_relative)}"><img src="{html.escape(output_relative)}">')
        for region in entry["regions"]:
            document.append('<pre>'+html.escape(f"{region['id']} {region['box']}\nOCR: {region['source_text']}\nEN: {region['text']}")+'</pre>')
        if not entry["regions"]:
            document.append('<p>No accepted regions. Inspect for missed text.</p>')
        # Small single-image previews allow the assistant to inspect at most five outputs.
        with Image.open(output) as image:
            image.thumbnail((900, 1200))
            image.save(rendered/f"preview_{index:03d}.png")
    (batch/"review.html").write_text("\n".join(document), encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ocr-only", action="store_true", help="Prepare local crops and OCR only; no translation requests.")
    parser.add_argument("--revision", help="Keep corrected results under a named revision inside each existing batch.")
    args=parser.parse_args()
    if not 1<=args.batches<=20:
        parser.error("batches must be from 1 to 20")
    if args.revision and (not args.resume or not re.fullmatch(r"revision_[a-zA-Z0-9_]+", args.revision)):
        parser.error("A revision needs --resume and a name such as revision_02")
    root=args.resume or args.output or ROOT/"test_output"/datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.resume:
        saved=json.loads((root/"plan.json").read_text(encoding="utf-8"))
    else:
        root.mkdir(parents=True, exist_ok=False)
        seed=secrets.randbits(64)
        saved={"seed": seed, "plan": make_plan(ROOT/"training_data/scantrad_merged", args.batches, seed),
               "sampling": "Distinct random batch sizes 1..20; unique source pages, different chapter groups within each batch. Images treated independently.",
               "python": sys.executable, "global_environment": sys.prefix==sys.base_prefix}
        save_json(root/"plan.json", saved)
    print(f"Review run: {root}; sizes: {[p['size'] for p in saved['plan']]}", flush=True)
    key=None if args.ocr_only else os.getenv("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter key: ")
    def complete_ocr(path, count):
        return path.exists() and len(json.loads(path.read_text(encoding="utf-8"))["images"])==count
    needs_ocr=any(not complete_ocr(root/f"batch_{i:02d}_{p['size']:02d}_images"/(args.revision or "")/"result/ocr.json", p["size"])
                  for i, p in enumerate(saved["plan"], 1))
    detector, ocr=load_models(ROOT/"models/best/speech_bubble_yolo_s_gpu.pt") if needs_ocr else (None, None)
    reports=[]
    for number, item in enumerate(saved["plan"], 1):
        original_batch=root/f"batch_{number:02d}_{item['size']:02d}_images"
        batch=original_batch/(args.revision or "")
        batch.mkdir(parents=True, exist_ok=True)
        if (batch/"result/summary.json").exists():
            reports.append(json.loads((batch/"result/summary.json").read_text(encoding="utf-8")))
            continue
        sources=original_batch/"sources"
        sources.mkdir(exist_ok=True)
        paths=[]
        for index, original in enumerate(item["sources"], 1):
            path=sources/f"{index:03d}_{Path(original).name}"
            if not path.exists():
                shutil.copy2(original, path)
            paths.append(path)
        save_json(batch/"sources.json", [{"path": str(p), "original": o,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p, o in zip(paths, item["sources"])])
        print(f"BATCH {number}: {len(paths)} images", flush=True)
        try:
            prepared=complete_ocr(batch/"result/ocr.json", len(paths))
            if args.ocr_only:
                if not prepared:
                    prepare_images(paths, batch/"result", detector=detector, ocr=ocr)
                report={"status": "ocr_prepared", "images": len(paths)}
                reports.append(report)
                save_json(root/f"run_summary{('_'+args.revision) if args.revision else ''}.json", reports)
                continue
            report=process_images(paths, batch/"result", detector=detector, ocr=ocr,
                                  font_path=Path("C:/Windows/Fonts/arial.ttf"), key=key, prepared=prepared)
            review_files(batch, paths)
            reports.append(report)
        except Exception as error:
            # Never write provider exceptions containing authentication details.
            message=str(error).replace(key, "[REDACTED]") if key else str(error)
            report={"status": "failed", "error_type": type(error).__name__, "message": message}
            save_json(batch/"failure.json", report)
            reports.append(report)
            print(f"Batch {number} failed: {report['message']}", flush=True)
        save_json(root/f"run_summary{('_'+args.revision) if args.revision else ''}.json", reports)
    print(f"Finished: {sum(r['status']=='complete' for r in reports)}/{len(reports)} batches complete", flush=True)
    return int(any(r["status"] not in {"complete", "ocr_prepared"} for r in reports))


if __name__=="__main__":
    raise SystemExit(main())
