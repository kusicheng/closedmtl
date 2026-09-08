"""Build a local entry page and verification report for a review run."""

import hashlib
import html
import json
from pathlib import Path
import sys
import zipfile


def build(root):
    root=Path(root).resolve()
    plan=json.loads((root/"plan.json").read_text(encoding="utf-8"))
    sizes=[batch["size"] for batch in plan["plan"]]
    assert len(sizes)==len(set(sizes)) and all(1<=n<=20 for n in sizes)
    originals=[s for batch in plan["plan"] for s in batch["sources"]]
    assert len(originals)==len(set(originals))
    rows=[]
    tokens=0
    cost=0
    checked=[]
    seen_responses=set()
    for index, batch in enumerate(plan["plan"], 1):
        directory=root/f"batch_{index:02d}_{batch['size']:02d}_images"
        links=[]
        for version in [directory]+sorted(directory.glob("revision_*")):
            summary=version/"result/summary.json"
            if not summary.exists():
                continue
            result=json.loads(summary.read_text(encoding="utf-8"))
            archive_path=version/"result/translated.zip"
            with zipfile.ZipFile(archive_path) as archive:
                assert archive.testzip() is None
                names=[name for name in archive.namelist() if name.endswith(".png")]
                assert len(names)==batch["size"]
                manifest=json.loads(archive.read("manifest.json"))
                assert len(manifest["images"])==batch["size"]
                for entry in manifest["images"]:
                    name=entry["output"]
                    assert hashlib.sha256(archive.read(name)).digest()==hashlib.sha256((version/"translated"/name).read_bytes()).digest()
            for source in json.loads((version/"sources.json").read_text(encoding="utf-8")):
                assert hashlib.sha256(Path(source["original"]).read_bytes()).hexdigest()==source["sha256"]
            assert result["model"]=="minimax/minimax-m3:free"
            responses_path=version/"result/translation_responses.json"
            responses=json.loads(responses_path.read_text(encoding="utf-8")) if responses_path.exists() else []
            for response in responses:
                fingerprint=hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
                if fingerprint in seen_responses:
                    continue
                seen_responses.add(fingerprint)
                usage=response["usage"]
                tokens+=usage.get("total_tokens", 0)
                cost+=usage.get("cost", 0)
            label="Original" if version==directory else version.name
            relative=(version/"review.html").relative_to(root).as_posix()
            links.append(f'<a href="{html.escape(relative)}">{label}</a> ({result["regions"]} boxes)')
            checked.append({"batch": index, "version": label, "images": result["images"], "regions": result["regions"]})
        rows.append(f'<tr><td>{index}</td><td>{batch["size"]}</td><td>'+" · ".join(links)+'</td></tr>')
    document='''<!doctype html><meta charset="utf-8"><title>Manga translation review</title>
<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}a{color:#1454aa}</style>
<h1>Japanese → English review</h1><p>Global Python · manga-ocr · minimax/minimax-m3:free</p>
<p>Five batches use 62 unique pages with distinct random batch sizes. Each page is treated independently.
Original attempts are preserved. Open the latest completed revision for the corrected OCR and component wiping.</p>
<p>Review both meaning and missed text: detection covers light speech bubbles and can miss captions, dark bubbles,
text over artwork, or small text. Text component wiping is heuristic, not artwork reconstruction.</p>
<table><tr><th>Batch</th><th>Images</th><th>Review versions</th></tr>'''+"".join(rows)+"</table>"
    (root/"index.html").write_text(document, encoding="utf-8")
    audit={"sizes": sizes, "unique_images": len(originals), "verified_exports": checked,
           "recorded_successful_response_tokens": tokens, "recorded_cost": cost,
           "usage_note": "Excludes 418-token smoke test and any rejected/invalid response without saved usage."}
    (root/"verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit))


if __name__=="__main__":
    build(sys.argv[1])
