# closedmtl
context + TL + redraw
Dataset and model are to be closed, while the training code is open source.

## Project layout

- `models/best/`: five selected detector checkpoints and `manifest.json`, which records source paths, SHA256 hashes, validation metrics, and file relocations.
- `models/best/speech_bubble_yolo_s_gpu.pt`: recommended detector by recorded validation F1 (0.8762 at image size 768).
- `models/best/root_best.pt`: former root `best.pt`, separately evaluated at F1 0.8537 and image size 640. These results do not establish F1 above 0.9.
- `models/pretrained/`: original downloaded YOLO weights, now grouped together.
- `models/`: existing baseline detector and metrics.
- `runs/`: original training checkpoints, settings, results, and validation examples, preserved for reference.
- `archive/notes/` and `archive/data/`: original root notes and dataset archive, preserved byte for byte.
- `training_data/`, `training_scripts/`: datasets and training code.
- `api_networking_components/translation_API.py`: OpenRouter text translation client.
- `tests/`: offline translation contract tests.

Historical run settings retain their original paths. When replaying a run, use
`models/pretrained/<filename>` for pretrained weights that were previously at the root.
Run checkpoints stay at their original paths. The existing validation helper now uses
`models/best/root_best.pt`. Model files remain excluded from git.

## Translate text

The detector finds speech bubbles; it does not translate text. Translation uses the
OpenRouter provider selected by the existing project stub. The client follows the
[OpenRouter HTTP API](https://openrouter.ai/docs/quickstart).

From this directory in PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:OPENROUTER_API_KEY="your-openrouter-key"
python main.py --text "Bonjour tout le monde" --target en
```

Alternatively, import the function:

```python
from api_networking_components.translation_API import translation_request

translated=translation_request("Bonjour", target_language="Japanese")
```

Source language defaults to `detect language`; `auto` and explicit names or codes
are also accepted through `source_language` (CLI: `--source`). Detection is performed
by the provider model. The return value is the translated string, not a detected-language
label. The prompt requests preserved meaning and line breaks; model output can vary.

Set `OPENROUTER_MODEL` or pass `model` (CLI: `--model`) to select a provider model.
The default is `minimax/minimax-m3:free`; availability depends on OpenRouter.
The function also accepts `key` directly. Keep real keys out of source files.
`.env.example` lists the settings; `.env` files are not automatically loaded.

Invalid inputs raise `ValueError`. Timeouts, HTTP failures, and malformed or incomplete
responses raise `RuntimeError` with a sanitized message. Requests time out after 60
seconds by default and are not automatically retried. This is a Python API client and
CLI; it does not start an HTTP server. Provider calls use your account credits.

Run offline tests without a key or provider charges:

```powershell
python tests/test_translation_api.py
```

## Replace translated text and export images

`api_networking_components/text_replacement.py` accepts identified regions and translations.
It supports manually supplied regions as shown below, or OCR regions from the image
pipeline described in the next section.

Create a UTF-8 JSON file such as `translated_regions.json`:

```json
{
  "images": [
    {
      "path": "pages/page01.png",
      "regions": [
        {
          "label": "panel_1_bubble_1",
          "box": [30, 40, 220, 150],
          "text": "Hello, world!",
          "background": "white",
          "color": "black"
        }
      ]
    },
    {"path": "pages/page02.jpg", "regions": []}
  ]
}
```

Paths are relative to the JSON file. Boxes are integer pixel coordinates in the stored
image: `[left, top, right, bottom]`, with exclusive right and bottom edges. The script
does not apply EXIF rotation. Optional labels and other JSON metadata are retained.
An empty regions list includes an unchanged image; empty translated text clears its box.

```powershell
python -m api_networking_components.text_replacement translated_regions.json outputs/translated.zip --font C:/Windows/Fonts/arial.ttf
```

For Python callers, use `replace_and_zip(images, output_zip, font_path=...)` from that
module. Python API image paths are relative to the working directory. Use `replace_text`
to render one image without packaging it.

The renderer uses a solid background fill (white by default), centers the translation,
and wraps/shrinks it to fit at 8–64 pixels. It rejects text that cannot fit instead of
silently truncating it. Supply a font that includes the target language's glyphs; Arial
is an example for Latin text, not a universal font. Complex-script shaping depends on
the font and Pillow build. This simple renderer does not reconstruct artwork behind text.
Use boxes tightly around text on plain speech-bubble backgrounds. Overlapping boxes are
applied in list order. Animated and multipage images are rejected.

The default batch size is 50 (configurable from 1 to 50). Images are rendered one at a
time into one temporary folder. All batches go into **one ZIP**, with sequential PNG
names to prevent filename collisions and a `manifest.json` mapping outputs to inputs.
Sources stay untouched; outputs use lossless RGBA PNG, regardless of input format.

The script verifies the ZIP's file list, CRCs, and SHA256 hashes before publishing it.
It publishes with a same-filesystem hard link, which requires a filesystem such as NTFS
that supports hard links. It then deletes only its own temporary folder. The ZIP remains
on disk. Existing output files are never overwritten. On failure, the error identifies
the retained recovery folder; an incomplete archive is not published at the output path.
If cleanup fails after publication, both the verified ZIP and recovery folder remain.

```powershell
python tests/test_text_replacement.py
```

## Japanese manga OCR pipeline

Use the **global Python installation**, not the training virtual environment.
On 2026-09-05 it contained manga-ocr 0.1.16, Torch 2.12.1+cu132 with working CUDA,
Ultralytics 8.4.61, Pillow 12.2.0, and Transformers 4.57.6. No packages were installed
or changed for this integration. `requirements-ocr.txt` records application dependencies.
OCR weights are cached inside `models/huggingface/`.

The pipeline uses the trained YOLO11s bubble detector, refines light bubble interiors,
reads each accepted crop with [manga-ocr](https://github.com/kha-white/manga-ocr), then
sends OCR text grouped by image to `minimax/minimax-m3:free`. The prompt specifies the
translation task and strict textbox JSON. Each original ID must occur exactly once;
missing, duplicate, or invented IDs are rejected before replacement. Unrelated pages
are explicitly treated independently. Token usage and the returned model are recorded.

```powershell
$env:OPENROUTER_API_KEY="your-openrouter-key"
python main.py images input.zip --output outputs/chapter_01 --target en --font C:/Windows/Fonts/arial.ttf
```

This is a ZIP-processing CLI; no web upload server is started. It preserves the input
ZIP. It saves OCR JSON, translation responses, translated region JSON, a summary, and
`translated.zip`. Extracted input folders and rendering staging folders are removed
only after successful export. Failure folders are retained for recovery.

Regions keep `box` (the original ink bounds), `wipe_rects` (individual text components),
and `layout_box` (the containing interior area available for English layout). When
`wipe_rects` is present, only those small rectangles are filled, preserving gaps and
surrounding art. Full-bubble crops go to OCR so edge columns are not cut off by the
wiping heuristic. Text may use the wider `layout_box`. Background RGB arrays from OCR are accepted. Output PNGs
and region labels stay aligned through IDs. Export still batches at most 50 images.

Manga-ocr is a Japanese recognizer, not a language detector or text locator. The
automatic source-language option applies to translation. Bubble detection and reading
order are heuristics; this version has no panel segmentation and can miss text outside
bubbles, text over artwork, dark bubbles, or small text. Blank/light-region checks reduce
OCR hallucinations but cannot guarantee accuracy. Original images and OCR are retained
for review. Flat fills do not reconstruct artwork.

## Live manual-review tests

```powershell
python tests/run_ocr_review_batches.py --batches 5
```

The runner prompts invisibly for an API key if the environment variable is unset.
It never saves the key. It selects distinct random sizes from 1–20 and unique Japanese
source pages from `training_data/scantrad_merged`. Each batch uses different chapter
groups, although chapters may belong to the same series. Selection seed, original
paths, and source hashes are recorded. The dataset's supplied README identifies its
license as CC BY 4.0 and gives the Roboflow source URL.

Each batch has `sources/`, `sources.json`, `result/` audit files and ZIP, `translated/`
review copies, and `review.html` showing source/output pairs and OCR/translation text.
Review copies intentionally remain for manual inspection. Failed attempts remain as
recovery folders and failure reports. A current `result/summary.json` indicates success.

Use `--ocr-only` to prepare local evidence without a translation call. Use
`--resume test_output/<run>` to continue the same selection and reuse saved translations.
Resume assumes saved OCR and target language are unchanged. The batch pipeline retries
HTTP 429 at most twice, 30 seconds apart, and invalid textbox JSON at most twice,
one second apart. The API functions themselves do not retry.
Requests group approximately 20 textboxes at a time, retaining whole-page groups.
API payloads contain text, stable IDs, image IDs and positions; local source paths and
image bytes are not sent. `--resume <run> --revision revision_03` preserves a separate
correction pass inside the same five batches. No source image or earlier result is deleted.

The review run created on 2026-09-05 is under `test_output/run_20260905_124851/`.
Its sizes are 12, 14, 13, 7, and 16 images (62 unique pages). Open its `index.html`
for review links; the latest complete revisions include OCR and wiping corrections.
`verification.json` records export checks and token usage from saved successful responses.
The final version is `revision_04`. It reuses the corrected translations from
`revision_03` and caps font size relative to source glyphs. All five final exports pass
integrity checks. `manual_review.json` records the five inspected outputs, remaining
cleanup issues, and the visual inspection limit. The final typography-only pass was
verified by tests and export checks, without further image inspection.

The run recorded 59,907 tokens including its smoke test, with reported cost 0.
Rejected/invalid responses without usage records may add tokens. No API key was saved.
There are 51 passing offline checks across the five test scripts below.

Offline checks (no provider calls):

```powershell
python tests/test_translation_api.py
python tests/test_structured_translation.py
python tests/test_text_replacement.py
python tests/test_ocr_regions.py
python tests/test_image_pipeline.py
```
