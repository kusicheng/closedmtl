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

## Planned items
Optimization of models with faster scripts and possibly renting a server to train it on.
To implement an actual GUI using PyQt possibly (Disclaimer: this project will not be monetized, ever, by me)
