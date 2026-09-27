# closedmtl
context + TL + redraw
Dataset and model are to be closed, while the training code is open source.

## Project layout

- `models/best/`: five selected detector checkpoints and `manifest.json`, which records source paths, SHA256 hashes, validation metrics, and file relocations.
- `models/best/speech_bubble_yolo_s_gpu.pt`: recommended detector by recorded validation F1 (0.8762 at image size 768).
- `models/best/root_best.pt`: former root `best.pt`, separately evaluated at six-class validation F1 0.8537 and image size 640. These historical scores use a different metric from the merged-class bubble benchmark below.
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

## Bubble segmentation training and verification

The recorded F1 **0.977907** is real and was reproduced with fresh inference:
1,173 true positives, 35 false positives and 18 misses on 181 test pages.
It merges six bubble shapes into one class and uses confidence 0.35 and box
IoU 0.50. The historical 0.876185 score uses class-averaged validation precision
and recall. Their difference is not evidence of training improvement.
Evidence: `outputs/bubble_training_20260922/contamination_audit/verification.json`.

The audit confirmed repeated artwork between training page `-1044-` and
validation page `-1047-`; only the dialogue differs. Exact hashes missed it.
The new preparation excludes that validation page and the overlapping
`scantrad_merged` data. The old test set remains a development benchmark;
book independence is not established. Mayocream's
[source model](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
already trained on Manga109/MangaSegmentation, so the new local split cannot
remove its prior exposure.

Downloaded images are organized with source-to-path mappings and SHA256 hashes:

- `training_data/reference_only/openmantra/images/<book>/<book>_page_####.jpg`:
  214 unchanged images, excluded by request because their text/panel rectangles
  are not bubble masks. Old repository metadata is in
  `ARCHIVE/dataset_sources/open-mantra-dataset/`.
- `training_data/evaluation/ai4va/images/<split>/<issue>/`: the original
  62-page external check, with descriptive publication/date/page filenames.
- `training_data/ai4va_additional/images/<issue>/`: 142 additional organized
  pages. `prepared/` records issue splits, exact masks, training polygons and
  annotation exclusions. Two incomplete/crowd pages are preserved for review.

Mayocream's first mask-training run reached exact local box/mask F1
0.970420/0.972392. Its 62-page AI4VA box/mask scores were 0.777358/0.780045,
and its original-data box score was 0.855876. These do not meet the overall
goal. The first YOLO11s mask run reached local box/mask F1 0.949571/0.952478,
but only 0.350220/0.348401 on the 62-page AI4VA check. These runs are retained
as unsuccessful candidates. The later mixed box/mask adaptation is complete.
The mixed trainer uses real masks and separate box-only batches; it never
turns rectangle labels into segmentation masks.

Mayocream's first mixed run scored 0.898876 for both outputs on the
27-page exact-mask calibration set. A later saved third-epoch checkpoint
reached box/mask F1 0.922078/0.926407 with 1024-pixel GPU inference. This is
calibration evidence, not final-test acceptance. The earlier resolution study
used an interrupted second-epoch small checkpoint and remains preserved in
`resolution_study/`; the table below checks the finished models.
Its new continuation, `small_joint_1024_run01`, completed two further epochs
at 1024 with threefold sampling weight for the 39 AI4VA training pages
containing bubbles. Exact box/mask F1 is 0.935412/0.935412 on AI4VA calibration
and 0.955933/0.959095 on the full Manga validation set; current-data box F1
is 0.967979. Mayocream's corresponding `mayocream_joint_1024_run01` completed
three epochs: AI4VA calibration box/mask F1 is 0.931915/0.931915, full Manga
validation is 0.965197/0.968258, and current-data box F1 is 0.970980.
Both trained on 84 AI4VA pages with 724 real bubble masks, mixed with Manga
mask replay and the existing box-only training data. These are calibration
results, separate from the final test below.
Both targeted queues and the additional rectangle-sampling trial completed.
The trial used threefold sampling weight for existing rectangle-containing
box pages, with published labels unchanged. Small-model current box F1 fell
to 0.960586. Mayocream improved wide-caption recall to 4/10, but its AI4VA
box/mask F1 fell to 0.920086/0.924406. Neither trial replaced the earlier
candidate. The original queue's Windows status-file replacement error and
successful continuation remain recorded. `pre_test_candidate_selection.json`
records the decision before final inference; all evidence is under
`outputs/bubble_training_20260927/`.

The once-evaluated final set contains 28 pages and 273 bubble masks from
publication issues excluded from training and calibration. Both selected
models pass F1 0.90 for boxes and masks at the frozen 1024 operating size:

| Model | Box F1 | Mask F1 | Box precision | Box recall |
| --- | --- | --- | --- | --- |
| Original Mayocream | 0.826797 | 0.833333 | 0.746313 | 0.926740 |
| Trained YOLO11s | 0.955556 | 0.959259 | 0.966292 | 0.945055 |
| Trained Mayocream YOLO11n | 0.955595 | 0.959147 | 0.927586 | 0.985348 |

The trained models satisfy the declared one-percentage-point comparison
margin. Mayocream finds more bubbles and produces more false positives; it
has 2,842,803 parameters versus 10,082,675 for the small model, about 72% fewer.
It is the practical choice when recall and model size matter most. The small
model offers higher precision. This is a comparison of trained checkpoints,
not proof that architecture alone caused the difference.
Use `models/bubble_segmentation/mayocream_joint_1024_run01/segment.pt` or
`models/bubble_segmentation/small_joint_1024_run01/segment.pt` with the paired
export command below. Training is stopped. Reports, prediction journals and
verified comparison are `final_test_{baseline,small,mayocream}.json` and
`final_test_comparison.json` in the same output directory.

F1 counts bubble instances: `2*TP/(2*TP+FP+FN)`. Predictions above confidence
0.35 undergo class-agnostic NMS at IoU 0.50, then one-to-one matching at
IoU 0.50. Boxes and masks are matched separately. Mask IoU counts foreground
intersection over foreground union, not correct background pixels. Evaluation
uniformly resizes each page while preserving its aspect ratio, centers it on
a square gray canvas, and maps outputs back to original image coordinates
before scoring. Empty padding cannot improve this formula. Regression tests
confirm padding invariance and reject masks with padded-canvas dimensions.

The installed trainer's native mask validation used rectangular padding while
deployed inference used square padding. New mixed training explicitly uses
square validation to match deployment. Padding can change predictions even
though background pixels do not enter F1. Training already applies uniform
scale augmentation of approximately 0.8 to 1.2; independent horizontal and
vertical stretching has not been shown to improve this task.

The finished checkpoints were checked at five input sizes on the same 27
calibration pages/229 targets, with fixed confidence and FP32 GPU inference.
Each cell is box F1 / mask F1 at IoU 0.50. This is inference-size robustness,
not separately trained models. The additional check did not change the frozen
1024 setting, checkpoint selection or final-test results. Evidence:
`outputs/bubble_training_20260927/selected_resolution_check_summary.json`.

| Input square | Trained YOLO11s | Trained Mayocream |
| --- | --- | --- |
| 640 | 0.733 / 0.733 | 0.769 / 0.764 |
| 768 | 0.889 / 0.894 | 0.905 / 0.905 |
| 1024 | 0.935 / 0.935 | 0.932 / 0.932 |
| 1280 | 0.928 / 0.928 | 0.930 / 0.934 |
| 1600 | 0.868 / 0.873 | 0.925 / 0.929 |

In current-data validation, the selected small model finds 157/170 rectangles
and 0/10 very wide targets (width/height>3); Mayocream finds 153/170 and 1/10.
Visual inspection identifies all ten wide targets as boxed character
names/identification captions. Both find 16/25 targets with a short side below
32 pixels after uniform scaling to a 768-pixel long side. These groups overlap
and must not be added together. Existing training has 557 rectangles on 195
pages, but only 17 wide targets on 14 pages. Category diagnostics report recall
and support: single-class predictions do not assign false positives to the
original shape categories. A direct
diagnostic found zero detections lost through empty masks: most missed
caption boxes are localized only below the fixed confidence cutoff. The
original recommended detector finds 9/10 captions at 1024 and 7/10 at 768.
Thus the small-model adaptation pipeline lost caption retention; this audit
does not isolate head conversion from subsequent training as the cause.
The intended caption scope is still awaiting clarification; the completed
trial kept existing positive labels without assuming an answer.

Final-test size/aspect diagnostics also retain support counts. Small and
Mayocream find 23/25 and 25/25 bubbles in the 16-to-24-pixel reference group,
and 61/68 and 65/68 in the 24-to-32 group. Both miss the single target below
16 pixels; that sample is too small to establish tiny-bubble performance.
They find 78/87 and 85/87 very wide AI4VA bubbles. These differ from the boxed
character captions in the current-data audit. All final AI4VA source pages
have a long side above 2048 pixels, so this test does not establish quality
on naturally low-resolution scans. In the current-data validation groups,
both find 78/79 targets on pages up to 1024 pixels and 132/134 on pages above
2048 pixels, using the common 1024 inference setting. These are recall
diagnostics, not per-group F1 or a controlled image-quality experiment.

On the final AI4VA test, stricter IoU 0.75 yields box/mask F1 of
0.874074/0.911111 for small and 0.888099/0.952043 for Mayocream. The declared
0.90 target uses IoU 0.50; tight box localization is a remaining limitation.
The calibration set's smallest reference-size group contains only three
targets, including a verified annotation error: image 252 annotation 42376
is a tail of bubble 42467, labeled as a separate instance. Published labels
and scores remain unchanged pending annotation-policy clarification. Evidence:
`outputs/bubble_training_20260927/resolution_study/summary.json` and
`outputs/bubble_training_20260927/category_audit/`.

The additional AI4VA split has 27 calibration pages with 229 masks and
28 final-test pages with 273 masks, in separate publication issues.
The original 62 pages are a development reference. The initial 768-pixel
protocol remains preserved as `joint_evaluation_protocol.json`; the executed
1024 protocol is `joint_evaluation_protocol_v2.json`, frozen with checkpoint
hashes and selection evidence before test inference. The final set was
evaluated once per model and was not used for training or calibration.
Published `Comic Bubble` labels remain unchanged; some narration labels are
inconsistent. Publication issues do not establish independent story series.
A full-page identity audit compared all 28 reserved pages against 7,421 unique
local training images across both corpora, including earlier training phases.
No exact-file, decoded-image or screened perceptual-hash candidates were
found in 207,788 comparisons. This does not exclude reused panels, altered
artwork or unknown upstream pretraining exposure.
The two models share the YOLO11 family and differ in capacity and pretraining;
their comparison cannot isolate architecture alone.

Export paired predictions from a completed one-class segmentation checkpoint:

```powershell
.venv_gpu/Scripts/python.exe training_scripts/predict_bubble_segments.py --model <checkpoint.pt> --image <page.png> --output <fresh-directory> --imgsz 1024
```

Outputs are `boxes.json`, individual `masks/bubble_####.png`, a full-resolution
`segmentation.png` layer and `preview.jpg`. These are model predictions.

For a calibrated checkpoint, a sibling `deployment.json` can bind its SHA256
to the operating resolution: `{"model_sha256":"<checkpoint SHA256>","imgsz":1024}`.
The exporter verifies this binding and uses that size when `--imgsz` is omitted.
Older checkpoints without the config retain the exporter's 768 default.
Direct Ultralytics calls still require an explicit `imgsz=1024`; this companion
config is consumed by the project exporter. A stale config is rejected.

Do not use the ask skill; request data or calibration directly only after verifying the specific need.
