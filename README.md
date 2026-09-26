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

## Manga OCR distillation

The teacher is downloaded at
`models/huggingface/hub/models--mayocream--manga-ocr/snapshots/4380edba990b959c508752350955350c1c80c31c`.
All 265 checkpoint tensors match the cached `kha-white/manga-ocr-base` tensors.
The [Mayocream model card](https://huggingface.co/mayocream/manga-ocr)
describes this Apache-2.0 checkpoint as a SafeTensors package of the base model.

The local Manga109-s release is under
`models/huggingface/distill-Manga109-s/Manga109s_released_2026_05_21`.
Its supplied terms permit local experiments, prohibit dataset redistribution,
and require Manga109-s attribution in published results and trained models.
Images and annotations remain local. No new dataset download is needed.

Prepare manifests with the current annotation version, a fixed seed, whole-book
splits, human text labels, and 10-pixel crop margins:

```powershell
python training_scripts/ocr_data.py --root models/huggingface/distill-Manga109-s/Manga109s_released_2026_05_21
```

`training_data/manga109_ocr/provenance.json` records annotation hashes, split
membership, manifest hashes, and 14 missing trailing pages without text labels.
There are 97,176 training, 13,127 validation, and 12,909 test crops.
These are student-heldout books. The original teacher's unseeded crop split is
not recoverable from [upstream code](https://github.com/kha-white/manga-ocr/blob/master/manga_ocr_dev/data/process_manga109s.py),
so teacher training overlap is unknown. Do not equate these scores with the
original published benchmark.

Use the existing global `python` interpreter. Package versions and the teacher
revision are recorded in `runs/ocr/setup.json`. No global package changes are
required. The available RTX 4050 has 6 GB VRAM.

```powershell
$teacher='models/huggingface/hub/models--mayocream--manga-ocr/snapshots/4380edba990b959c508752350955350c1c80c31c'
python training_scripts/ocr_benchmark.py --model $teacher --output runs/ocr/teacher_validation.json --limit 0
python training_scripts/ocr_benchmark.py --model $teacher --variant int8 --output runs/ocr/int8_validation.json --limit 0
python training_scripts/ocr_distill.py --teacher $teacher --output models/ocr/student_8e2d_run01 --steps 1000
python training_scripts/ocr_export.py --model models/ocr/student_8e2d_run01 --output models/ocr/student_8e2d_run01_int8
python training_scripts/ocr_benchmark.py --model models/ocr/student_8e2d_run01_int8 --variant int8-saved --output runs/ocr/student_int8_validation.json --limit 0
```

Each benchmark must run in a fresh process. It reports character error rate,
exact match, process peak working set, private memory, and CUDA allocation.
Use the same manifest, sample count, and generation settings for comparisons.
`--limit 0` evaluates the full manifest. `--no-cache` measures recomputation;
`--variant lowrank` measures SVD factorization. Small smoke runs establish only
that an option runs. They do not establish accuracy or memory acceptance.

To deploy without importing Transformers, export and run the TorchScript model:

```powershell
python training_scripts/ocr_script_export.py --model models/ocr/student_8e2d_run01_int8 --output models/ocr/student_8e2d_run01_script
python training_scripts/ocr_script_runtime.py --model models/ocr/student_8e2d_run01_script --output runs/ocr/student_script_validation.json --limit 0
```

The exporter checks dynamic sequence lengths and beam batch sizes against the
original decoder. The CPU runtime preserves preprocessing and four-beam search,
and recomputes decoder states without a cache. It validates the exported
configuration before loading. Use a fresh output directory for each export.
The images command accepts `--ocr-model <export-directory>` to select this
backend. Omitting that option retains the current base model. The callable
`ScriptOcr` backend accepts a PIL image or an image path.
The first 64-crop runtime check used 895,877,120 bytes peak working set, but its
accuracy did not pass the proposed gate. This is not a completed model result.

`ocr_cache_probe.py` measured actual four-beam FP32 cache storage from 10.5 MB
at 16 decoder tokens to 24.4 MB at 299 tokens. Lower-precision cache sizes in
its report are estimates. They do not establish compressed-cache accuracy.

Distillation combines ground-truth cross entropy and teacher token-distribution
KL loss at temperature 2. Checkpoints need separate quality and RAM evaluation.
`--resume` loads weights into a new run; it does not restore optimizer state.
INT8 export runs separately from deployment. `int8-saved` loads the saved
quantized tensors without first allocating a dense FP32 model. Use
`training_scripts/ocr_compare.py` with full teacher/student test reports to
check the gate. It checks each ordered ID and normalized human label against
the manifest, recalculates edit counts and aggregate quality, and rejects
partial, inconsistent, or mismatched reports. Peak RAM uses the Windows
lifetime peak working-set counter. Private committed memory is also reported;
the resident-RAM target is not a claim that the program fits on a 1 GB machine.
A GPU teacher may provide the full accuracy reference when the deployed student
passes the absolute CPU RAM limit. The one-third memory comparison requires
both reports to use CPU. Teacher CPU and GPU predictions matched on the initial
64 validation crops; each final report still uses all identical test crops.
The initial proposed quality gate is no more than 1 percentage point additional
CER and no more than 2 points lost exact match on the full identical test set.
The RAM gate is below 1,000,000,000 bytes or at most one third of the matching
teacher baseline. Report private memory as well as resident working set.

Alternative models reviewed: `mayocream/manga-ocr-onnx` exports the same teacher;
`mayocream/mit48px-ocr` needs different line preprocessing and has GPL-3.0 terms;
`bluolightning/manga-ocr-mobile` uses TFLite and a different architecture. Their
reported scores do not establish parity on this local benchmark.

### Verified distilled model

`models/ocr/student_8e2d_run03_cached` contains the verified CPU model:
8 encoder blocks, 2 decoder blocks, INT8 linear weights, and FP32 decoder cache.
Both models completed all 12,909 identical test crops on September 15, 2026.

| Metric | Mayocream teacher | Distilled student |
|---|---:|---:|
| Character error rate | 7.4282% | 8.3552% |
| Exact-match accuracy | 69.1146% | 67.4413% |

The CER increase is 0.9270 percentage points and exact-match loss is 1.6733
points, within the fixed limits of 1 and 2 points respectively. This is a
measured accuracy tradeoff. Training loss and F1 are not reported by this
generation benchmark.

The student's full lifetime CPU peak resident RAM is **915,546,112 bytes**,
below the strict 1,000,000,000-byte target. A retained Windows process handle
verified the peak after successful exit, including startup and report writing.
Peak private committed memory is 2,190,557,184 bytes; this does not establish
that the program fits on a computer with only 1 GB of RAM.
The original CPU teacher's 64-crop peak was 1,266,176,000 bytes. Relative to
that smaller baseline run, the student uses about 27.7% less resident RAM.
Acceptance uses the absolute sub-1GB target, not the one-third alternative.
The full teacher accuracy reference ran on CUDA; its process RAM is not used
for a CPU memory reduction ratio.

Evidence is in `runs/ocr/teacher_full_retry.json`, `student_full_retry.json`,
`student_full_retry.memory.json`, and `run03_full_acceptance.json`.
The gate recomputed every prediction's score against the manifest. Incremental
prediction journals match the final reports, and model hashes were verified.
The model includes `provenance/training_and_results.json` with teacher revision,
Manga109-s attribution, training lineage, result hashes, and measured limits.

Load the model directly from the project root:

```python
from training_scripts.ocr_cached_runtime import CachedScriptOcr

ocr=CachedScriptOcr("models/ocr/student_8e2d_run03_cached")
text=ocr("path/to/text_crop.png")
```

The images command selects this same backend with
`--ocr-model models/ocr/student_8e2d_run03_cached`.
The standalone OCR memory measurement excludes the image pipeline's detector
and translation components.

To reproduce training after preparing the manifests, use fresh output
directories (existing checkpoints are preserved):

```powershell
$teacher='models/huggingface/hub/models--mayocream--manga-ocr/snapshots/4380edba990b959c508752350955350c1c80c31c'
python training_scripts/ocr_distill.py --teacher $teacher --output models/ocr/reproduce01 --steps 1000 --batch-size 4 --seed 20260914 --lr 0.00002
python training_scripts/ocr_distill.py --teacher $teacher --resume models/ocr/reproduce01 --output models/ocr/reproduce02 --steps 3000 --batch-size 4 --seed 20260915 --lr 0.00001
python training_scripts/ocr_distill.py --teacher $teacher --resume models/ocr/reproduce02 --output models/ocr/reproduce03 --steps 1500 --batch-size 16 --seed 20260916 --lr 0.00001
python training_scripts/ocr_export.py --model models/ocr/reproduce03 --output models/ocr/reproduce03_int8
python training_scripts/ocr_cached_export.py --model models/ocr/reproduce03_int8 --output models/ocr/reproduce03_cached --cache-dtype fp32
```

Training samples labeled crops with replacement and combines label cross
entropy with teacher KL loss. Reproduction does not imply bitwise identical
GPU training. Each reproduced checkpoint needs its own accuracy evaluation.

Rank-0.25 SVD reduced quality in the initial screen. FP16 cache storage was
tested on run02: it halved cache tensor bytes but saved only about 3.3 MB of
process peak RAM and changed two of 64 predictions. The selected candidate
therefore retains FP32 cache. Saved INT8 export and the standalone runtime
provide the measured memory reduction.

Recheck the completed full evaluation:

```powershell
python training_scripts/ocr_compare.py --teacher runs/ocr/teacher_full_retry.json --student runs/ocr/student_full_retry.json --output runs/ocr/rechecked_acceptance.json
```

New reference and cached-runtime benchmarks also write per-prediction JSONL
journals and startup metadata. Use a fresh output name; existing journals are
protected from replacement. `ocr_run_watch.py --prefix <new-prefix> -- <script>
<arguments>` launches a benchmark with a log, process identity, and independent
post-exit lifetime-memory report. Partial journals are recovery evidence and
do not pass the full-test gate.

## Bubble segmentation training and F1 audit

The selected starting detector is `models/best/speech_bubble_yolo_s_gpu.pt`.
Its historical **six-class validation** F1 is 0.876185. The September 22
comparison's **single-class test** F1 is 0.977907: 1,173 true positives, 35
false positives, and 18 misses on 181 pages. It merges all bubble shapes into
one class, uses confidence 0.35 and box IoU 0.50, and aggregates instance counts.
The historical score instead takes the harmonic mean of class-averaged
precision and recall. These scores do not demonstrate a training improvement.
The saved result is in `outputs/bubble_comparison_20260922/yolo11s_0.json`.

The split audit confirmed repeated artwork across training page `-1044-`
and validation page `-1047-`, with different text. Exact image hashes missed
this duplication. The comparison is a development benchmark; it does not
establish performance on independent books. The limited perceptual screen
found no train/test duplicate, which is insufficient to prove independence.
See `outputs/bubble_training_20260922/contamination_audit/candidate_00.jpg`
and `split_audit.json`. Reproduce with `tests/audit_bubble_splits.py` and
`tests/verify_bubble_f1.py` using `.venv_gpu/Scripts/python.exe`.

The new Parquet files contain real MangaSegmentation balloon masks. The
audit found 7,869 usable positive pages with 99,023 masks in the available
Manga109-s images. Exclude mismatched `PrayerHaNemurenai` pages. Keep whole
books in separate fine-tuning splits. Do not treat unannotated pages as
verified negatives. Do not mix in `scantrad_merged/train`, which overlaps
the old validation and test data.

Training preparation converts the selected YOLO11s detector into a one-class
segmenter and retains Mayocream's YOLO11n segmentation weights. Box-only
adaptation and actual mask training use separate supervision; rectangles
are not mask labels. Stop the segmentation training stage when validation
box and instance-mask F1 both reach 0.90 at confidence 0.35 and IoU 0.50.
These are training controls, not independent test acceptance.

Mayocream's [source model](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
already trained on MangaSegmentation/Manga109. A new local book split cannot
remove that prior exposure. Independent human-labeled evaluation pages with
book/chapter IDs outside the previous corpora are needed for final acceptance.
Existing training masks are sufficient; no new training checkpoint has yet
passed this goal. Fine-tuning compares the two initialized models; it does
not isolate architecture from model size or pretraining.

Downloaded images are organized by purpose and source identity:

```text
training_data/
  bubble_joint_20260922/          # Prepared training splits and exact-mask evaluation manifests
  evaluation/ai4va/
    images/<split>/<issue>/      # Publication, issue date and page in each filename
    annotations/                # Unmodified source COCO annotations
    evaluation/                 # Normalized masks, boxes and provenance
    mapping_manifest.json       # Original names, organized paths and SHA256 hashes
  reference_only/openmantra/
    images/<book>/              # <book>_page_####.jpg
    annotations.json
    manifest.json               # Original-to-new mapping; excluded from this experiment
```

OpenMantra's 214 images were renamed without changing their bytes. Its old
repository metadata is preserved in `ARCHIVE/dataset_sources/open-mantra-dataset/`.
It contains text/panel rectangles, not bubble masks, and is excluded by request.

[AI4VA](https://github.com/IVRL/AI4VA) supplies external evaluation data:
62 available annotated pages, 738 bubble boxes and 737 masks. All pages,
including 29 supplied negative pages, remain in the evaluation. One bubble
has no source polygon; its box remains scored, and mask results include a
conservative missing-mask score. This is not a rigorous bound on a fully
annotated greedy match. One unavailable, completely unannotated page is excluded.
Class `Comic Bubble` includes narration captions. Publication issues are
grouping identifiers; they do not establish independent books or series.
None of these pages is used for training or confidence calibration.

The fixed external protocol is saved in
`outputs/bubble_training_20260922/external_evaluation_protocol.json`.
Original Mayocream scored box F1 0.620155 and known-mask F1 0.629310 on these
pages at 768 pixels. This limited external subset does not establish broad
generalization. Retrained comparisons remain pending.

Both models completed one box-training epoch on the mixed current/MangaSeg
data. Frozen checkpoints and verified mask-preserving transfers are in
`models/bubble_segmentation/box_epoch1_handoff/`. Segmentation run03 starts
from those weights with a fresh optimizer and trains on real masks. Logs,
checkpoint hashes and process identities are in `outputs/bubble_training_20260922/`.

To export predictions from a completed one-class segmentation checkpoint:

```powershell
.venv_gpu/Scripts/python.exe training_scripts/predict_bubble_segments.py --model <checkpoint.pt> --image <page.png> --output <fresh-directory>
```

The output contains `boxes.json`, individual `masks/bubble_####.png`, a merged
`segmentation.png` layer and `preview.jpg`. These are model predictions.

Do not use the ask skill; request data or calibration directly only after verifying the specific need.
