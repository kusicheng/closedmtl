"""Local bubble-model comparison; outputs stay in the ignored outputs directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/bubble_comparison_20260922"
MODELS={
    "yolo11s": ("models/best/speech_bubble_yolo_s_gpu.pt", 768),
    "root_yolo11s": ("models/best/root_best.pt", 768),
    "yolo11n": ("models/best/speech_bubble_yolo_gpu.pt", 768),
    "yolo11m": ("models/best/speech_bubble_yolo_m_gpu.pt", 768),
    "early_yolo11n": ("models/best/speech_bubble_yolo.pt", 768),
    "mayocream768": ("models/segmentation_mayocreamVer/model.safetensors", 768),
    "mayocream1600": ("models/segmentation_mayocreamVer/model.safetensors", 1600),
}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def worker(args):
    os.environ["YOLO_CONFIG_DIR"]=str(ROOT/"models/ultralytics_config")
    os.environ["OMP_NUM_THREADS"]="8"
    os.environ["MKL_NUM_THREADS"]="8"
    import platform
    import cv2
    import numpy as np
    import psutil
    import torch
    import ultralytics
    from safetensors.torch import load_file
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor, DetectionValidator
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.metrics import DetMetrics, box_iou

    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    process=psutil.Process()

    def memory():
        info=process.memory_info()
        return {k:int(getattr(info, k)) for k in
                ("rss", "vms", "peak_wset", "private", "peak_pagefile") if hasattr(info, k)}

    baseline=memory()
    relative, size=MODELS[args.model]
    path=ROOT/relative
    segmented=path.suffix==".safetensors"
    started=time.perf_counter()
    if segmented:
        network=SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
        state=load_file(str(path))
        network.load_state_dict(state, strict=True)
        assert all(torch.equal(network.state_dict()[k], v) for k, v in state.items())
        state_count=len(state)
        del state
        network.names={0:"balloon"}
        model=YOLO("yolo11n-seg.yaml", task="segment")
        model.model=network.eval()
        model.overrides.update({"task":"segment", "imgsz":size})
    else:
        model=YOLO(str(path))
        state_count=None
    parameter_count=sum(p.numel() for p in model.model.parameters())
    architecture=model.model.yaml.get("yaml_file")
    load_ms=(time.perf_counter()-started)*1000
    loaded=memory()
    files=sorted((ROOT/"training_data/speech-bubbles-detection-yolo/images/test").glob("*.jpg"))
    assert len(files)==181, len(files)
    indices=random.Random(20260922).sample(range(len(files)), 24)
    images=[cv2.imread(str(files[i])) for i in indices]
    assert all(image is not None for image in images)
    predict_args=dict(imgsz=size, device=args.device, half=False, conf=0.35,
                      iou=0.5, agnostic_nms=True, max_det=300, rect=False,
                      verbose=False, save=False)

    def synchronize():
        if args.device!="cpu":
            torch.cuda.synchronize()

    # Warm inference includes actual masks for the segmentation model.
    for i in range(5):
        result=model.predict(images[i], **predict_args)
        synchronize()
        del result
    if args.device!="cpu":
        torch.cuda.reset_peak_memory_stats()
    warmed=memory()
    times=[]
    stage_times=[]
    mask_count=0
    for repeat in range(2):
        for image in images:
            synchronize()
            start=time.perf_counter()
            result=model.predict(image, **predict_args)
            synchronize()
            times.append((time.perf_counter()-start)*1000)
            stage_times.append(result[0].speed)
            if result[0].masks is not None:
                mask_count+=len(result[0].masks.data)
            del result
    profiled=memory()
    speed={"samples":len(times), "median_ms":float(np.median(times)),
           "mean_ms":float(np.mean(times)), "p95_ms":float(np.percentile(times, 95)),
           "fps_from_mean":float(1000/np.mean(times)), "times_ms":times,
           "mean_stages_ms":{key:float(np.mean([x[key] for x in stage_times]))
                            for key in stage_times[0]}, "masks_produced":mask_count}
    report={"model":args.model, "path":relative, "sha256":digest(path),
            "size_bytes":path.stat().st_size, "parameters_unfused":parameter_count,
            "architecture":architecture, "segmentation":segmented,
            "safetensors_verified_tensors":state_count, "device":args.device,
            "imgsz":size, "precision":"FP32", "batch":1, "rect":False,
            "threads":8, "speed":speed, "load_ms":load_ms,
            "memory_bytes":{"after_imports":baseline, "after_load":loaded,
                            "after_warmup":warmed, "after_timing":profiled},
            "environment":{"python":sys.version, "torch":torch.__version__,
                           "ultralytics":ultralytics.__version__, "platform":platform.platform(),
                           "cpu":platform.processor(), "physical_cores":psutil.cpu_count(False),
                           "logical_cores":psutil.cpu_count(),
                           "gpu":torch.cuda.get_device_name(0) if args.device!="cpu" else None},
            "timing_image_paths":[str(files[i].relative_to(ROOT)) for i in indices],
            "method":"Preloaded BGR pages; batch 1, square letterbox, five warmups, "
                     "24 fixed test images twice, synchronous predict including preprocessing, "
                     "NMS and masks (if supported). Excludes file I/O. Memory before quality evaluation."}
    if args.device!="cpu":
        report["cuda_memory_bytes"]={"peak_allocated":torch.cuda.max_memory_allocated(),
                                     "peak_reserved":torch.cuda.max_memory_reserved()}
    target=OUT/f"{args.model}_{args.device}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TIMING {args.model} {args.device} {speed['median_ms']:.2f} ms "
          f"peak RSS {profiled.get('peak_wset', 0)/1e6:.1f} MB", flush=True)
    if args.device=="cpu":
        return

    class SegmentBoxesPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            # Current YOLO11 Segment output is ((boxes+coefficients, prototypes), features).
            while isinstance(preds, (tuple, list)):
                preds=preds[0]
            return super().postprocess(preds[:, :5, :], img, orig_imgs, **kwargs)

    # Accuracy evaluates boxes for every model: local labels contain no masks.
    model.predictor=None
    metrics=DetMetrics(names={0:"balloon"})
    validator=DetectionValidator(args={"task":"detect", "plots":False}, save_dir=OUT)
    validator.iouv=torch.linspace(0.5, 0.95, 10)
    rows=[]
    corpus=hashlib.sha256()
    quality_args=dict(predict_args, conf=0.001)
    if segmented:
        quality_args["predictor"]=SegmentBoxesPredictor
    for index, image_path in enumerate(files):
        image=cv2.imread(str(image_path))
        height, width=image.shape[:2]
        label_path=ROOT/"training_data/speech-bubbles-detection-yolo/labels/test"/f"{image_path.stem}.txt"
        labels=np.loadtxt(label_path, ndmin=2).astype(np.float32)
        assert labels.shape[1]==5
        xywh=torch.tensor(labels[:, 1:])*torch.tensor([width, height, width, height])
        targets=torch.cat((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), 1)
        boxes=model.predict(image, **quality_args)[0].boxes
        prediction=boxes.xyxy.cpu()
        scores=boxes.conf.cpu()
        pred_classes=torch.zeros(len(prediction))
        target_classes=torch.zeros(len(targets))
        ious=box_iou(targets, prediction)
        correct=validator.match_predictions(pred_classes, target_classes, ious)
        metrics.update_stats(dict(tp=correct.numpy(), conf=scores.numpy(),
                                  pred_cls=pred_classes.numpy(), target_cls=target_classes.numpy(),
                                  target_img=np.zeros(1) if len(targets) else np.zeros(0),
                                  im_name=image_path.name))
        selected=scores>=0.35
        fixed=validator.match_predictions(pred_classes[selected], target_classes, ious[:, selected])
        tp=int(fixed[:, 0].sum())
        rows.append({"image":str(image_path.relative_to(ROOT)), "targets":len(targets),
                     "tp":tp, "fp":int(selected.sum())-tp, "fn":len(targets)-tp,
                     "predictions":[{"xyxy":b.tolist(), "confidence":float(c)}
                                    for b, c in zip(prediction[selected], scores[selected])]})
        corpus.update(image_path.name.encode())
        corpus.update(bytes.fromhex(digest(image_path)))
        corpus.update(bytes.fromhex(digest(label_path)))
        if (index+1)%60==0:
            print(f"QUALITY {args.model} {index+1}/{len(files)}", flush=True)
    metrics.process(plot=False)
    counts={key:sum(row[key] for row in rows) for key in ["tp", "fp", "fn"]}
    tp, fp, fn=(counts[key] for key in ["tp", "fp", "fn"])
    fixed={**counts, "precision":tp/(tp+fp), "recall":tp/(tp+fn), "f1":2*tp/(2*tp+fp+fn)}
    report["quality"]={"split":"test", "images":len(files), "targets":sum(x["targets"] for x in rows),
                       "corpus_sha256":corpus.hexdigest(), "class_mapping":"all six shapes to balloon",
                       "nms_iou":0.5, "ap_min_confidence":0.001, "fixed_confidence":0.35,
                       "fixed_iou":0.5, "fixed_threshold":fixed,
                       "ultralytics_metrics":{k:float(v) for k, v in metrics.results_dict.items()},
                       "per_image":rows}
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"DONE {args.model} F1={fixed['f1']:.6f} AP50={metrics.box.map50:.6f}", flush=True)


def run_all(args):
    OUT.mkdir(parents=True, exist_ok=True)
    for device in ("0", "cpu"):
        for name in MODELS:
            target=OUT/f"{name}_{device}.json"
            if target.exists():
                previous=json.loads(target.read_text(encoding="utf-8"))
                if device=="cpu" or "quality" in previous:
                    print(f"EXISTING {name} {device}", flush=True)
                    continue
            log=OUT/f"{name}_{device}.log"
            print(f"START {name} {device}", flush=True)
            with log.open("w", encoding="utf-8") as output:
                result=subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                       "--model", name, "--device", device],
                                      cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            print(log.read_text(encoding="utf-8")[-2500:], flush=True)
            if result.returncode:
                raise RuntimeError(f"{name} {device} failed; see {log}")
    summarize()


def summarize():
    import html
    import numpy as np

    reports={name:{device:json.loads((OUT/f"{name}_{device}.json").read_text())
                   for device in ("0", "cpu")} for name in MODELS}
    quality=[reports[name]["0"]["quality"] for name in MODELS]
    assert len({q["corpus_sha256"] for q in quality})==1
    assert all(q["images"]==181 and q["targets"]==1191 for q in quality)
    rng=np.random.default_rng(20260922)
    samples=rng.integers(0, 181, (5000, 181))
    bootstrap={}
    for name, devices in reports.items():
        rows=devices["0"]["quality"]["per_image"]
        counts=np.array([[r[k] for k in ("tp", "fp", "fn")] for r in rows])
        values=counts[samples].sum(axis=1)
        bootstrap[name]=2*values[:, 0]/(2*values[:, 0]+values[:, 1]+values[:, 2])
    intervals={name:np.quantile(values, [0.025, 0.975]).tolist()
               for name, values in bootstrap.items()}
    paired={name:{"f1_delta":reports["yolo11s"]["0"]["quality"]["fixed_threshold"]["f1"]-
                             reports[name]["0"]["quality"]["fixed_threshold"]["f1"],
                  "ci95":np.quantile(bootstrap["yolo11s"]-bootstrap[name], [0.025, 0.975]).tolist()}
            for name in ("root_yolo11s", "yolo11n", "mayocream768", "mayocream1600")}
    summary={"date":"2026-09-22", "cpu":"Intel Core Ultra 9 185H", "gpu":"RTX 4050 Laptop 6 GB",
             "bootstrap":{"unit":"image", "resamples":5000, "seed":20260922,
                          "f1_ci95":intervals, "yolo11s_minus_other":paired},
             "limitations":["Mask quality is unmeasured: the test set has only box labels.",
                            "This test is from the local training dataset distribution; Mayocream uses other datasets.",
                            "No exact file hashes overlap local splits; book separation, near duplicates and upstream overlap are unknown.",
                            "Image bootstrap does not account for correlation between pages of the same book.",
                            "Timing is a short local benchmark, excluding I/O and OCR; it includes masks for segmentation.",
                            "RAM is peak process resident working set before accuracy testing, not model-only RAM.",
                            "CUDA values are PyTorch allocated/reserved peaks, excluding driver/context allocations.",
                            "Raw box AP/F1 bypasses segmentation mask postprocessing; empty-mask filtering may alter deployed outputs."],
             "records":{name:{device:{k:v for k, v in report.items() if k!="quality"}|
                             ({"quality":{k:v for k, v in report["quality"].items() if k!="per_image"}}
                              if "quality" in report else {})
                              for device, report in devices.items()} for name, devices in reports.items()}}
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    quality_rows=[]
    runtime_rows=[]
    for name, devices in reports.items():
        gpu=devices["0"]
        cpu=devices["cpu"]
        q=gpu["quality"]
        fixed=q["fixed_threshold"]
        ap=q["ultralytics_metrics"]
        lo, hi=intervals[name]
        label=f"{name} ({gpu['imgsz']}px)"
        quality_rows.append(f"<tr><td>{label}</td><td>{100*fixed['precision']:.2f}</td>"
                            f"<td>{100*fixed['recall']:.2f}</td><td>{100*fixed['f1']:.2f}</td>"
                            f"<td>{100*lo:.2f}–{100*hi:.2f}</td>"
                            f"<td>{100*ap['metrics/mAP50(B)']:.2f}</td>"
                            f"<td>{100*ap['metrics/mAP50-95(B)']:.2f}</td>"
                            f"<td>{fixed['tp']}/{fixed['fp']}/{fixed['fn']}</td></tr>")
        runtime_rows.append(f"<tr><td>{label}</td><td>{gpu['parameters_unfused']/1e6:.3f}</td>"
                            f"<td>{gpu['size_bytes']/1e6:.2f}</td>"
                            f"<td>{gpu['speed']['median_ms']:.2f} / {gpu['speed']['p95_ms']:.2f}</td>"
                            f"<td>{cpu['speed']['median_ms']:.2f} / {cpu['speed']['p95_ms']:.2f}</td>"
                            f"<td>{cpu['memory_bytes']['after_timing']['peak_wset']/1e6:.1f}</td>"
                            f"<td>{gpu['memory_bytes']['after_timing']['peak_wset']/1e6:.1f}</td>"
                            f"<td>{gpu['cuda_memory_bytes']['peak_allocated']/1e6:.1f} / "
                            f"{gpu['cuda_memory_bytes']['peak_reserved']/1e6:.1f}</td></tr>")
    links="".join(f'<li><a href="{name}_0.json">{name}: quality/GPU</a>; '
                  f'<a href="{name}_cpu.json">CPU</a>; '
                  f'<a href="../../{MODELS[name][0]}">checkpoint</a></li>' for name in MODELS)
    limitations="".join(f"<li>{html.escape(x)}</li>" for x in summary["limitations"])
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Bubble model comparison — 2026-09-22</title>
<style>body{{font:16px system-ui;max-width:1280px;margin:36px auto;padding:0 24px;color:#172333}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:right;padding:10px;border-bottom:1px solid #dde3eb}}
td:first-child,th:first-child{{text-align:left}}th{{background:#eef4fa}}h1,h2{{color:#164571}}li{{margin:7px 0}}
.scroll{{overflow:auto}}code{{background:#eef4fa;padding:2px 4px}}</style>
<h1>Speech-bubble model comparison</h1><p>Measured locally on 2026-09-22. These models find bubbles; they do not recognize text.</p>
<p><strong>Recommendation:</strong> keep <code>models/best/speech_bubble_yolo_s_gpu.pt</code> for the current box-based pipeline.
It has the highest box mAP50–95 on this test and strong speed. The older root YOLO11s is statistically tied in F1.
Use the local GPU-trained YOLO11n when file size and CPU latency matter most.
Mayocream supplies masks, but mask quality needs a labeled segmentation test before replacement.</p>
<h2>Quality on the same local test set</h2><p>181 pages, 1,191 bubbles, six shape classes collapsed into one balloon class.
All percentages below are newly measured. P/R/F1: confidence 0.35, matching IoU 0.50; class-agnostic NMS IoU 0.50.
AP uses confidence ≥0.001 and standard Ultralytics thresholds. F1 confidence intervals: 5,000 paired image-bootstrap samples.</p>
<div class="scroll"><table><tr><th>Model</th><th>P %</th><th>R %</th><th>F1 %</th><th>F1 95% CI</th>
<th>AP50 %</th><th>AP50–95 %</th><th>TP/FP/FN</th></tr>{''.join(quality_rows)}</table></div>
<h2>Size, latency and memory</h2><p>Intel Core Ultra 9 185H; NVIDIA RTX 4050 Laptop, 6 GB; Windows 11;
PyTorch 2.11.0+cu128, Ultralytics 8.4.61. FP32, batch 1, eight CPU threads, square letterbox,
five warmups then 24 fixed pages twice. Timed prediction includes preprocessing, NMS and masks when supported;
file decoding and OCR are excluded. Each model/device runs in a fresh process. MB means 1,000,000 bytes.</p>
<div class="scroll"><table><tr><th>Model</th><th>Params M</th><th>File MB</th><th>GPU median / p95 ms</th>
<th>CPU median / p95 ms</th><th>CPU peak RAM MB</th><th>GPU-process RAM MB</th>
<th>CUDA allocated / reserved MB</th></tr>{''.join(runtime_rows)}</table></div>
<p>The original local .pt files store half-precision parameters; the Mayocream SafeTensors file stores FP32.
All inference here uses FP32. File size therefore does not measure compute or runtime RAM.
The JSON reports also contain process private-commit peaks, current RAM, mean latency and all 48 timings.</p>
<h2>Published and historical scores</h2><p>The selected local YOLO11s previously recorded 87.62% shape-aware validation F1 and
92.44% mAP50. Those scores classify bubble shape; the new test scores above only locate bubbles.
<a href="https://huggingface.co/mayocream/manga109-segmentation-bubble">Mayocream's model card</a> identifies this as
a lossless SafeTensors conversion of YOLO11n segmentation. Its <a href="https://huggingface.co/huyvux3005/manga109-segmentation-bubble">upstream card</a>
reports box P 97.55%, R 97.03%, AP50 99.10%, AP50–95 96.67%, and mask AP50 99.13%, AP50–95 94.69% on its own evaluation.
These published values are not a shared-dataset comparison.</p>
<h2>Limits</h2><ul>{limitations}</ul><h2>Evidence and reproduction</h2><ul>{links}</ul>
<p><a href="summary.json">Summary and paired confidence intervals</a> · <a href="split_integrity.json">Local split hash check</a> ·
<a href="../../tests/benchmark_bubble_models.py">Benchmark script</a></p>
<p>Run <code>.venv_gpu\\Scripts\\python.exe tests/benchmark_bubble_models.py</code> from the project root.
Existing completed records are reused; archive the output folder first to measure again.
No checkpoint or application code was changed.</p></html>"""
    (OUT/"report.html").write_text(page, encoding="utf-8")
    print("SUMMARY "+json.dumps({"f1_ci95":intervals, "paired":paired}), flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="0")
    arguments=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if arguments.model:
        worker(arguments)
    else:
        run_all(arguments)
