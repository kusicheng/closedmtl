"""Measure labeled OCR quality and complete process RAM in a fresh process."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import psutil


def edit_distance(first, second):
    row=list(range(len(second)+1))
    for i, a in enumerate(first, 1):
        current=[i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1]+1, row[j]+1, row[j-1]+(a!=b)))
        row=current
    return row[-1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="training_data/manga109_ocr/validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--variant", choices=["fp32", "int8", "int8-saved", "fp16", "lowrank"], default="fp32")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args=parser.parse_args()
    started=time.perf_counter()
    import torch
    from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
    from ocr_text import post_process
    from ocr_data import crop, read_rows
    from ocr_artifacts import fingerprint, file_hash
    from ocr_journal import PredictionJournal

    torch.set_num_threads(args.threads)
    artifacts=fingerprint(args.model)
    source_hash=file_hash(__file__)
    journal=PredictionJournal(args.output, {**vars(args), **artifacts,
                              "benchmark_source_sha256":source_hash,
                              "manifest_sha256":file_hash(args.manifest)})
    if args.variant in ("int8", "int8-saved") and args.device!="cpu":
        raise ValueError("Dynamic INT8 requires CPU")
    dtype=torch.float16 if args.variant=="fp16" else torch.float32
    if args.variant=="int8-saved":
        from ocr_compression import load_int8
        model=load_int8(args.model)
    else:
        model=VisionEncoderDecoderModel.from_pretrained(args.model, local_files_only=True, dtype=dtype)
    tokenizer=AutoTokenizer.from_pretrained(args.model, tokenizer_type="bert-japanese", local_files_only=True)
    processor=ViTImageProcessor.from_pretrained(args.model, local_files_only=True)
    if args.variant=="int8":
        torch.backends.quantized.engine="onednn"
        torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    elif args.variant=="fp16":
        model.half()
    elif args.variant=="lowrank":
        from ocr_compression import factorize_model
        model=factorize_model(model, rank_fraction=0.25, inplace=True)
    model.to(args.device).eval()
    rows=read_rows(args.manifest)
    if args.limit>0:
        rows=rows[:args.limit]
    if not rows:
        raise ValueError("Empty evaluation manifest")
    predictions=[]
    total_edits=0
    total_chars=0
    exact=0
    process=psutil.Process()
    loaded_ram=process.memory_info().rss
    if args.device=="cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_start=time.perf_counter()
    for index, row in enumerate(rows):
        pixels=processor(crop(row), return_tensors="pt").pixel_values.to(args.device, dtype=model.dtype)
        with torch.inference_mode():
            tokens=model.generate(pixels, max_length=300, use_cache=not args.no_cache)
        prediction=post_process(tokenizer.decode(tokens[0], skip_special_tokens=True))
        target=post_process(row["text"])
        distance=edit_distance(target, prediction)
        total_edits+=distance
        total_chars+=len(target)
        exact+=prediction==target
        predictions.append({"id":row["id"], "prediction":prediction, "target":target, "edits":distance})
        journal.append(predictions[-1])
        if (index+1)%16==0:
            print(json.dumps({"done":index+1, "cer":total_edits/max(1, total_chars)}), flush=True)
    journal.close()
    info=process.memory_info()
    if not hasattr(info, "peak_wset"):
        raise RuntimeError("This benchmark requires a real Windows lifetime peak working-set counter")
    result={"model":args.model, "variant":args.variant, "device":args.device,
            **artifacts, "benchmark_source_sha256":source_hash,
            "use_cache":not args.no_cache, "samples":len(rows),
            "manifest_sha256":hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "sample_ids_sha256":hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
            "cer":total_edits/max(1, total_chars), "exact_match":exact/len(rows),
            "ram_loaded_bytes":loaded_ram, "ram_final_bytes":info.rss,
            "ram_peak_bytes":info.peak_wset,
            "ram_peak_method":"windows_peak_working_set",
            "private_bytes":getattr(info, "private", None),
            "cuda_peak_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda" else 0,
            "inference_seconds":time.perf_counter()-inference_start,
            "total_seconds":time.perf_counter()-started, "torch":torch.__version__,
            "predictions":predictions}
    output=Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key:value for key, value in result.items() if key!="predictions"}), flush=True)


if __name__=="__main__":
    main()
