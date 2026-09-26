"""Distill teacher token distributions on human-labeled Manga109-s crops."""

import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

from ocr_data import crop, digest, read_rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="training_data/manga109_ocr/train.jsonl")
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--resume", default=None)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    if args.temperature<=0 or args.steps<1 or args.batch_size<1:
        raise ValueError("Temperature, steps, and batch size must be positive")
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory to preserve checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng=random.Random(args.seed)
    teacher=VisionEncoderDecoderModel.from_pretrained(args.teacher, local_files_only=True).cuda().eval()
    teacher.requires_grad_(False)
    from ocr_compression import build_student
    if args.resume:
        student=VisionEncoderDecoderModel.from_pretrained(args.resume, local_files_only=True)
    else:
        student=build_student(teacher, args.encoder_layers, args.decoder_layers)
    student.cuda().train().requires_grad_(True)
    student.config.use_cache=False
    student.gradient_checkpointing_enable()
    tokenizer=AutoTokenizer.from_pretrained(args.teacher, tokenizer_type="bert-japanese", local_files_only=True)
    processor=ViTImageProcessor.from_pretrained(args.teacher, local_files_only=True)
    rows=read_rows(args.manifest)
    # Never truncate a ground-truth label silently.
    usable=[]
    for row in rows:
        tokens=tokenizer(row["text"], add_special_tokens=True).input_ids
        if len(tokens)<=300:
            usable.append((row, tokens))
    if len(usable)<args.batch_size:
        raise ValueError("Insufficient labeled training crops")
    optimizer=torch.optim.AdamW(student.parameters(), lr=args.lr)
    scaler=torch.amp.GradScaler("cuda")
    start=time.perf_counter()
    with (output/"training.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.steps):
            batch=rng.sample(usable, args.batch_size)
            pixels=processor([crop(row) for row, _ in batch], return_tensors="pt").pixel_values.cuda()
            length=max(len(tokens) for _, tokens in batch)
            labels=torch.full((len(batch), length), -100, device="cuda", dtype=torch.long)
            for i, (_, tokens) in enumerate(batch):
                labels[i, :len(tokens)]=torch.tensor(tokens, device="cuda")
            decoder_ids=teacher.prepare_decoder_input_ids_from_labels(labels=labels)
            decoder_mask=decoder_ids.ne(tokenizer.pad_token_id).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad():
                    target=teacher(pixel_values=pixels, decoder_input_ids=decoder_ids,
                                   decoder_attention_mask=decoder_mask, use_cache=False).logits
                prediction=student(pixel_values=pixels, decoder_input_ids=decoder_ids,
                                   decoder_attention_mask=decoder_mask, use_cache=False).logits
                mask=labels!=-100
                hard=F.cross_entropy(prediction.float().reshape(-1, prediction.shape[-1]), labels.reshape(-1))
                temperature=args.temperature
                soft=F.kl_div(F.log_softmax(prediction[mask].float()/temperature, dim=-1),
                              F.softmax(target[mask].float()/temperature, dim=-1),
                              reduction="batchmean")*temperature**2
                loss=0.5*hard+0.5*soft
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step+1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if (step+1)%10==0 or step==0 or step+1==args.steps:
                record={"step":step+1, "loss":loss.item(), "hard_ce":hard.item(),
                        "teacher_kl":soft.item(), "seconds":time.perf_counter()-start}
                log.write(json.dumps(record)+"\n")
                log.flush()
                print(json.dumps(record), flush=True)
    student.gradient_checkpointing_disable()
    student.config.use_cache=True
    student.eval().save_pretrained(output)
    processor.save_pretrained(output)
    tokenizer.save_pretrained(output)
    report={**vars(args), "source_dataset":"Manga109-s released 2026-05-21",
            "manifest_sha256":digest(args.manifest), "eligible_crops":len(usable),
            "too_long_skipped":len(rows)-len(usable), "parameters":sum(p.numel() for p in student.parameters()),
            "cuda_peak_bytes":torch.cuda.max_memory_allocated(),
            "note":"Training checkpoint only; quality and memory require independent benchmark.",
            "resume_semantics":"Weights only; optimizer and random stream restart."}
    (output/"distillation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__=="__main__":
    main()
