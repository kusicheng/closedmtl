"""Standalone CPU OCR with TorchScript decoder key/value cache."""

import argparse
import hashlib
import json
from pathlib import Path
import time

PROCESS_STARTED=time.perf_counter()

from PIL import Image
import psutil
import torch

if __package__:
    from .ocr_artifacts import fingerprint, file_hash
    from .ocr_journal import PredictionJournal
    from .ocr_script_runtime import (beam_search, decode, load_vocabulary, post_process,
                                      preprocess, validate_configuration)
    from .ocr_data import crop, read_rows
    from .ocr_benchmark import edit_distance
else:
    from ocr_artifacts import fingerprint, file_hash
    from ocr_journal import PredictionJournal
    from ocr_script_runtime import (beam_search, decode, load_vocabulary, post_process,
                                     preprocess, validate_configuration)
    from ocr_data import crop, read_rows
    from ocr_benchmark import edit_distance


class CachedDecoder:
    """Adapt scripted prefill/step outputs to the shared beam search."""

    def __init__(self, prefill, step):
        self.prefill=prefill
        self.step=step
        self.cache=None

    def __call__(self, ids, hidden):
        if self.cache is None:
            logits, self.cache=self.prefill(ids, hidden)
        else:
            logits, self.cache=self.step(ids[:, -1:], hidden, self.cache)
        return logits

    def reorder(self, parents):
        if self.cache is None:
            raise RuntimeError("Cannot reorder before prefill")
        self.cache=tuple(tensor.index_select(0, parents) for tensor in self.cache)


class CachedScriptOcr:
    """Recognize PIL images or paths using local cached INT8 CPU artifacts."""

    def __init__(self, directory, *, threads=4):
        self.directory=Path(directory)
        if threads is not None:
            if threads<1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(threads)
        self.manifest=validate_configuration(self.directory, use_cache=True)
        if self.manifest.get("cache_dtype") not in ("fp32", "fp16"):
            raise ValueError("Unsupported cache precision")
        if self.manifest.get("cache_order")!=["self_key", "self_value", "cross_key", "cross_value"]:
            raise ValueError("Unsupported cache tensor order")
        self.encoder=torch.jit.load(str(self.directory/"encoder.pt"), map_location="cpu").eval()
        self.prefill=torch.jit.load(str(self.directory/"decoder_prefill.pt"), map_location="cpu").eval()
        self.step=torch.jit.load(str(self.directory/"decoder_step.pt"), map_location="cpu").eval()
        self.vocab, self.special=load_vocabulary(self.directory)

    @torch.inference_mode()
    def predict_with_score(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                pixels=preprocess(source)
        elif isinstance(image, Image.Image):
            pixels=preprocess(image)
        else:
            raise ValueError("Expected a PIL image or image path")
        hidden=self.encoder(pixels)
        decoder=CachedDecoder(self.prefill, self.step)
        tokens, score=beam_search(decoder, hidden, cache_reorder=decoder.reorder)
        return decode(tokens, self.vocab, self.special), score

    def __call__(self, image):
        return self.predict_with_score(image)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="training_data/manga109_ocr/validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args=parser.parse_args()
    artifacts=fingerprint(args.model)
    source_hash=file_hash(__file__)
    journal=PredictionJournal(args.output, {**vars(args), **artifacts,
                              "benchmark_source_sha256":source_hash,
                              "manifest_sha256":file_hash(args.manifest)})
    ocr=CachedScriptOcr(args.model, threads=args.threads)
    rows=read_rows(args.manifest)
    if args.limit>0:
        rows=rows[:args.limit]
    if not rows:
        raise ValueError("Empty evaluation manifest")
    process=psutil.Process()
    loaded_ram=process.memory_info().rss
    inference_start=time.perf_counter()
    predictions=[]
    edits=characters=exact=0
    for index, row in enumerate(rows):
        prediction, score=ocr.predict_with_score(crop(row))
        target=post_process(row["text"])
        distance=edit_distance(target, prediction)
        edits+=distance
        characters+=len(target)
        exact+=prediction==target
        predictions.append({"id":row["id"], "prediction":prediction, "target":target,
                            "edits":distance, "sequence_score":score})
        journal.append(predictions[-1])
        if (index+1)%16==0:
            print(json.dumps({"done":index+1, "cer":edits/max(1, characters)}), flush=True)
    journal.close()
    info=process.memory_info()
    if not hasattr(info, "peak_wset"):
        raise RuntimeError("This benchmark requires a real Windows lifetime peak working-set counter")
    result={"model":args.model, "variant":"torchscript-cached", "device":"cpu",
            **artifacts, "benchmark_source_sha256":source_hash,
            "use_cache":True, "cache_dtype":ocr.manifest["cache_dtype"],
            "samples":len(rows),
            "manifest_sha256":hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "sample_ids_sha256":hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
            "cer":edits/max(1, characters), "exact_match":exact/len(rows),
            "ram_loaded_bytes":loaded_ram, "ram_final_bytes":info.rss,
            "ram_peak_bytes":info.peak_wset, "ram_peak_method":"windows_peak_working_set",
            "private_bytes":getattr(info, "private", None), "cuda_peak_bytes":0,
            "inference_seconds":time.perf_counter()-inference_start,
            "total_seconds":time.perf_counter()-PROCESS_STARTED, "torch":torch.__version__,
            "predictions":predictions}
    output=Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key:value for key, value in result.items() if key!="predictions"}), flush=True)


if __name__=="__main__":
    main()
