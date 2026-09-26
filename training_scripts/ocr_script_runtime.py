"""Small TorchScript OCR runtime, independent of Transformers and manga_ocr.

Generation matches the teacher's deterministic four-beam, uncached search.
The exporter must provide encoder.pt, decoder.pt and the original tokenizer files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

PROCESS_STARTED=time.perf_counter()

import jaconv
import numpy as np
from PIL import Image
import psutil
import torch

if __package__:
    from .ocr_artifacts import fingerprint, file_hash
    from .ocr_benchmark import edit_distance
    from .ocr_data import crop, read_rows
else:
    from ocr_artifacts import fingerprint, file_hash
    from ocr_benchmark import edit_distance
    from ocr_data import crop, read_rows


def validate_configuration(directory, device="cpu", *, use_cache=False):
    directory=Path(directory)
    expected={
        "generation_config.json":{"decoder_start_token_id":2, "eos_token_id":3,
                                  "pad_token_id":0, "num_beams":4, "length_penalty":2.0,
                                  "early_stopping":True, "no_repeat_ngram_size":3,
                                  "max_length":300},
        "preprocessor_config.json":{"do_resize":True, "do_rescale":True, "do_normalize":True,
                                    "size":{"height":224, "width":224}, "resample":2,
                                    "rescale_factor":1/255, "image_mean":[0.5]*3,
                                    "image_std":[0.5]*3, "image_processor_type":"ViTImageProcessor"},
        "tokenizer_config.json":{"tokenizer_class":"BertJapaneseTokenizer",
                                "subword_tokenizer_type":"character",
                                "clean_up_tokenization_spaces":False}}
    for filename, values in expected.items():
        actual=json.loads(directory.joinpath(filename).read_text(encoding="utf-8"))
        for key, value in values.items():
            if actual.get(key)!=value:
                raise ValueError(f"Unsupported {filename} {key}: {actual.get(key)!r}")
        if filename=="generation_config.json":
            for key, default in {"do_sample":False, "repetition_penalty":1.0,
                                 "min_length":0, "min_new_tokens":None, "max_new_tokens":None,
                                 "forced_bos_token_id":None, "forced_eos_token_id":None,
                                 "bad_words_ids":None, "suppress_tokens":None,
                                 "begin_suppress_tokens":None, "renormalize_logits":False,
                                 "encoder_no_repeat_ngram_size":0, "num_beam_groups":1}.items():
                if actual.get(key, default)!=default:
                    raise ValueError(f"Unsupported generation setting {key}")
    manifest=json.loads(directory.joinpath("script_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version")!=1 or manifest.get("use_cache") is not use_cache:
        raise ValueError("Unsupported TorchScript artifact format")
    engine=manifest.get("engine")
    if engine:
        if device!="cpu":
            raise ValueError("Dynamic INT8 TorchScript requires CPU")
        if engine not in torch.backends.quantized.supported_engines:
            raise ValueError(f"Quantization engine unavailable: {engine}")
        torch.backends.quantized.engine=engine
    return manifest


def post_process(text):
    text="".join(text.split()).replace("…", "...")
    text=re.sub("[・.]{2,}", lambda match:(match.end()-match.start())*".", text)
    return jaconv.h2z(text, ascii=True, digit=True)


def preprocess(image):
    image=image.convert("L").convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    pixels=np.asarray(image).astype(np.float64)*(1/255)
    pixels=pixels.astype(np.float32)
    pixels=(pixels-np.float32(0.5))/np.float32(0.5)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)


def load_vocabulary(directory):
    directory=Path(directory)
    vocab=directory.joinpath("vocab.txt").read_text(encoding="utf-8").splitlines()
    special={"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"}
    path=directory/"special_tokens_map.json"
    if path.exists():
        for value in json.loads(path.read_text(encoding="utf-8")).values():
            for token in value if isinstance(value, list) else [value]:
                special.add(token["content"] if isinstance(token, dict) else token)
    return vocab, special


def decode(tokens, vocab, special):
    pieces=[vocab[int(token)] for token in tokens if vocab[int(token)] not in special]
    return post_process(" ".join(pieces).replace(" ##", "").strip())


def ban_repeated_ngrams(sequences, scores, size):
    if size<=0 or sequences.shape[1]+1<size:
        return
    for index, sequence in enumerate(sequences.tolist()):
        prefix=tuple(sequence[-(size-1):]) if size>1 else ()
        banned=[sequence[start+size-1] for start in range(len(sequence)-size+1)
                if tuple(sequence[start:start+size-1])==prefix]
        scores[index, banned]=-float("inf")


@torch.inference_mode()
def beam_search(decoder, hidden, max_length=300, num_beams=4, length_penalty=2.0,
                no_repeat_ngram_size=3, start_id=2, eos_id=3, pad_id=0, cache_reorder=None):
    """Return the best sequence and its length-normalized log probability.

    Single-image search follows the installed HF tensor beam algorithm, including
    its top-2B candidates, top-B finishing eligibility and max-length scoring.
    """
    if hidden.shape[0]!=1 or num_beams<2 or max_length<2:
        raise ValueError("Expected one image, at least two beams and max_length>=2")
    device=hidden.device
    hidden=hidden.expand(num_beams, -1, -1).contiguous()
    running=torch.full((num_beams, max_length), pad_id or eos_id, dtype=torch.long, device=device)
    running[:, 0]=start_id
    finished=running.clone()
    running_scores=torch.zeros(num_beams, device=device)
    running_scores[1:]=-1e9
    finished_scores=torch.full((num_beams,), -1e9, device=device)
    finished_flags=torch.zeros(num_beams, dtype=torch.bool, device=device)
    finished_lengths=torch.ones(num_beams, dtype=torch.long, device=device)
    eligible=torch.arange(2*num_beams, device=device)<num_beams
    for position in range(1, max_length):
        logits=decoder(running[:, :position], hidden)
        logits=logits[:, -1, :] if logits.ndim==3 else logits
        scores=torch.log_softmax(logits.float(), dim=-1)
        del logits
        ban_repeated_ngrams(running[:, :position], scores, no_repeat_ngram_size)
        vocab_size=scores.shape[-1]
        top_scores, top_indices=torch.topk((scores+running_scores[:, None]).flatten(), 2*num_beams)
        candidates=running[top_indices//vocab_size].clone()
        tokens=top_indices%vocab_size
        candidates[:, position]=tokens
        stopped=(tokens==eos_id)|(position+1==max_length)
        live_scores=top_scores+stopped.float()*-1e9
        next_indices=torch.topk(live_scores, num_beams).indices
        if cache_reorder is not None:
            cache_reorder((top_indices//vocab_size)[next_indices])
        running=candidates[next_indices]
        running_scores=live_scores[next_indices]
        just_finished=stopped&eligible
        normalized=top_scores/(position**length_penalty)+(~just_finished)*-1e9
        merged_scores=torch.cat((finished_scores, normalized))
        selected=torch.topk(merged_scores, num_beams).indices
        finished=torch.cat((finished, candidates))[selected]
        finished_scores=merged_scores[selected]
        finished_flags=torch.cat((finished_flags, just_finished))[selected]
        finished_lengths=torch.cat((finished_lengths,
                                     torch.full((2*num_beams,), position+1, device=device)))[selected]
        worst=torch.where(finished_flags, finished_scores.min(), -1e9)
        improvement=torch.any(running_scores[0]/(position**length_penalty)>worst)
        if finished_flags.all() or stopped.all() or not improvement:
            break
    return finished[0, :int(finished_lengths[0])], float(finished_scores[0])


class ScriptOcr:
    """Callable local OCR backend for PIL images or image paths.

    INT8 exports run on CPU. The optional thread limit also applies to other
    PyTorch operations in the current process.
    """

    def __init__(self, directory, *, device="cpu", threads=4):
        self.directory=Path(directory)
        self.device=device
        if threads is not None:
            if threads<1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(threads)
        self.manifest=validate_configuration(self.directory, device)
        self.encoder=torch.jit.load(str(self.directory/"encoder.pt"), map_location=device).eval()
        self.decoder=torch.jit.load(str(self.directory/"decoder.pt"), map_location=device).eval()
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
        hidden=self.encoder(pixels.to(self.device))
        tokens, score=beam_search(self.decoder, hidden)
        return decode(tokens, self.vocab, self.special), score

    def __call__(self, image):
        return self.predict_with_score(image)[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="training_data/manga109_ocr/validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args=parser.parse_args()
    started=PROCESS_STARTED
    artifacts=fingerprint(args.model)
    source_hash=file_hash(__file__)
    ocr=ScriptOcr(args.model, device=args.device, threads=args.threads)
    rows=read_rows(args.manifest)
    if args.limit>0:
        rows=rows[:args.limit]
    if not rows:
        raise ValueError("Empty evaluation manifest")
    process=psutil.Process()
    loaded_ram=process.memory_info().rss
    if args.device=="cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_start=time.perf_counter()
    predictions=[]
    edits=characters=exact=0
    with torch.inference_mode():
        for index, row in enumerate(rows):
            prediction, score=ocr.predict_with_score(crop(row))
            target=post_process(row["text"])
            distance=edit_distance(target, prediction)
            edits+=distance
            characters+=len(target)
            exact+=prediction==target
            predictions.append({"id":row["id"], "prediction":prediction, "target":target,
                                "edits":distance, "sequence_score":score})
            if (index+1)%16==0:
                print(json.dumps({"done":index+1, "cer":edits/max(1, characters)}), flush=True)
    info=process.memory_info()
    if not hasattr(info, "peak_wset"):
        raise RuntimeError("This benchmark requires a real Windows lifetime peak working-set counter")
    result={"model":args.model, "variant":"torchscript", "device":args.device,
            **artifacts, "benchmark_source_sha256":source_hash,
            "use_cache":False, "samples":len(rows),
            "manifest_sha256":hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "sample_ids_sha256":hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
            "cer":edits/max(1, characters), "exact_match":exact/len(rows),
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
