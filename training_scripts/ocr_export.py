"""Export quantized weights offline, then benchmark loading in a new process."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

from ocr_compression import export_int8


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new export directory")
    torch.set_num_threads(4)
    torch.backends.quantized.engine="onednn"
    model=VisionEncoderDecoderModel.from_pretrained(args.model, local_files_only=True)
    export_int8(model, output, inplace=True)
    AutoTokenizer.from_pretrained(args.model, tokenizer_type="bert-japanese", local_files_only=True).save_pretrained(output)
    ViTImageProcessor.from_pretrained(args.model, local_files_only=True).save_pretrained(output)
    print(output)
