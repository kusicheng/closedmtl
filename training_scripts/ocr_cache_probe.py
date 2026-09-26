"""Measure actual decoder cache storage for the configured four-beam OCR."""

import argparse
import json
from pathlib import Path

import torch
from transformers import VisionEncoderDecoderModel

from ocr_compression import cache_tensor_bytes


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    model=VisionEncoderDecoderModel.from_pretrained(args.model, local_files_only=True).eval()
    encoder=model.config.encoder
    patches=(encoder.image_size//encoder.patch_size)**2+1
    hidden=torch.zeros(4, patches, model.config.decoder.hidden_size)
    records=[]
    with torch.inference_mode():
        for length in (16, 64, 299):
            tokens=torch.full((4, length), model.config.decoder_start_token_id, dtype=torch.long)
            output=model.decoder(input_ids=tokens, encoder_hidden_states=hidden, use_cache=True)
            allocated=cache_tensor_bytes(output.past_key_values)
            records.append({"decoder_tokens":length, "cache_bytes_fp32":allocated,
                            "fp16_storage_lower_bound":allocated//2,
                            "int8_storage_lower_bound_without_scales":allocated//4})
    result={"model":args.model, "beams":4, "encoder_tokens":patches, "records":records,
            "notes":["Actual FP32 cache tensor storage, excluding Python and allocator overhead.",
                     "Lower-precision entries are storage estimates, not tested compressed-cache predictions.",
                     "Cache-off quality and RAM are measured separately by ocr_benchmark --no-cache."]}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
