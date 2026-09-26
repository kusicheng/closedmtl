"""Export a local INT8 ViT/BERT to a Transformers-free CPU runtime."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch import nn

try:
    from .ocr_compression import load_int8
except ImportError:
    from ocr_compression import load_int8


class EncoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder=model.encoder
        self.projection=getattr(model, "enc_to_dec_proj", nn.Identity())

    def forward(self, pixel_values):
        return self.projection(self.encoder(pixel_values, return_dict=False)[0])


class DecoderGraph(nn.Module):
    """Use explicit causal attention at every length, including the first token."""

    def __init__(self, model):
        super().__init__()
        self.embeddings=model.decoder.bert.embeddings
        self.layers=model.decoder.bert.encoder
        self.head=model.decoder.cls
        self.layer_count=model.config.decoder.num_hidden_layers

    def forward(self, input_ids, encoder_hidden_states):
        hidden=self.embeddings(input_ids=input_ids,
                               token_type_ids=torch.zeros_like(input_ids))
        length=input_ids.shape[1]
        positions=torch.arange(length, device=input_ids.device)
        forbidden=positions.unsqueeze(0)>positions.unsqueeze(1)
        mask=forbidden.to(hidden.dtype)*torch.finfo(hidden.dtype).min
        mask=mask.unsqueeze(0).unsqueeze(0)
        result=self.layers(hidden, attention_mask=mask,
                           head_mask=[None]*self.layer_count,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=None, use_cache=False,
                           return_dict=False)[0]
        return self.head(result)


def trace_and_verify(model):
    """Check unseen lengths and beam batch sizes against the original decoder."""
    model.cpu().eval()
    encoder=EncoderGraph(model).eval()
    decoder=DecoderGraph(model).eval()
    size=model.config.encoder.image_size
    height, width=(size, size) if isinstance(size, int) else size
    generator=torch.Generator().manual_seed(20260914)
    pixels=torch.randn(1, model.config.encoder.num_channels, height, width,
                       generator=generator)
    ids=torch.randint(1, model.config.decoder.vocab_size, (1, 4), generator=generator)
    with torch.inference_mode():
        hidden=encoder(pixels)
        encoded=torch.jit.trace(encoder, (pixels,), check_trace=False)
        decoded=torch.jit.trace(decoder, (ids, hidden), check_trace=False)
        encoded=torch.jit.freeze(encoded.eval())
        decoded=torch.jit.freeze(decoded.eval())
        checks=[]
        for batch in (1, 4):
            inputs=torch.randn(batch, model.config.encoder.num_channels, height, width,
                               generator=generator)
            expected_hidden=encoder(inputs)
            actual_hidden=encoded(inputs)
            torch.testing.assert_close(actual_hidden, expected_hidden, atol=1e-5, rtol=1e-5)
            for length in (1, 4, 9):
                tokens=torch.randint(1, model.config.decoder.vocab_size, (batch, length),
                                     generator=generator)
                expected=model.decoder(input_ids=tokens, encoder_hidden_states=expected_hidden,
                                       use_cache=False, return_dict=False)[0]
                actual=decoded(tokens, actual_hidden)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                checks.append({"batch": batch, "length": length,
                               "max_abs_logit_error": (actual-expected).abs().max().item(),
                               "argmax_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))})
    return encoded, decoded, checks


def export_script(source, output):
    source=Path(source)
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new export directory")
    model=load_int8(source)
    encoder, decoder, checks=trace_and_verify(model)
    output.mkdir(parents=True, exist_ok=True)
    torch.jit.save(encoder, str(output/"encoder.pt"))
    torch.jit.save(decoder, str(output/"decoder.pt"))
    for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                 "special_tokens_map.json", "tokenizer_config.json", "vocab.txt",
                 "tokenizer.json", "added_tokens.json"):
        if (source/name).is_file():
            shutil.copyfile(source/name, output/name)
    manifest={"format_version": 1, "source": str(source.resolve()),
              "engine": torch.backends.quantized.engine, "torch_version": torch.__version__,
              "encoder": "encoder.pt", "decoder": "decoder.pt", "use_cache": False,
              "decoder_output": "full_sequence_logits", "verification": checks}
    (output/"script_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    print(json.dumps(export_script(args.model, args.output), indent=2))
