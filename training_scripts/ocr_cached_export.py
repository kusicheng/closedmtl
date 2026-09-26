"""Export BERT self/cross attention caches as explicit TorchScript tensors."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .ocr_compression import load_int8
    from .ocr_script_export import EncoderGraph
except ImportError:
    from ocr_compression import load_int8
    from ocr_script_export import EncoderGraph


class CachedDecoder(nn.Module):
    """BERT absolute-position SDPA with flat self-K/V, cross-K/V per layer."""

    def __init__(self, model, prefill=False, cache_dtype="fp32"):
        super().__init__()
        if model.config.decoder.position_embedding_type!="absolute":
            raise ValueError("Only absolute-position BERT is supported")
        if cache_dtype not in ("fp32", "fp16"):
            raise ValueError("cache_dtype must be fp32 or fp16")
        self.embeddings=model.decoder.bert.embeddings
        self.layers=model.decoder.bert.encoder.layer
        self.head=model.decoder.cls
        self.prefill=prefill
        self.cache_dtype=torch.float16 if cache_dtype=="fp16" else torch.float32

    def split(self, tensor, attention):
        return tensor.view(tensor.shape[0], -1, attention.num_attention_heads,
                           attention.attention_head_size).transpose(1, 2)

    def attend(self, query, key, value, mask, attention):
        result=F.scaled_dot_product_attention(query, key, value,
                                              attn_mask=mask, dropout_p=0.0,
                                              is_causal=False)
        return result.transpose(1, 2).reshape(query.shape[0], query.shape[2],
                                              attention.all_head_size)

    def forward(self, input_ids, encoder_hidden_states, cache=()):
        past=0 if self.prefill else cache[0].shape[2]
        positions=torch.arange(input_ids.shape[1], device=input_ids.device)+past
        hidden=self.embeddings(input_ids=input_ids, position_ids=positions.unsqueeze(0),
                               token_type_ids=torch.zeros_like(input_ids))
        keys=torch.arange(past+input_ids.shape[1], device=input_ids.device)
        forbidden=keys.unsqueeze(0)>positions.unsqueeze(1)
        mask=(forbidden.to(hidden.dtype)*torch.finfo(hidden.dtype).min).unsqueeze(0).unsqueeze(0)
        updated=[]
        for index, layer in enumerate(self.layers):
            attention=layer.attention.self
            query=self.split(attention.query(hidden), attention)
            key=self.split(attention.key(hidden), attention)
            value=self.split(attention.value(hidden), attention)
            if not self.prefill:
                key=torch.cat((cache[4*index].to(hidden.dtype), key), dim=2)
                value=torch.cat((cache[4*index+1].to(hidden.dtype), value), dim=2)
            attended=self.attend(query, key, value, mask, attention)
            hidden=layer.attention.output(attended, hidden)
            cross=layer.crossattention.self
            cross_query=self.split(cross.query(hidden), cross)
            if self.prefill:
                cross_key=self.split(cross.key(encoder_hidden_states), cross)
                cross_value=self.split(cross.value(encoder_hidden_states), cross)
            else:
                cross_key=cache[4*index+2].to(hidden.dtype)
                cross_value=cache[4*index+3].to(hidden.dtype)
            attended=self.attend(cross_query, cross_key, cross_value, None, cross)
            hidden=layer.crossattention.output(attended, hidden)
            hidden=layer.output(layer.intermediate(hidden), hidden)
            updated.extend(tensor.to(self.cache_dtype) for tensor in
                           (key, value, cross_key, cross_value))
        return self.head(hidden), tuple(updated)


def trace_cached(model, cache_dtype="fp32"):
    """Compare cached decoding to HF at unseen past lengths and beam sizes."""
    model.cpu().eval()
    prefill=CachedDecoder(model, prefill=True, cache_dtype=cache_dtype).eval()
    step=CachedDecoder(model, cache_dtype=cache_dtype).eval()
    generator=torch.Generator().manual_seed(20260914)
    width=model.config.decoder.hidden_size
    hidden=torch.randn(1, 5, width, generator=generator)
    tokens=torch.randint(1, model.config.decoder.vocab_size, (1, 4), generator=generator)
    with torch.inference_mode():
        _, cache=prefill(tokens, hidden)
        prefill_graph=torch.jit.freeze(torch.jit.trace(prefill, (tokens, hidden), check_trace=False).eval())
        step_graph=torch.jit.freeze(torch.jit.trace(step, (tokens[:, :1], hidden, cache), check_trace=False).eval())
        checks=[]
        for batch in (1, 4):
            for length in (1, 4, 9):
                hidden=torch.randn(batch, 7, width, generator=generator)
                tokens=torch.randint(1, model.config.decoder.vocab_size, (batch, length+3), generator=generator)
                reference=model.decoder(input_ids=tokens[:, :length], encoder_hidden_states=hidden,
                                        use_cache=True, return_dict=True)
                logits, cache=prefill_graph(tokens[:, :length], hidden)
                torch.testing.assert_close(logits, reference.logits, atol=2e-4, rtol=2e-4)
                errors=[]
                matches=[]
                for offset in range(3):
                    token=tokens[:, length+offset:length+offset+1]
                    reference=model.decoder(input_ids=token, encoder_hidden_states=hidden,
                                            past_key_values=reference.past_key_values,
                                            use_cache=True, return_dict=True)
                    logits, cache=step_graph(token, hidden, cache)
                    if cache_dtype=="fp32":
                        torch.testing.assert_close(logits, reference.logits, atol=2e-4, rtol=2e-4)
                    errors.append((logits-reference.logits).abs().max().item())
                    matches.append(bool(torch.equal(logits.argmax(-1), reference.logits.argmax(-1))))
                checks.append({"batch": batch, "past_length": length,
                               "max_abs_logit_errors": errors, "argmax_equal": matches,
                               "cache_bytes": sum(t.numel()*t.element_size() for t in cache)})
    return prefill_graph, step_graph, checks


def export_cached(source, output, cache_dtype="fp32"):
    source=Path(source)
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new export directory")
    model=load_int8(source)
    prefill, step, checks=trace_cached(model, cache_dtype)
    size=model.config.encoder.image_size
    height, width=(size, size) if isinstance(size, int) else size
    pixels=torch.zeros(1, model.config.encoder.num_channels, height, width)
    with torch.inference_mode():
        wrapper=EncoderGraph(model).eval()
        encoder=torch.jit.freeze(torch.jit.trace(wrapper, (pixels,), check_trace=False).eval())
        torch.testing.assert_close(encoder(pixels), wrapper(pixels))
    output.mkdir(parents=True, exist_ok=True)
    for name, graph in (("encoder.pt", encoder), ("decoder_prefill.pt", prefill), ("decoder_step.pt", step)):
        torch.jit.save(graph, str(output/name))
    for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                 "special_tokens_map.json", "tokenizer_config.json", "vocab.txt",
                 "tokenizer.json", "added_tokens.json"):
        if (source/name).is_file():
            shutil.copyfile(source/name, output/name)
    manifest={"format_version": 1, "source": str(source.resolve()),
              "engine": torch.backends.quantized.engine, "torch_version": torch.__version__,
              "encoder": "encoder.pt", "decoder_prefill": "decoder_prefill.pt",
              "decoder_step": "decoder_step.pt", "use_cache": True,
              "cache_dtype": cache_dtype, "cache_order": ["self_key", "self_value", "cross_key", "cross_value"],
              "decoder_output": "full_sequence_logits_and_flat_cache", "verification": checks}
    (output/"script_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    print(json.dumps(export_cached(args.model, args.output, args.cache_dtype), indent=2))
