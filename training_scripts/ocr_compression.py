"""CPU quantization and structural compression for ViT/BERT manga OCR.

Structural changes need accuracy evaluation and usually distillation. Save
factorized/quantized state with transformation settings: save_pretrained alone
cannot describe those custom module types.
"""

from copy import deepcopy
from collections.abc import Mapping
import json
import math
from pathlib import Path

import torch
from torch import nn


class LowRankLinear(nn.Module):
    """Two linear maps initialized with a truncated SVD of a dense map."""

    def __init__(self, source, rank):
        super().__init__()
        if not 1<=rank<=min(source.in_features, source.out_features):
            raise ValueError("rank must fit both linear dimensions")
        self.in_features=source.in_features
        self.out_features=source.out_features
        self.rank=rank
        options={"device": source.weight.device, "dtype": source.weight.dtype}
        self.down=nn.Linear(source.in_features, rank, bias=False, **options)
        self.up=nn.Linear(rank, source.out_features, bias=source.bias is not None, **options)
        # SVD needs float32 for half/bfloat16 inputs; retain float64 precision.
        dtype=torch.float64 if source.weight.dtype==torch.float64 else torch.float32
        with torch.no_grad():
            u, s, vh=torch.linalg.svd(source.weight.to(dtype), full_matrices=False)
            self.down.weight.copy_(vh[:rank])
            self.up.weight.copy_(u[:, :rank]*s[:rank])
            if source.bias is not None:
                self.up.bias.copy_(source.bias)
        self.down.weight.requires_grad_(source.weight.requires_grad)
        self.up.weight.requires_grad_(source.weight.requires_grad)
        if source.bias is not None:
            self.up.bias.requires_grad_(source.bias.requires_grad)
        self.train(source.training)

    def forward(self, inputs):
        return self.up(self.down(inputs))


def factorize_model(model, rank_fraction=0.25, inplace=False, predicate=None):
    """Replace eligible Linear modules, returning the transformed model.

    Rank is ceil(min(in, out)*fraction). Layers without parameter savings are
    retained. predicate(name, module) can restrict replacement. The default
    copies the input; inplace=True avoids that extra memory during conversion.
    """
    if not 0<rank_fraction<=1:
        raise ValueError("rank_fraction must be in (0, 1]")
    result=model if inplace else deepcopy(model)
    replacements={}

    def replace(module, path):
        if isinstance(module, nn.Linear):
            if predicate is not None and not predicate(path, module):
                return module
            rank=math.ceil(min(module.in_features, module.out_features)*rank_fraction)
            if rank*(module.in_features+module.out_features)>=module.weight.numel():
                return module
            if id(module) not in replacements:
                replacements[id(module)]=LowRankLinear(module, rank)
            return replacements[id(module)]
        # _modules preserves aliases, unlike named_children's deduplication.
        for name, child in list(module._modules.items()):
            if child is not None:
                module._modules[name]=replace(child, f"{path}.{name}" if path else name)
        return module

    return replace(result, "")


def quantize_dynamic_int8(model, inplace=False):
    """Return an eval-mode CPU model with dynamically quantized Linear maps."""
    result=model if inplace else deepcopy(model)
    result.cpu().eval()
    # Wrapping also converts a model whose root is itself a Linear.
    wrapper=nn.Sequential(result)
    return torch.ao.quantization.quantize_dynamic(
        wrapper, {nn.Linear}, dtype=torch.qint8, inplace=True
    )[0]


def export_int8(model, output, inplace=False):
    """Save an inference-only INT8 checkpoint for direct loading without FP32.

    Conversion happens during export, outside the deployment process. Only
    ViT/BERT VisionEncoderDecoder models are supported. Save processor and
    tokenizer files to the same output directory separately.
    """
    from torch.ao.nn.quantized.dynamic import Linear as Int8Linear

    output=Path(output)
    filenames=("config.json", "generation_config.json", "int8_state.pt", "int8_manifest.json")
    if any((output/name).exists() for name in filenames):
        raise FileExistsError("INT8 checkpoint files already exist")
    result=quantize_dynamic_int8(model, inplace=inplace)
    if not hasattr(result.decoder, "bert"):
        raise ValueError("INT8 export requires a BERT decoder")
    output.mkdir(parents=True, exist_ok=True)
    result.config.save_pretrained(output)
    result.generation_config.save_pretrained(output)
    # Keep the OrderedDict's _metadata: quantized loaders use its versions.
    torch.save(result.state_dict(), output/"int8_state.pt")
    manifest={"format_version": 1, "engine": torch.backends.quantized.engine,
              "linear_modules": [name for name, module in result.named_modules()
                                 if isinstance(module, Int8Linear)]}
    (output/"int8_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def load_int8(path):
    """Load a locally exported INT8 model without materializing dense weights."""
    from torch.ao.nn.quantized.dynamic import Linear as Int8Linear
    from transformers import GenerationConfig, VisionEncoderDecoderConfig
    from transformers import VisionEncoderDecoderModel

    path=Path(path)
    manifest=json.loads((path/"int8_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version")!=1:
        raise ValueError("Unsupported INT8 checkpoint format")
    engine=manifest["engine"]
    if engine not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Quantized engine is unavailable: {engine}")
    torch.backends.quantized.engine=engine
    config=VisionEncoderDecoderConfig.from_pretrained(path, local_files_only=True)
    with torch.device("meta"):
        model=VisionEncoderDecoderModel(config)
    for name in manifest["linear_modules"]:
        module=model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Expected a dense Linear at {name}")
        parent_name, _, child_name=name.rpartition(".")
        parent=model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, Int8Linear(module.in_features, module.out_features,
                                             bias_=module.bias is not None))
    state=torch.load(path/"int8_state.pt", map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, assign=True, strict=True)
    # Nonpersistent BERT buffers are absent from state_dict. Leaving them on
    # meta can silently produce incorrect generation instead of raising.
    embeddings=model.decoder.bert.embeddings
    positions=embeddings.position_ids.shape[-1]
    embeddings.position_ids=torch.arange(positions, dtype=torch.long).expand((1, -1))
    embeddings.token_type_ids=torch.zeros((1, positions), dtype=torch.long)
    if any(tensor.device.type=="meta" for tensor in model.parameters()):
        raise RuntimeError("INT8 load left unmaterialized parameters")
    if any(tensor.device.type=="meta" for tensor in model.buffers()):
        raise RuntimeError("INT8 load left unmaterialized buffers")
    model.generation_config=GenerationConfig.from_pretrained(path, local_files_only=True)
    return model.eval()


def _layer_indices(total, requested):
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError("layer count must be an integer")
    if not 1<=requested<=total:
        raise ValueError(f"layer count must be between 1 and {total}")
    if requested==1:
        return [total-1]
    return [round(index*(total-1)/(requested-1)) for index in range(requested)]


def build_student(teacher, encoder_layers, decoder_layers):
    """Copy a ViT/BERT teacher, retaining evenly spaced blocks and endpoints.

    A single requested layer takes the final teacher block. Width, vocabulary,
    embeddings, projection heads, generation settings and tied weights remain.
    The result supports normal Hugging Face save_pretrained/from_pretrained.
    """
    try:
        encoder=teacher.encoder.encoder.layer
        decoder=teacher.decoder.bert.encoder.layer
    except AttributeError as error:
        raise ValueError("build_student requires a ViT encoder and BERT decoder") from error
    encoder_indices=_layer_indices(len(encoder), encoder_layers)
    decoder_indices=_layer_indices(len(decoder), decoder_layers)
    # Memoized replacements avoid copying the teacher blocks we discard.
    memo={}
    selected_encoder=nn.ModuleList([deepcopy(encoder[i], memo) for i in encoder_indices])
    selected_decoder=nn.ModuleList([deepcopy(decoder[i], memo) for i in decoder_indices])
    memo[id(encoder)]=selected_encoder
    memo[id(decoder)]=selected_decoder
    student=deepcopy(teacher, memo)
    student.config.encoder.num_hidden_layers=encoder_layers
    student.encoder.config.num_hidden_layers=encoder_layers
    student.config.decoder.num_hidden_layers=decoder_layers
    student.decoder.config.num_hidden_layers=decoder_layers
    student.decoder.bert.config.num_hidden_layers=decoder_layers
    # Modern Transformers uses layer_idx to address decoder KV-cache slots.
    for index, block in enumerate(student.decoder.bert.encoder.layer):
        for child in block.modules():
            if hasattr(child, "layer_idx"):
                child.layer_idx=index
    return student


def cache_tensor_bytes(cache):
    """Count unique tensor storage bytes in legacy or Transformers KV caches.

    Counts actual allocated storage (including a view's complete backing
    storage), excluding Python objects, allocator overhead and process RAM.
    """
    seen_objects=set()
    seen_storages=set()

    def visit(value):
        if value is None or id(value) in seen_objects:
            return 0
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type=="meta":
                return 0
            storage=value.untyped_storage()
            key=(str(value.device), storage.data_ptr(), storage.nbytes())
            if key in seen_storages:
                return 0
            seen_storages.add(key)
            return storage.nbytes()
        if isinstance(value, Mapping):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        if hasattr(value, "to_legacy_cache"):
            return visit(value.to_legacy_cache())
        # Current Cache and CacheLayer objects store tensors in attributes.
        if value.__class__.__module__.startswith("transformers.cache_utils"):
            return visit(vars(value))
        return 0

    return visit(cache)
