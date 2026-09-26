"""Small, offline checks of compression numerics and student reload/generation."""

import tempfile
import unittest
from pathlib import Path
import sys

import torch
from torch import nn
from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig
from transformers import VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.ocr_compression import (
    LowRankLinear, build_student, cache_tensor_bytes, factorize_model,
    quantize_dynamic_int8, export_int8, load_int8,
)


class CompressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        torch.set_num_threads(1)

    def test_full_rank_reconstructs_bias_and_double_precision(self):
        original=nn.Linear(7, 5, dtype=torch.float64).eval()
        original.weight.requires_grad_(False)
        reduced=LowRankLinear(original, 5)
        inputs=torch.randn(3, 7, dtype=torch.float64)
        torch.testing.assert_close(reduced(inputs), original(inputs))
        self.assertFalse(reduced.training)
        self.assertFalse(reduced.down.weight.requires_grad)
        self.assertEqual(reduced.up.weight.dtype, torch.float64)

    def test_rank_two_matrix_reconstruction_and_copy(self):
        original=nn.Sequential(nn.Linear(12, 10), nn.ReLU())
        with torch.no_grad():
            original[0].weight.copy_(torch.randn(10, 2)@torch.randn(2, 12))
        reduced=factorize_model(original, rank_fraction=0.2)
        self.assertIsInstance(original[0], nn.Linear)
        self.assertIsInstance(reduced[0], LowRankLinear)
        inputs=torch.randn(5, 12)
        torch.testing.assert_close(reduced(inputs), original(inputs), atol=1e-5, rtol=1e-5)
        self.assertLess(sum(p.numel() for p in reduced.parameters()),
                        sum(p.numel() for p in original.parameters()))

    def test_factorization_filter_and_invalid_rank(self):
        original=nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        reduced=factorize_model(original, predicate=lambda name, module: name=="0")
        self.assertIsInstance(reduced[0], LowRankLinear)
        self.assertIsInstance(reduced[1], nn.Linear)
        with self.assertRaises(ValueError):
            factorize_model(original, rank_fraction=0)

    def test_quantization_root_and_nested_numerics(self):
        for original in [nn.Linear(16, 8), nn.Sequential(nn.Linear(16, 8))]:
            quantized=quantize_dynamic_int8(original)
            inputs=torch.randn(4, 16)
            torch.testing.assert_close(quantized(inputs), original(inputs), atol=0.03, rtol=0.03)
            layer=quantized if isinstance(original, nn.Linear) else quantized[0]
            self.assertEqual(layer.weight().dtype, torch.qint8)
            self.assertEqual(next(original.parameters()).dtype, torch.float32)

    def test_cache_storage_deduplication(self):
        tensor=torch.zeros(2, 3, 4, dtype=torch.float32)
        cache=((tensor, tensor[:, :, :2]), {"v": torch.zeros(5, dtype=torch.float16)})
        self.assertEqual(cache_tensor_bytes(cache), 2*3*4*4+5*2)
        recursive=[]
        recursive.append(recursive)
        self.assertEqual(cache_tensor_bytes(recursive), 0)

    def test_student_layer_selection_reload_and_cache_generation(self):
        encoder=ViTConfig(image_size=16, patch_size=8, hidden_size=16,
                          num_hidden_layers=4, num_attention_heads=2, intermediate_size=24)
        decoder=BertConfig(vocab_size=23, hidden_size=16, num_hidden_layers=3,
                           num_attention_heads=2, intermediate_size=24, is_decoder=True,
                           add_cross_attention=True, pad_token_id=0)
        config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
        config.decoder_start_token_id=1
        config.pad_token_id=0
        config.eos_token_id=2
        teacher=VisionEncoderDecoderModel(config).eval()
        student=build_student(teacher, 2, 2).eval()
        self.assertEqual(len(teacher.encoder.encoder.layer), 4)
        self.assertEqual(student.config.encoder.num_hidden_layers, 2)
        self.assertEqual(student.config.decoder.num_hidden_layers, 2)
        torch.testing.assert_close(student.encoder.encoder.layer[1].layernorm_before.weight,
                                   teacher.encoder.encoder.layer[3].layernorm_before.weight)
        torch.testing.assert_close(student.decoder.bert.encoder.layer[1].attention.self.query.weight,
                                   teacher.decoder.bert.encoder.layer[2].attention.self.query.weight)
        self.assertNotEqual(student.encoder.embeddings.cls_token.data_ptr(),
                            teacher.encoder.embeddings.cls_token.data_ptr())
        pixels=torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            generated=student.generate(pixels, max_new_tokens=4, use_cache=True)
            uncached=student.generate(pixels, max_new_tokens=4, use_cache=False)
            self.assertTrue(torch.equal(generated, uncached))
            with tempfile.TemporaryDirectory() as directory:
                student.save_pretrained(directory)
                restored=VisionEncoderDecoderModel.from_pretrained(directory).eval()
                self.assertTrue(torch.equal(generated, restored.generate(pixels, max_new_tokens=4)))
        with self.assertRaises(ValueError):
            build_student(teacher, 5, 2)

    def test_int8_export_direct_load_preserves_cached_generation(self):
        encoder=ViTConfig(image_size=16, patch_size=8, hidden_size=16,
                          num_hidden_layers=2, num_attention_heads=2, intermediate_size=24)
        decoder=BertConfig(vocab_size=23, hidden_size=16, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=24, is_decoder=True,
                           add_cross_attention=True, pad_token_id=0)
        config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
        config.decoder_start_token_id=1
        config.pad_token_id=0
        teacher=VisionEncoderDecoderModel(config).eval()
        pixels=torch.randn(1, 3, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            exported=export_int8(teacher, directory)
            restored=load_int8(directory)
            self.assertFalse(any(tensor.is_meta for tensor in restored.buffers()))
            with torch.inference_mode():
                expected=exported.generate(pixels, max_new_tokens=5, use_cache=True)
                actual=restored.generate(pixels, max_new_tokens=5, use_cache=True)
                self.assertTrue(torch.equal(expected, actual))
                ids=torch.tensor([[1, 4, 7]])
                torch.testing.assert_close(exported(pixel_values=pixels, decoder_input_ids=ids).logits,
                                           restored(pixel_values=pixels, decoder_input_ids=ids).logits,
                                           rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                export_int8(teacher, directory)


if __name__=="__main__":
    unittest.main()
