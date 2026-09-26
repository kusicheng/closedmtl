"""Exercise traced decoder cache shapes, reuse, precision, and reload."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig
from transformers import VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.ocr_compression import quantize_dynamic_int8
from training_scripts.ocr_cached_export import trace_cached


class CachedExportTests(unittest.TestCase):
    def test_cached_precision_shapes_and_reload(self):
        torch.set_num_threads(1)
        torch.manual_seed(14)
        encoder=ViTConfig(image_size=16, patch_size=8, hidden_size=16,
                          num_hidden_layers=2, num_attention_heads=2, intermediate_size=24)
        decoder=BertConfig(vocab_size=23, hidden_size=16, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=24, is_decoder=True,
                           add_cross_attention=True, pad_token_id=0)
        model=quantize_dynamic_int8(VisionEncoderDecoderModel(
            VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)))
        byte_counts=[]
        for precision in ("fp32", "fp16"):
            prefill, step, checks=trace_cached(model, precision)
            self.assertEqual(len(checks), 6)
            byte_counts.append(checks[-1]["cache_bytes"])
            with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
                path=str(Path(directory)/"step.pt")
                step.save(path)
                restored=torch.jit.load(path)
                hidden=torch.randn(4, 11, 16)
                _, cache=prefill(torch.ones(4, 13, dtype=torch.long), hidden)
                ids=torch.ones(4, 2, dtype=torch.long)
                actual, updated=restored(ids, hidden, cache)
                expected, _=step(ids, hidden, cache)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(len(updated), 8)
                self.assertEqual(updated[0].shape, (4, 2, 15, 8))
                self.assertEqual(updated[2].shape, (4, 2, 11, 8))
                self.assertEqual(updated[0].dtype,
                                 torch.float16 if precision=="fp16" else torch.float32)
        self.assertEqual(byte_counts[0], byte_counts[1]*2)


if __name__=="__main__":
    unittest.main()
