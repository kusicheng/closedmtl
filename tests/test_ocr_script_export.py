"""Verify traced attention remains causal across sequence and beam sizes."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig
from transformers import VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.ocr_compression import quantize_dynamic_int8
from training_scripts.ocr_script_export import trace_and_verify


class ScriptExportTests(unittest.TestCase):
    def test_quantized_dynamic_lengths_and_reload(self):
        torch.set_num_threads(1)
        torch.manual_seed(14)
        encoder=ViTConfig(image_size=16, patch_size=8, hidden_size=16,
                          num_hidden_layers=2, num_attention_heads=2, intermediate_size=24)
        decoder=BertConfig(vocab_size=23, hidden_size=16, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=24, is_decoder=True,
                           add_cross_attention=True, pad_token_id=0)
        config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
        model=quantize_dynamic_int8(VisionEncoderDecoderModel(config))
        encoder_graph, decoder_graph, checks=trace_and_verify(model)
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(check["argmax_equal"] for check in checks))
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path=str(Path(directory)/"decoder.pt")
            decoder_graph.save(path)
            restored=torch.jit.load(path)
            hidden=encoder_graph(torch.randn(4, 3, 16, 16))
            tokens=torch.randint(1, 23, (4, 13))
            torch.testing.assert_close(restored(tokens, hidden), decoder_graph(tokens, hidden))
            # Changing future tokens must not influence the first-token logit.
            changed=tokens.clone()
            changed[:, 1:]=(changed[:, 1:]+1)%23
            torch.testing.assert_close(restored(tokens, hidden)[:, 0],
                                       restored(changed, hidden)[:, 0], atol=0.02, rtol=0.02)


if __name__=="__main__":
    unittest.main()
