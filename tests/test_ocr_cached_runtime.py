"""Check cache beam ancestry and cached sequence parity against Hugging Face."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import torch
from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers.cache_utils import EncoderDecoderCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.ocr_cached_runtime import CachedDecoder
from training_scripts.ocr_script_runtime import beam_search


class CachedRuntimeTests(unittest.TestCase):
    def test_reorder_duplicates_every_layer_cache_and_step_uses_last_token(self):
        states=tuple(torch.arange(4).reshape(4, 1, 1, 1)+offset for offset in range(8))
        prefill=Mock(return_value=(torch.zeros(4, 2, 11), states))
        step=Mock(return_value=(torch.zeros(4, 1, 11), states))
        decoder=CachedDecoder(prefill, step)
        hidden=torch.zeros(4, 3, 8)
        ids=torch.tensor([[2, 5]]*4)
        decoder(ids, hidden)
        parents=torch.tensor([2, 2, 0, 3])
        decoder.reorder(parents)
        for original, actual in zip(states, decoder.cache):
            torch.testing.assert_close(actual, original[parents])
        decoder(torch.tensor([[2, 5, 7]]*4), hidden)
        torch.testing.assert_close(step.call_args.args[0], torch.tensor([[7]]*4))
        self.assertEqual(prefill.call_count, 1)
        self.assertEqual(step.call_count, 1)

    def test_cached_beams_match_hf_sequences_and_scores(self):
        torch.set_num_threads(1)
        for seed in [3, 14]:
            torch.manual_seed(seed)
            encoder=ViTConfig(image_size=16, patch_size=8, hidden_size=16,
                              num_hidden_layers=1, num_attention_heads=2, intermediate_size=24)
            decoder=BertConfig(vocab_size=11, hidden_size=16, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=24, is_decoder=True,
                               add_cross_attention=True, max_position_embeddings=32)
            config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
            model=VisionEncoderDecoderModel(config).eval()
            pixels=torch.randn(1, 3, 16, 16)

            def prefill(ids, hidden):
                result=model.decoder(ids, encoder_hidden_states=hidden, use_cache=True)
                return result.logits, tuple(tensor for layer in result.past_key_values.to_legacy_cache()
                                             for tensor in layer)

            def step(ids, hidden, flat):
                cache=EncoderDecoderCache.from_legacy_cache(tuple(tuple(flat[i:i+4])
                                                                  for i in range(0, len(flat), 4)))
                result=model.decoder(ids, encoder_hidden_states=hidden, past_key_values=cache, use_cache=True)
                return result.logits, tuple(tensor for layer in result.past_key_values.to_legacy_cache()
                                             for tensor in layer)

            for eos_bias in [-5.0, 0.5]:
                with torch.inference_mode():
                    model.decoder.cls.predictions.bias[3]=eos_bias
                    hidden=model.encoder(pixels).last_hidden_state
                    expected=model.generate(pixels, max_length=15, num_beams=4,
                                             length_penalty=2.0, early_stopping=True,
                                             no_repeat_ngram_size=3, decoder_start_token_id=2,
                                             eos_token_id=3, pad_token_id=0, use_cache=True,
                                             return_dict_in_generate=True, output_scores=True)
                    adapter=CachedDecoder(prefill, step)
                    tokens, score=beam_search(adapter, hidden, max_length=15, cache_reorder=adapter.reorder)
                torch.testing.assert_close(tokens, expected.sequences[0])
                self.assertAlmostEqual(score, float(expected.sequences_scores[0]), places=6)


if __name__=="__main__":
    unittest.main()
