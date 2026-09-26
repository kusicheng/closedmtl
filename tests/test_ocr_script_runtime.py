"""Compare independent runtime processing/search to the installed HF teacher API."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from transformers import BertConfig, BertJapaneseTokenizer, ViTConfig, ViTImageProcessor, VisionEncoderDecoderConfig
from transformers import VisionEncoderDecoderModel
from manga_ocr.ocr import post_process as reference_post_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training_scripts"))
from ocr_script_runtime import (beam_search, post_process, preprocess, ban_repeated_ngrams,
                                validate_configuration, load_vocabulary, decode)
import ocr_script_runtime as runtime
from ocr_text import post_process as shared_post_process


class ScriptRuntimeTests(unittest.TestCase):
    def test_callable_accepts_pil_and_path_and_preserves_input(self):
        encoder=Mock(return_value=torch.zeros(1, 2, 3))
        encoder.eval.return_value=encoder
        decoder=Mock()
        decoder.eval.return_value=decoder
        with tempfile.TemporaryDirectory() as temporary:
            image=Image.new("RGB", (9, 13), (180, 30, 75))
            path=Path(temporary)/"image.png"
            image.save(path)
            with patch.object(runtime, "validate_configuration", return_value={}), \
                 patch.object(runtime.torch.jit, "load", side_effect=[encoder, decoder]) as load, \
                 patch.object(runtime, "load_vocabulary", return_value=(["[CLS]", "猫", "[SEP]"],
                                                                         {"[CLS]", "[SEP]"})), \
                 patch.object(runtime, "beam_search", return_value=(torch.tensor([0, 1, 2]), -0.1)):
                ocr=runtime.ScriptOcr(temporary, threads=None)
                self.assertEqual(ocr(image), "猫")
                self.assertEqual(ocr(path), "猫")
                self.assertEqual(ocr.predict_with_score(str(path)), ("猫", -0.1))
                with self.assertRaisesRegex(ValueError, "PIL image"):
                    ocr(17)
            self.assertEqual(image.getpixel((0, 0)), (180, 30, 75))
            self.assertEqual(load.call_args_list[0].args[0], str(Path(temporary)/"encoder.pt"))
            torch.testing.assert_close(encoder.call_args_list[0].args[0], encoder.call_args_list[1].args[0])

    def test_character_decode_matches_special_token_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            directory.joinpath("vocab.txt").write_text(
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n猫\n…\nｶ\nﾞ\n1\n", encoding="utf-8")
            tokenizer=BertJapaneseTokenizer(str(directory/"vocab.txt"),
                                            subword_tokenizer_type="character", do_word_tokenize=False)
            vocabulary, special=load_vocabulary(directory)
            tokens=[2, 5, 6, 7, 8, 9, 1, 4, 3, 0]
            expected=reference_post_process(tokenizer.decode(tokens, skip_special_tokens=True))
            self.assertEqual(decode(tokens, vocabulary, special), expected)

    def test_metadata_rejects_incompatible_processor_and_cuda_int8(self):
        configs={"generation_config.json":{"decoder_start_token_id":2, "eos_token_id":3,
                                          "pad_token_id":0, "num_beams":4, "length_penalty":2.0,
                                          "early_stopping":True, "no_repeat_ngram_size":3,
                                          "max_length":300},
                 "preprocessor_config.json":{"do_resize":True, "do_rescale":True, "do_normalize":True,
                                             "size":{"height":224, "width":224}, "resample":2,
                                             "rescale_factor":1/255, "image_mean":[0.5]*3,
                                             "image_std":[0.5]*3, "image_processor_type":"ViTImageProcessor"},
                 "tokenizer_config.json":{"tokenizer_class":"BertJapaneseTokenizer",
                                         "subword_tokenizer_type":"character",
                                         "clean_up_tokenization_spaces":False},
                 "script_manifest.json":{"format_version":1, "engine":"onednn", "use_cache":False}}
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for name, values in configs.items():
                directory.joinpath(name).write_text(json.dumps(values), encoding="utf-8")
            validate_configuration(directory)
            self.assertEqual(torch.backends.quantized.engine, "onednn")
            with self.assertRaisesRegex(ValueError, "requires CPU"):
                validate_configuration(directory, "cuda")
            configs["preprocessor_config.json"]["resample"]=3
            directory.joinpath("preprocessor_config.json").write_text(
                json.dumps(configs["preprocessor_config.json"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resample"):
                validate_configuration(directory)

    def test_pixels_and_text_match_reference(self):
        image=Image.fromarray(np.random.default_rng(14).integers(0, 256, (79, 51, 3), dtype=np.uint8))
        processor=ViTImageProcessor(size={"height":224, "width":224})
        expected=processor(image.convert("L").convert("RGB"), return_tensors="pt").pixel_values
        torch.testing.assert_close(preprocess(image), expected, atol=0, rtol=0)
        for text in [" abc 123\n…・・...", "ｶﾞｯﾂ です。", "[UNK]", "・.・a"]:
            self.assertEqual(post_process(text), reference_post_process(text))
            self.assertEqual(shared_post_process(text), reference_post_process(text))

    def test_ngram_bans_only_repeated_completion(self):
        tokens=torch.tensor([[2, 4, 5, 4, 5], [2, 4, 5, 4, 6]])
        scores=torch.zeros(2, 8)
        ban_repeated_ngrams(tokens, scores, 3)
        self.assertTrue(torch.isneginf(scores[0, 4]))
        self.assertEqual(int(torch.isneginf(scores).sum()), 1)

    def test_search_matches_hf_eos_and_max_length(self):
        torch.set_num_threads(1)
        for seed in [3, 14, 22]:
            torch.manual_seed(seed)
            encoder=ViTConfig(image_size=16, patch_size=8, hidden_size=16,
                              num_hidden_layers=1, num_attention_heads=2, intermediate_size=24)
            decoder=BertConfig(vocab_size=11, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=24, is_decoder=True,
                               add_cross_attention=True, max_position_embeddings=32)
            config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
            model=VisionEncoderDecoderModel(config).eval()
            pixels=torch.randn(1, 3, 16, 16)
            for eos_bias in [-5.0, 0.5]:
                with torch.no_grad():
                    model.decoder.cls.predictions.bias[3]=eos_bias
                    hidden=model.encoder(pixels).last_hidden_state
                    expected=model.generate(pixels, max_length=15, num_beams=4,
                                             length_penalty=2.0, early_stopping=True,
                                             no_repeat_ngram_size=3, decoder_start_token_id=2,
                                             eos_token_id=3, pad_token_id=0, use_cache=False,
                                             return_dict_in_generate=True, output_scores=True)
                    actual, score=beam_search(
                        lambda ids, states:model.decoder(ids, encoder_hidden_states=states,
                                                        use_cache=False).logits,
                        hidden, max_length=15)
                torch.testing.assert_close(actual, expected.sequences[0])
                self.assertAlmostEqual(score, float(expected.sequences_scores[0]), places=6)


if __name__=="__main__":
    unittest.main()
