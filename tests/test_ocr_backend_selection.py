"""Offline CLI routing and lazy local OCR backend selection."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_networking_components import image_pipeline, ocr_regions
import main as entrypoint


class OcrBackendSelectionTests(unittest.TestCase):
    def test_cached_export_selects_cached_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            weights=root/"weights.pt"
            weights.touch()
            root.joinpath("script_manifest.json").write_text('{"use_cache":true}', encoding="utf-8")
            cached=Mock()
            with patch.dict(sys.modules, {"ultralytics":SimpleNamespace(YOLO=Mock()),
                                          "training_scripts.ocr_cached_runtime":SimpleNamespace(CachedScriptOcr=cached)}):
                _, ocr=ocr_regions.load_models(weights, str(root))
            cached.assert_called_once_with(str(root))
            self.assertIs(ocr, cached.return_value)

    def test_export_selects_script_backend_and_base_remains_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            weights=root/"weights.pt"
            weights.touch()
            export=root/"export"
            export.mkdir()
            export.joinpath("script_manifest.json").write_text("{}", encoding="utf-8")
            yolo=Mock()
            script=Mock()
            manga=Mock()
            with patch.dict(sys.modules, {"ultralytics":SimpleNamespace(YOLO=yolo),
                                          "manga_ocr":SimpleNamespace(MangaOcr=manga),
                                          "training_scripts.ocr_script_runtime":SimpleNamespace(ScriptOcr=script)}):
                detector, ocr=ocr_regions.load_models(weights, str(export))
                script.assert_called_once_with(str(export))
                manga.assert_not_called()
                self.assertIs(ocr, script.return_value)
                self.assertIs(detector, yolo.return_value)
                _, standard=ocr_regions.load_models(weights)
                manga.assert_called_once_with(pretrained_model_name_or_path="kha-white/manga-ocr-base")
                self.assertIs(standard, manga.return_value)

    def test_images_cli_forwards_option_and_preserves_default_loader_call(self):
        for selected in [None, "models/ocr/local_export"]:
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                args=["input.zip", "--output", str(root/"output"), "--font", "font.ttf",
                      "--weights", "weights.pt"]
                if selected is not None:
                    args.extend(["--ocr-model", selected])
                with patch.object(image_pipeline, "extract_zip", return_value=[root/"page.png"]), \
                     patch.object(image_pipeline, "load_models", return_value=("detector", "ocr")) as load, \
                     patch.object(image_pipeline, "process_images", return_value={"status":"complete"}) as process, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(image_pipeline.main(args), 0)
                if selected is None:
                    load.assert_called_once_with(Path("weights.pt"))
                else:
                    load.assert_called_once_with(Path("weights.pt"), ocr_model=selected)
                self.assertEqual(process.call_args.kwargs["ocr"], "ocr")
                self.assertFalse(list(root.glob(".upload-*")))

    def test_top_level_images_dispatch_preserves_ocr_option(self):
        arguments=["input.zip", "--ocr-model", "models/ocr/local_export"]
        with patch.object(sys, "argv", ["main.py", "images"]+arguments), \
             patch.object(image_pipeline, "main", return_value=0) as images:
            self.assertEqual(entrypoint.main(), 0)
        images.assert_called_once_with(arguments)


if __name__=="__main__":
    unittest.main()
