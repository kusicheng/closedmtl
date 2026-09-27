"""CPU-only deployment selection; no model inference or real checkpoints required."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training_scripts.predict_bubble_segments import (
    digest, predict, resolve_deployment, verify_deployment_unchanged,
)


class DeploymentConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name).resolve()
        self.model=self.root/"segment.pt"
        self.model.write_bytes(b"checkpoint identity fixture")
        self.config=self.root/"deployment.json"
        self.output=self.root/"new_predictions"

    def write_config(self, **changes):
        value={"model_sha256":digest(self.model), "imgsz":1024}
        value.update(changes)
        self.config.write_text(json.dumps(value), encoding="utf-8")

    def assert_rejected_before_model_or_output(self, override=None):
        args=SimpleNamespace(model=str(self.model), image=str(self.root/"absent_image.png"),
                             output=str(self.output), imgsz=override, mayocream=False, device="cpu")
        with patch("training_scripts.predict_bubble_segments.YOLO") as model_loader:
            with patch("training_scripts.predict_bubble_segments.cv2.imdecode") as decoder:
                with self.assertRaises(ValueError):
                    predict(args)
                model_loader.assert_not_called()
                decoder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_absent_config_keeps_legacy768_without_creating_outputs(self):
        selected=resolve_deployment(self.model)
        self.assertEqual(selected["effective_imgsz"], 768)
        self.assertEqual(selected["imgsz_source"], "legacy_default")
        self.assertIsNone(selected["config_path"])
        self.assertIsNone(selected["config_sha256"])
        self.assertIsNone(selected["explicit_imgsz"])
        self.assertFalse(self.output.exists())

    def test_valid_hash_bound_config_selects1024_and_records_evidence(self):
        self.write_config()
        selected=resolve_deployment(self.model)
        self.assertEqual(selected["effective_imgsz"], 1024)
        self.assertEqual(selected["imgsz_source"], "deployment_config")
        self.assertEqual(selected["config_path"], str(self.config))
        self.assertEqual(selected["config_sha256"], digest(self.config))
        self.assertEqual(selected["model_sha256"], digest(self.model))
        verify_deployment_unchanged(selected)

    def test_explicit_override_wins_and_preserves_config_identity(self):
        self.write_config()
        selected=resolve_deployment(self.model, 1280)
        self.assertEqual(selected["effective_imgsz"], 1280)
        self.assertEqual(selected["explicit_imgsz"], 1280)
        self.assertEqual(selected["config_imgsz"], 1024)
        self.assertEqual(selected["imgsz_source"], "explicit_override")
        self.assertEqual(selected["config_sha256"], digest(self.config))

    def test_explicit_override_without_config(self):
        selected=resolve_deployment(self.model, 640)
        self.assertEqual(selected["effective_imgsz"], 640)
        self.assertEqual(selected["imgsz_source"], "explicit_override")
        self.assertIsNone(selected["config_path"])

    def test_stale_config_rejected_even_with_explicit_override(self):
        self.write_config(model_sha256="0"*64)
        self.assert_rejected_before_model_or_output(1024)

    def test_invalid_config_sizes_rejected_before_model_or_output(self):
        for size in (0, -32, 1000, 1024.0, True, "1024", None):
            with self.subTest(size=size):
                self.write_config(imgsz=size)
                self.assert_rejected_before_model_or_output(1024)

    def test_invalid_explicit_sizes_rejected_before_model_or_output(self):
        for size in (0, -32, 1000, 1024.0, True, "1024"):
            with self.subTest(size=size):
                self.assert_rejected_before_model_or_output(size)

    def test_malformed_unknown_or_duplicate_config_fields_rejected(self):
        values=("{", "[]", "{}", json.dumps({"model_sha256":digest(self.model), "imgsz":1024, "conf":0.1}),
                '{"model_sha256":"'+digest(self.model)+'","imgsz":768,"imgsz":1024}')
        for text in values:
            with self.subTest(text=text):
                self.config.write_text(text, encoding="utf-8")
                self.assert_rejected_before_model_or_output()

    def test_config_change_after_selection_prevents_publication(self):
        self.write_config()
        selected=resolve_deployment(self.model)
        self.write_config(imgsz=1280)
        with self.assertRaisesRegex(RuntimeError, "Deployment config changed"):
            verify_deployment_unchanged(selected)

    def test_model_change_invalidates_existing_config(self):
        self.write_config()
        self.model.write_bytes(b"different checkpoint")
        self.assert_rejected_before_model_or_output()


if __name__=="__main__":
    unittest.main()
