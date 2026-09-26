"""Test transfer contracts with actual YOLO11 model topology on CPU."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from ultralytics.nn.tasks import DetectionModel, SegmentationModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts import bubble_models


class BubbleModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_box_round_trip_preserves_masks_and_updates_shared_values(self):
        segment=SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
        before={key:value.clone() for key, value in segment.state_dict().items()}
        detector, report=bubble_models.detection_copy(segment)
        self.assertGreater(report["mask_tensors_retained_in_segment"], 0)
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in detector.state_dict().items()))
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in segment.state_dict().items()))
        with torch.no_grad():
            detector.model[0].conv.weight.add_(0.25)
            detector.model[-1].cv3[0][-1].bias.add_(0.5)
        report=bubble_models.restore_detection_weights(segment, detector)
        self.assertEqual(report["reinitialized_tensors"], 0)
        self.assertTrue(torch.equal(segment.model[0].conv.weight, detector.model[0].conv.weight))
        self.assertFalse(torch.equal(segment.model[0].conv.weight, before["model.0.conv.weight"]))
        for key in report["preserved_mask_keys"]:
            self.assertTrue(torch.equal(segment.state_dict()[key], before[key]), key)

    def test_incompatible_restore_does_not_partially_modify_segment(self):
        segment=SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
        detector=DetectionModel("yolo11n.yaml", nc=2, verbose=False)
        before={key:value.clone() for key, value in segment.state_dict().items()}
        with self.assertRaises(ValueError):
            bubble_models.restore_detection_weights(segment, detector)
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in segment.state_dict().items()))

    def test_six_class_detector_leaves_new_classifier_and_masks_untouched(self):
        detector=DetectionModel("yolo11s.yaml", nc=6, verbose=False)
        wrapper=type("LoadedYOLO", (), {"model":detector})()
        seed=41
        expected=bubble_models._new_segment("s", seed)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"source.pt"
            path.write_bytes(b"local-test-checkpoint-fingerprint")
            with patch.object(bubble_models, "YOLO", return_value=wrapper):
                segment, report=bubble_models.build_small_segment(path, seed=seed)
        self.assertEqual(report["copied_tensors"], 493)
        self.assertEqual(len(report["new_classifier_keys"]), 6)
        self.assertEqual(len(report["new_mask_keys"]), 62)
        for key in report["reinitialized_keys"]:
            self.assertTrue(torch.equal(segment.state_dict()[key], expected.state_dict()[key]), key)
        for key in report["copied_keys"]:
            self.assertTrue(torch.equal(segment.state_dict()[key], detector.state_dict()[key]), key)

    def test_mayocream_restores_every_tensor_and_only_defaults_counters(self):
        model=SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
        source={key:value for key, value in model.state_dict().items()
                if not key.endswith(".num_batches_tracked")}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.safetensors"
            save_file(source, str(path))
            segment, report=bubble_models.load_mayocream(path)
        self.assertEqual(report["copied_tensors"], 471)
        self.assertEqual(report["reinitialized_tensors"], 90)
        for key, value in source.items():
            self.assertTrue(torch.equal(segment.state_dict()[key], value), key)
        for key in report["omitted_batchnorm_counters"]:
            self.assertEqual(segment.state_dict()[key].item(), 0)

    def test_mayocream_rejects_missing_learned_tensor(self):
        model=SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
        source=deepcopy(model.state_dict())
        del source["model.0.conv.weight"]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.safetensors"
            save_file(source, str(path))
            with self.assertRaises(ValueError):
                bubble_models.load_mayocream(path)


if __name__=="__main__":
    unittest.main()
