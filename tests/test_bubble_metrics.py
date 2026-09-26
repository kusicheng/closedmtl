"""Synthetic CPU checks of fixed-confidence box and mask matching."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training_scripts"))
from bubble_metrics import FixedDetectionValidator, FixedSegmentationValidator


class FixedBubbleMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def validator(self, segmentation=False, **kwargs):
        kind=FixedSegmentationValidator if segmentation else FixedDetectionValidator
        result=kind(save_dir=Path(self.directory.name),
                    args={"plots":False, "imgsz":32, "conf":0.8,
                          "overlap_mask":False}, **kwargs)
        result.data={"val":"synthetic"}
        result.device=torch.device("cpu")
        result.init_metrics(SimpleNamespace(names={0:"bubble"}))
        return result

    def data(self, boxes, confidence, targets, image="page.png", masks=None,
             target_masks=None):
        preds={"bboxes":torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
               "conf":torch.tensor(confidence, dtype=torch.float32),
               "cls":torch.zeros(len(boxes))}
        batch={"bboxes":torch.tensor(targets, dtype=torch.float32).reshape(-1, 4),
               "cls":torch.zeros(len(targets)), "im_file":image}
        if masks is not None:
            preds["masks"]=masks
            batch["masks"]=target_masks
        return preds, batch

    def test_low_confidence_prediction_cannot_steal_fixed_match(self):
        for segmentation in (False, True):
            with self.subTest(segmentation=segmentation):
                validator=self.validator(segmentation)
                masks=torch.ones((2, 8, 8)) if segmentation else None
                if segmentation:
                    masks[1, :, 6:]=0
                preds, batch=self.data([[0, 0, 10, 10], [0, 0, 8, 10]],
                                       [0.1, 0.9], [[0, 0, 10, 10]],
                                       masks=masks,
                                       target_masks=torch.ones((1, 8, 8)))
                base=SegmentationValidator if segmentation else DetectionValidator
                expected=base._process_batch(validator, preds, batch)
                actual=validator._process_batch(preds, batch)
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(actual[key][:, 0].tolist(), [True, False])
                for kind in ("boxes", "masks") if segmentation else ("boxes",):
                    score=validator.fixed_report[kind]
                    self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 0, 0))
                    self.assertEqual(score["f1"], 1.0)
                self.assertEqual(validator.args.conf, 0.001)

    def test_duplicate_predictions_are_one_to_one(self):
        validator=self.validator(True)
        preds, batch=self.data([[0, 0, 10, 10]]*2, [0.9, 0.8], [[0, 0, 10, 10]],
                               masks=torch.ones((2, 8, 8)),
                               target_masks=torch.ones((1, 8, 8)))
        validator._process_batch(preds, batch)
        for kind in ("boxes", "masks"):
            score=validator.fixed_report[kind]
            self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 1, 0))
            self.assertAlmostEqual(score["f1"], 2/3)

    def test_empty_targets_predictions_and_both_score_zero(self):
        for predictions, targets in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(predictions=predictions, targets=targets):
                validator=self.validator(True)
                preds, batch=self.data([[0, 0, 10, 10]]*predictions,
                                       [0.9]*predictions, [[0, 0, 10, 10]]*targets,
                                       masks=torch.zeros((predictions, 8, 8)),
                                       target_masks=torch.zeros((targets, 8, 8)))
                validator._process_batch(preds, batch)
                for kind in ("boxes", "masks"):
                    score=validator.fixed_report[kind]
                    self.assertEqual((score["tp"], score["fp"], score["fn"]),
                                     (0, predictions, targets))
                    self.assertEqual(score["f1"], 0.0)
                self.assertEqual(validator.fixed_report["valid"], targets>0)

    def test_box_and_mask_matches_are_independent(self):
        validator=self.validator(True)
        pred_mask=torch.zeros((1, 8, 8))
        pred_mask[:, :, :4]=1
        target_mask=1-pred_mask
        preds, batch=self.data([[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10]],
                               masks=pred_mask, target_masks=target_mask)
        validator._process_batch(preds, batch)
        report=validator.fixed_report
        self.assertEqual(report["boxes"]["f1"], 1.0)
        self.assertEqual(report["masks"]["f1"], 0.0)
        self.assertEqual(report["masks"]["fp"], 1)
        self.assertEqual(report["masks"]["fn"], 1)

    def test_global_counts_and_visit_order_survive_statistics(self):
        validator=self.validator()
        samples=[self.data([[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10]], "z.png"),
                 self.data([], [], [[0, 0, 10, 10]]*9, "a.png")]
        for preds, prepared in samples:
            boxes=prepared["bboxes"]
            xywh=boxes.clone()
            xywh[:, :2]=(boxes[:, :2]+boxes[:, 2:])/2
            xywh[:, 2:]=boxes[:, 2:]-boxes[:, :2]
            batch={"img":torch.zeros((1, 3, 32, 32)),
                   "batch_idx":torch.zeros(len(boxes)),
                   "cls":prepared["cls"][:, None], "bboxes":xywh/32,
                   "ori_shape":[(32, 32)], "ratio_pad":[((1, 1), (0, 0))],
                   "im_file":[prepared["im_file"]]}
            validator.update_metrics([preds], batch)
        stats=validator.get_stats()
        self.assertIn("metrics/mAP50(B)", stats)
        self.assertIn("fitness", stats)
        self.assertAlmostEqual(stats["metrics/fixed_f1(B)"], 2/11)
        self.assertEqual(stats["metrics/fixed_targets"], 10)
        self.assertEqual(stats["metrics/fixed_valid"], 1)
        self.assertEqual([row["image"] for row in validator.fixed_report["image_rows"]],
                         ["z.png", "a.png"])
        self.assertEqual(validator.fixed_report["boxes"]["fn"], 9)
        validator.init_metrics(SimpleNamespace(names={0:"bubble"}))
        self.assertEqual(validator.fixed_image_rows, [])
        self.assertEqual(validator.fixed_report["boxes"]["f1"], 0.0)
        self.assertFalse(validator.fixed_report["valid"])

    def test_fixed_confidence_boundary_is_inclusive(self):
        validator=self.validator()
        preds, batch=self.data([[0, 0, 10, 10], [20, 20, 30, 30]], [0.35, 0.349],
                               [[0, 0, 10, 10], [20, 20, 30, 30]])
        validator._process_batch(preds, batch)
        score=validator.fixed_report["boxes"]
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 0, 1))

    def test_segmentation_update_and_returned_metrics_preserve_fixed_report(self):
        validator=self.validator(True)
        preds, _=self.data([[0, 0, 16, 16]], [0.9], [[0, 0, 16, 16]],
                           masks=torch.zeros((1, 8, 8)),
                           target_masks=torch.ones((1, 8, 8)))
        batch={"img":torch.zeros((1, 3, 32, 32)), "batch_idx":torch.zeros(1),
               "cls":torch.zeros((1, 1)), "bboxes":torch.tensor([[0.25, 0.25, 0.5, 0.5]]),
               "masks":torch.ones((1, 8, 8)), "ori_shape":[(32, 32)],
               "ratio_pad":[((1, 1), (0, 0))], "im_file":["mask.png"]}
        validator.update_metrics([preds], batch)
        stats=validator.get_stats()
        self.assertIn("metrics/mAP50(M)", stats)
        self.assertEqual(stats["metrics/fixed_f1(B)"], 1.0)
        self.assertEqual(stats["metrics/fixed_f1(M)"], 0.0)
        self.assertEqual(validator.metrics.fixed_report, validator.fixed_report)
        validator.init_metrics(SimpleNamespace(names={0:"bubble"}))
        self.assertEqual(validator.metrics.fixed_report["image_rows"], [])

    def test_invalid_operating_points_are_rejected(self):
        for confidence in (-1, 0, 1.1, float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    self.validator(fixed_conf=confidence)


if __name__=="__main__":
    unittest.main()
