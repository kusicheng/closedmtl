"""CPU checks for deployed box regression, preserved labels, and report identity."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from training_scripts.evaluate_bubble_boxes import deployed_boxes, evaluate, score_image, source_boxes


def annotation(ident=1, category=1, bbox=None, image_id=1):
    return {"id":ident, "image_id":image_id, "category_id":category,
            "bbox":bbox if bbox is not None else [1, 1, 3, 2], "iscrowd":0}


def record(annotations=None):
    return {"image_path":"page.png", "source_image":"source.png", "source_sha256":"a"*64,
            "width":8, "height":6, "annotations":annotations if annotations is not None else [annotation()]}


def prediction(bbox=None, confidence=0.9):
    return {"bbox":bbox if bbox is not None else [1, 1, 4, 3], "confidence":confidence}


class FakeModel:
    def __init__(self, outputs, after_prediction=None):
        self.outputs=list(outputs)
        self.settings=[]
        self.after_prediction=after_prediction

    def predict(self, image, **settings):
        self.settings.append(settings)
        predictions=self.outputs.pop(0)
        boxes=SimpleNamespace(xyxy=torch.tensor([row["bbox"] for row in predictions]).reshape(-1, 4),
                              conf=torch.tensor([row["confidence"] for row in predictions]))
        masks=SimpleNamespace(data=torch.ones((len(predictions), *image.shape[:2]))) if predictions else None
        if self.after_prediction is not None:
            self.after_prediction()
        return [SimpleNamespace(boxes=boxes, masks=masks)]


class BoxMatchingTests(unittest.TestCase):
    def test_six_categories_merge_and_duplicate_predictions_stay_false_positive(self):
        annotations=[annotation(index, index+1, [index, 1, 1, 2]) for index in range(6)]
        predictions=[prediction([index, 1, index+1, 3]) for index in range(6)]
        predictions.append(predictions[0])
        row=score_image(record(annotations), predictions)
        self.assertEqual((row["boxes"]["tp"], row["boxes"]["fp"], row["boxes"]["fn"]), (6, 1, 0))
        self.assertEqual(len({pair["annotation_id"] for pair in row["matches"]}), 6)

    def test_one_pixel_overhang_is_preserved_and_explicit(self):
        source=record([annotation(bbox=[7, 1, 2, 2])])
        row=score_image(source, [prediction([7, 1, 8, 3])])
        self.assertEqual(row["matches"][0]["iou"], 0.5)
        self.assertEqual(row["source_box_overhangs"],
                         [{"annotation_id":1, "right_pixels":1, "bottom_pixels":0}])
        self.assertEqual(source["annotations"][0]["bbox"], [7, 1, 2, 2])
        source["annotations"][0]["bbox"][2]=2.001
        with self.assertRaisesRegex(ValueError, "tolerance"):
            source_boxes(source)

    def test_negative_page_predictions_count_false_positives_and_empty_page_scores_zero(self):
        for predictions in ([], [prediction()]):
            row=score_image(record([]), predictions)
            self.assertEqual(row["boxes"]["tp"], 0)
            self.assertEqual(row["boxes"]["fp"], len(predictions))
            self.assertEqual(row["boxes"]["f1"], 0.0)

    def test_deployment_score_filter_matches_float32_confidence_boundary(self):
        model=FakeModel([[prediction(confidence=float(np.float32(0.35))), prediction(confidence=0.349)]])
        rows=deployed_boxes(model, np.zeros((6, 8, 3), dtype=np.uint8), {})
        self.assertEqual(len(rows), 1)

    def test_invalid_categories_boxes_and_mixed_source_ids_fail(self):
        cases=[[annotation(category=0)], [annotation(bbox=[-1, 1, 1, 1])],
               [annotation(bbox=[1, 1, float("nan"), 1])],
               [annotation(), annotation(2, image_id=2)]]
        for annotations in cases:
            with self.subTest(annotations=annotations):
                with self.assertRaises(ValueError):
                    source_boxes(record(annotations))


class BoxEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.rows=[record(), record([])]
        for index, row in enumerate(self.rows):
            path=self.root/f"page_{index}.png"
            Image.new("RGB", (8, 6), (index, 0, 0)).save(path)
            row.update({"image_path":str(path), "source_image":str(self.root/f"source_{index}.png"),
                        "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest=self.root/"boxes.jsonl"
        self.manifest.write_text("".join(json.dumps(row)+"\n" for row in self.rows), encoding="utf-8")
        self.model_path=self.root/"model.pt"
        self.model_path.write_bytes(b"test model identity")
        self.output=self.root/"report.json"
        self.args=SimpleNamespace(manifest=self.manifest, model=self.model_path, output=self.output,
                                  mayocream=False, device="cpu", limit=None)

    def test_full_coverage_global_f1_identity_settings_and_no_mask_score(self):
        model=FakeModel([[prediction()], [prediction()]])
        report=evaluate(self.args, model=model)
        self.assertTrue(report["complete"])
        self.assertEqual((report["images"], report["targets"], report["negative_images"]), (2, 1, 1))
        self.assertAlmostEqual(report["boxes"]["f1"], 2/3)
        self.assertFalse(report["acceptance_eligible"])
        self.assertEqual(report["usage"], "development_regression_only")
        self.assertNotIn("masks", report)
        rows=[json.loads(line) for line in self.output.with_suffix(".predictions.jsonl").read_text().splitlines()]
        self.assertEqual([row["image_id"] for row in rows], [row["source_image"] for row in self.rows])
        self.assertTrue(all(row["identity_sha256"]==report["identity_sha256"] for row in rows))
        for name, expected in (("imgsz", 768), ("half", False), ("retina_masks", True),
                               ("conf", 0.35), ("iou", 0.5), ("agnostic_nms", True)):
            self.assertEqual(model.settings[0][name], expected)
        with self.assertRaises(FileExistsError):
            evaluate(self.args, model=FakeModel([]))

    def test_wrong_image_hash_leaves_partial_evidence_without_report(self):
        Image.new("RGB", (8, 6), "red").save(self.rows[1]["image_path"])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            evaluate(self.args, model=FakeModel([[prediction()]]))
        self.assertFalse(self.output.exists())
        self.assertEqual(len(self.output.with_suffix(".predictions.jsonl").read_text().splitlines()), 1)

    def test_wrong_image_dimensions_fail_even_when_hash_matches(self):
        path=Path(self.rows[0]["image_path"])
        Image.new("RGB", (7, 6)).save(path)
        self.rows[0]["source_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest.write_text("".join(json.dumps(row)+"\n" for row in self.rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            evaluate(self.args, model=FakeModel([]))
        self.assertFalse(self.output.exists())

    def test_changed_checkpoint_cannot_publish_old_predictions_with_new_identity(self):
        model=FakeModel([[prediction()], []], after_prediction=lambda:self.model_path.write_bytes(b"new checkpoint"))
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluate(self.args, model=model)
        self.assertFalse(self.output.exists())

    def test_smoke_limit_never_marks_full_coverage(self):
        self.args.limit=10
        report=evaluate(self.args, model=FakeModel([[prediction()], []]))
        self.assertFalse(report["complete"])
        self.assertFalse(report["acceptance_eligible"])
        self.assertEqual(report["images"], 2)

    def test_duplicate_manifest_page_fails_before_evidence_creation(self):
        self.manifest.write_text((json.dumps(self.rows[0])+"\n")*2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate(self.args, model=FakeModel([]))
        self.assertFalse(self.output.with_suffix(".identity.json").exists())


if __name__=="__main__":
    unittest.main()
