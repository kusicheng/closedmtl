"""CPU contracts for exact-RLE deployed evaluation and durable partial evidence."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from training_scripts.evaluate_bubble_segments import (
    aggregate_rows, box_ious, crop_mask, deployed_predictions, evaluate,
    mask_ious, match_instances, score_image,
)


def record(image_id=1, book="A", annotations=True):
    annotation={"id":10+image_id, "image_id":image_id, "category_id":5,
                "bbox":[1, 1, 3, 2], "area":6, "iscrowd":0,
                "segmentation":{"size":[4, 5], "counts":"5220003"}}
    return {"image_id":image_id, "book":book, "image_path":"placeholder.png",
            "source_sha256":"placeholder", "width":5, "height":4,
            "annotations":[annotation] if annotations else [],
            "training_polygon_rejected":image_id==1}


def perfect_prediction():
    mask=np.zeros((4, 5))
    mask[1:3, 1:4]=1
    return {"bbox":[1, 1, 4, 3], "confidence":0.9, "mask":crop_mask(mask)}


def missing_annotation(image_id=1):
    return {"id":f"missing:{image_id}", "image_id":image_id, "category_id":5,
            "bbox":[0.0, 0.0, 1.0, 1.0], "area":1, "iscrowd":0,
            "segmentation":None, "mask_status":"missing_source_polygon"}


def unknown_prediction():
    return {"bbox":[0, 0, 1, 1], "confidence":0.9, "mask":crop_mask(np.ones((1, 1)))}


class FakeModel:
    def __init__(self, outputs):
        self.outputs=list(outputs)
        self.settings=[]

    def predict(self, image, **settings):
        self.settings.append(settings)
        predictions=self.outputs.pop(0)
        boxes=SimpleNamespace(xyxy=torch.tensor([p["bbox"] for p in predictions]).reshape(-1, 4),
                              conf=torch.tensor([p["confidence"] for p in predictions]))
        masks=[]
        for prediction in predictions:
            mask=np.zeros(image.shape[:2])
            x1, y1, x2, y2=prediction["mask"]["bbox"]
            mask[y1:y2, x1:x2]=prediction["mask"]["pixels"]
            masks.append(mask)
        return [SimpleNamespace(boxes=boxes,
                                masks=SimpleNamespace(data=torch.tensor(np.array(masks))) if masks else None)]


class ExactMaskTests(unittest.TestCase):
    def test_cropped_intersections_match_full_rasters_with_holes(self):
        rng=np.random.default_rng(2)
        targets=rng.random((7, 20, 30))>0.7
        predictions=rng.random((5, 20, 30))>0.6
        targets[:, :4]=0
        predictions[:, :, :6]=0
        targets[0]=0
        predictions[0]=0
        measured=mask_ious([crop_mask(mask) for mask in targets],
                           [crop_mask(mask) for mask in predictions])
        expected=np.zeros_like(measured)
        for ti, target in enumerate(targets):
            for pi, prediction in enumerate(predictions):
                union=np.count_nonzero(target|prediction)
                expected[ti, pi]=np.count_nonzero(target&prediction)/union if union else 0
        np.testing.assert_array_equal(measured, expected)

    def test_threshold_edges_are_inclusive_and_duplicates_cannot_reuse_target(self):
        boxes=box_ious([[0, 0, 2, 2]], [[0, 0, 1, 2], [0, 0, 2, 2]])
        self.assertEqual(boxes[0, 0], 0.5)
        self.assertEqual(len(match_instances(boxes[:, :1], 0.5)), 1)
        self.assertEqual(len(match_instances(boxes, 0.5)), 1)
        target=crop_mask(np.ones((2, 2)))
        prediction=crop_mask(np.array([[1, 1], [1, 0]]))
        masks=mask_ious([target], [prediction])
        self.assertEqual(masks[0, 0], 0.75)
        self.assertEqual(len(match_instances(masks, 0.75)), 1)
        self.assertEqual(len(match_instances(masks, 0.750001)), 0)

    def test_exact_source_bbox_and_duplicate_predictions(self):
        row=score_image(record(), [perfect_prediction(), perfect_prediction()], (0.5, 0.75))
        for threshold in ("0.50", "0.75"):
            for kind in ("boxes", "masks"):
                counts=row["counts"][threshold][kind]
                self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 1, 0))
                self.assertEqual(row["matches"][threshold][kind][0]["iou"], 1.0)
        self.assertTrue(row["training_polygon_rejected"])

    def test_box_success_does_not_hide_bad_mask(self):
        prediction=perfect_prediction()
        prediction["mask"]=crop_mask(np.ones((1, 1)), origin=(1, 1))
        row=score_image(record(), [prediction])
        self.assertEqual(row["counts"]["0.50"]["boxes"]["f1"], 1.0)
        self.assertEqual(row["counts"]["0.50"]["masks"]["f1"], 0.0)

    def test_negative_pages_and_confidence_filter_do_not_earn_true_positives(self):
        positive=record()
        prediction=perfect_prediction()
        prediction["confidence"]=float(np.float32(0.35))
        self.assertEqual(score_image(positive, [prediction])["counts"]["0.50"]["masks"]["tp"], 1)
        prediction["confidence"]=0.349
        self.assertEqual(score_image(positive, [prediction])["counts"]["0.50"]["masks"]["fn"], 1)
        negative=record(2, annotations=False)
        empty=score_image(negative, [])
        self.assertFalse(aggregate_rows([empty])["valid"])
        self.assertEqual(aggregate_rows([empty])["boxes"]["f1"], 0.0)
        false_positive=score_image(negative, [perfect_prediction()])
        for kind in ("boxes", "masks"):
            self.assertEqual(false_positive["counts"]["0.50"][kind]["fp"], 1)
        mixed=aggregate_rows([score_image(positive, [perfect_prediction()]), false_positive, empty])
        self.assertAlmostEqual(mixed["masks"]["f1"], 2/3)
        self.assertEqual(mixed["negative_images"], 2)


class MissingSourceMaskTests(unittest.TestCase):
    def test_missing_mask_retains_known_masks_and_all_boxes(self):
        source=record()
        source["annotations"].append(missing_annotation())
        row=score_image(source, [perfect_prediction(), unknown_prediction()], (0.5, 0.75))
        self.assertEqual(row["targets"], 2)
        self.assertEqual(row["known_mask_targets"], 1)
        self.assertEqual(row["missing_mask_targets"], 1)
        self.assertEqual(row["ignored_mask_predictions"], 1)
        for key in ("0.50", "0.75"):
            metrics=row["counts"][key]
            self.assertEqual(metrics["boxes"]["tp"], 2)
            self.assertEqual(metrics["masks"]["f1"], 1.0)
            self.assertEqual(metrics["known_mask_targets"], 1)
            self.assertEqual(metrics["missing_mask_targets"], 1)
            self.assertEqual(metrics["masks_conservative"]["fp"], 1)
            self.assertEqual(metrics["masks_conservative"]["fn"], 1)
            self.assertEqual(metrics["mask_f1_lower_bound"], 0.5)

    def test_duplicate_predictions_near_unknown_mask_still_count_false_positive(self):
        source=record()
        source["annotations"].append(missing_annotation())
        row=score_image(source, [perfect_prediction(), unknown_prediction(), unknown_prediction()])
        metrics=row["counts"]["0.50"]
        self.assertEqual(metrics["ignored_mask_predictions"], 1)
        self.assertEqual(metrics["masks"]["tp"], 1)
        self.assertEqual(metrics["masks"]["fp"], 1)
        self.assertEqual(metrics["masks_conservative"]["fp"], 2)
        self.assertEqual(metrics["mask_f1_lower_bound"], 0.4)

    def test_known_mask_match_has_priority_over_unknown_box_assignment(self):
        source=record()
        unknown=missing_annotation()
        unknown["bbox"]=[0, 0, 5, 4]
        source["annotations"].insert(0, unknown)
        prediction=perfect_prediction()
        prediction["bbox"]=[0, 0, 5, 4]
        row=score_image(source, [prediction])
        self.assertEqual(row["matches"]["0.50"]["boxes"][0]["annotation_id"], unknown["id"])
        known=row["matches"]["0.50"]["masks"][0]
        self.assertEqual(known["target_index"], 1)
        self.assertEqual(known["annotation_id"], source["annotations"][1]["id"])
        self.assertEqual(row["ignored_mask_predictions"], 0)
        self.assertEqual(row["counts"]["0.50"]["masks"]["tp"], 1)
        self.assertEqual(row["counts"]["0.50"]["masks"]["fp"], 0)
        self.assertAlmostEqual(row["mask_f1_lower_bound"], 2/3)

    def test_ignore_is_recomputed_at_each_iou_threshold(self):
        source=record()
        unknown=missing_annotation()
        unknown["bbox"]=[0, 0, 2, 1]
        source["annotations"].append(unknown)
        row=score_image(source, [perfect_prediction(), unknown_prediction()], (0.5, 0.75))
        self.assertEqual(row["counts"]["0.50"]["ignored_mask_predictions"], 1)
        self.assertEqual(row["counts"]["0.75"]["ignored_mask_predictions"], 0)
        self.assertEqual(row["counts"]["0.75"]["masks"]["fp"], 1)
        summary=aggregate_rows([row], (0.5, 0.75))
        self.assertEqual(summary["by_iou"]["0.75"]["ignored_mask_predictions"], 0)

    def test_conservative_bound_uses_global_counts_and_penalizes_missing_target(self):
        rows=[score_image(record(image_id=index), [perfect_prediction()]) for index in range(1, 32)]
        source=record(32, annotations=False)
        source["annotations"]=[missing_annotation(32)]
        rows.append(score_image(source, [unknown_prediction()]))
        report=aggregate_rows(rows)
        self.assertEqual(report["known_mask_targets"], 31)
        self.assertEqual(report["missing_mask_targets"], 1)
        self.assertEqual(report["ignored_mask_predictions"], 1)
        self.assertEqual(report["masks"]["f1"], 1.0)
        self.assertEqual(report["masks_conservative"]["tp"], 31)
        self.assertEqual(report["masks_conservative"]["fp"], 1)
        self.assertEqual(report["masks_conservative"]["fn"], 1)
        self.assertEqual(report["mask_f1_lower_bound"], 62/64)

    def test_only_explicit_missing_source_polygon_is_supported(self):
        for status in (None, "unknown", "known"):
            source=record()
            source["annotations"][0]["segmentation"]=None
            source["annotations"][0]["mask_status"]=status
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "explicit"):
                    score_image(source, [])
        source=record()
        source["annotations"][0]["mask_status"]="missing_source_polygon"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            score_image(source, [])


class EvaluationJournalTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        self.rows=[record(), record(2, "B", annotations=False)]
        for row in self.rows:
            path=self.root/f"{row['image_id']}.png"
            Image.new("RGB", (5, 4)).save(path)
            row["image_path"]=str(path)
            row["source_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest=self.root/"full_masks.jsonl"
        self.manifest.write_text("".join(json.dumps(row)+"\n" for row in self.rows), encoding="utf-8")
        self.model_path=self.root/"model.pt"
        self.model_path.write_bytes(b"synthetic model identity")
        self.output=self.root/"result.json"
        self.args=SimpleNamespace(manifest=self.manifest, model=self.model_path, output=self.output,
                                  limit=None, iou75=True, imgsz=768, device="cpu", mayocream=False)

    def test_complete_report_preserves_all_pages_books_hashes_and_deployment_settings(self):
        model=FakeModel([[perfect_prediction()], []])
        result=evaluate(self.args, model=model)
        self.assertTrue(result["complete"])
        self.assertTrue(result["development_target_reached"])
        self.assertFalse(result["acceptance_eligible"])
        self.assertIn("provenance and human label verification", result["acceptance_reason"])
        self.assertNotIn("target_reached", result)
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["negative_images"], 1)
        self.assertEqual(result["polygon_rejected_images"], 1)
        self.assertEqual([book["book"] for book in result["books"]], ["A", "B"])
        self.assertFalse(result["books"][1]["valid"])
        self.assertEqual(result["masks"]["f1"], 1.0)
        rows=[json.loads(line) for line in self.output.with_suffix(".predictions.jsonl").read_text().splitlines()]
        self.assertEqual([row["image_id"] for row in rows], [1, 2])
        self.assertTrue(all(row["identity_sha256"]==result["identity_sha256"] for row in rows))
        self.assertEqual(model.settings[0]["retina_masks"], True)
        self.assertEqual(model.settings[0]["half"], False)
        self.assertEqual(model.settings[0]["conf"], 0.35)
        self.assertEqual(model.settings[0]["iou"], 0.5)
        with self.assertRaises(FileExistsError):
            evaluate(self.args, model=FakeModel([]))

    def test_limit_is_ineligible_even_when_larger_than_entire_manifest(self):
        self.args.limit=100
        result=evaluate(self.args, model=FakeModel([[perfect_prediction()], []]))
        self.assertFalse(result["complete"])
        self.assertFalse(result["acceptance_eligible"])
        self.assertFalse(result["development_target_reached"])
        self.assertEqual(result["images"], 2)

    def test_hash_failure_leaves_only_verified_partial_journal(self):
        Image.new("RGB", (5, 4), "red").save(self.rows[1]["image_path"])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            evaluate(self.args, model=FakeModel([[perfect_prediction()]]))
        self.assertFalse(self.output.exists())
        partial=self.output.with_suffix(".predictions.jsonl").read_text().splitlines()
        self.assertEqual(len(partial), 1)
        self.assertEqual(json.loads(partial[0])["image_id"], 1)
        self.assertTrue(self.output.with_suffix(".identity.json").exists())

    def test_duplicate_manifest_ids_fail_before_creating_evidence(self):
        self.manifest.write_text((json.dumps(self.rows[0])+"\n")*2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Repeated"):
            evaluate(self.args, model=FakeModel([]))
        self.assertFalse(self.output.with_suffix(".identity.json").exists())

    def test_all_unknown_masks_cannot_reach_development_target(self):
        source=self.rows[0]
        source["annotations"]=[missing_annotation()]
        self.manifest.write_text(json.dumps(source)+"\n", encoding="utf-8")
        report=evaluate(self.args, model=FakeModel([[unknown_prediction()]]))
        self.assertTrue(report["complete"])
        self.assertEqual(report["boxes"]["f1"], 1.0)
        self.assertEqual(report["known_mask_targets"], 0)
        self.assertFalse(report["mask_valid"])
        self.assertFalse(report["development_target_reached"])
        self.assertFalse(report["acceptance_eligible"])
        self.assertEqual(report["mask_f1_lower_bound"], 0.0)

    def test_development_target_uses_conservative_mask_score(self):
        source=self.rows[0]
        source["annotations"].append(missing_annotation())
        self.manifest.write_text(json.dumps(source)+"\n", encoding="utf-8")
        report=evaluate(self.args, model=FakeModel([[perfect_prediction(), unknown_prediction()]]))
        self.assertEqual(report["boxes"]["f1"], 1.0)
        self.assertEqual(report["masks"]["f1"], 1.0)
        self.assertEqual(report["mask_f1_lower_bound"], 0.5)
        self.assertFalse(report["development_target_reached"])


if __name__=="__main__":
    unittest.main()
