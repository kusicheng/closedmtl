"""Check global category assignment, invariant geometry bins and evidence rejection."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from training_scripts.analyze_bubble_calibration import analyze, geometry_groups, recompute_box_matches
from training_scripts.compare_bubble_segments import aggregate, digest, metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


class CalibrationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.categories=self.root/"categories.yaml"
        self.categories.write_text(yaml.safe_dump({"names":{0:"Elipse", 1:"cloude", 2:"other", 3:"rectangle",
                                                             4:"sea_uchirin", 5:"thorn"}}), encoding="utf-8")

    def bundle(self, segmentation=False, imgsz=768, name="run"):
        manifest=self.root/("masks.jsonl" if segmentation else "boxes.jsonl")
        records=[]
        rows=[]
        thresholds=[0.5, 0.75] if segmentation else [0.5]
        for page in range(2):
            ident=f"ai4va:train:{page}" if segmentation else str(self.root/f"source{page}.png")
            annotations=[]
            if page==0:
                for index, box in enumerate(([0, 0, 20, 20], [2, 0, 20, 20])):
                    annotations.append({"id":index+1, "image_id":ident if segmentation else 0,
                                        "category_id":5 if segmentation else (1 if index==0 else 6),
                                        "bbox":box, "iscrowd":0,
                                        "segmentation":{"size":[768, 768], "counts":"journal_contract_fixture"} if segmentation else []})
            record={"image_path":str(self.root/f"image{page}.png"), "source_image":str(self.root/f"source{page}.png"),
                    "source_sha256":hashlib.sha256(f"page{page}".encode()).hexdigest(),
                    "width":768, "height":768, "annotations":annotations}
            if segmentation:
                record.update({"image_id":ident, "book":"fixture_issue"})
            records.append(record)
            predictions=[] if page else [{"bbox":[0, 0, 20, 20], "confidence":0.9},
                                         {"bbox":[100, 100, 110, 110], "confidence":0.8}]
            if segmentation:
                for prediction in predictions:
                    prediction.update({"mask_crop_sha256":"a"*64})
            row={"image_id":ident, "image_path":record["image_path"], "source_sha256":record["source_sha256"],
                 "targets":len(annotations), "predictions":predictions, "index":page}
            if segmentation:
                row.update({"book":"fixture_issue", "known_mask_targets":len(annotations), "missing_mask_targets":0,
                            "ignored_mask_predictions":0, "training_polygon_rejected":False, "counts":{}, "matches":{}})
                for threshold in thresholds:
                    pairs=recompute_box_matches(annotations, predictions, threshold)
                    counts=metrics(len(pairs), len(predictions)-len(pairs), len(annotations)-len(pairs))
                    key=f"{threshold:.2f}"
                    row["matches"][key]={"boxes":pairs, "masks":deepcopy(pairs), "ignored_masks":[]}
                    row["counts"][key]={"boxes":counts, "masks":counts, "masks_conservative":counts,
                                         "known_mask_targets":len(annotations), "missing_mask_targets":0,
                                         "ignored_mask_predictions":0, "mask_f1_lower_bound":counts["f1"]}
                row["mask_f1_lower_bound"]=row["counts"]["0.50"]["masks"]["f1"]
            else:
                pairs=recompute_box_matches(annotations, predictions, 0.5)
                row.update({"matches":pairs, "boxes":metrics(len(pairs), len(predictions)-len(pairs), len(annotations)-len(pairs))})
            rows.append(row)
        write_rows(manifest, records)
        report_path=self.root/f"{name}.json"
        journal=report_path.with_suffix(".predictions.jsonl")
        identity_path=report_path.with_suffix(".identity.json")
        ids=[str(record["image_id"] if segmentation else record["source_image"]) for record in records]
        order_key="ordered_image_ids_sha256" if segmentation else "ordered_source_images_sha256"
        identity={"manifest":{"path":str(manifest), "sha256":digest(manifest), "images":len(records),
                                order_key:hashlib.sha256("".join(ident+"\n" for ident in ids).encode()).hexdigest()},
                  "model":{"path":str(self.root/"model.pt"), "sha256":"b"*64}, "limit":None,
                  "match_iou":thresholds if segmentation else 0.5,
                  "settings":{"imgsz":imgsz, "conf":0.35, "iou":0.5, "half":False, "rect":False,
                              "retina_masks":True, "agnostic_nms":True, "max_det":300}}
        write_json(identity_path, identity)
        for row in rows:
            row["identity_sha256"]=digest(identity_path)
        write_rows(journal, rows)
        totals=aggregate(rows, thresholds) if segmentation else {"images":2, "targets":2, "boxes":metrics(1, 1, 1)}
        report={**totals, "complete":True, "status":"complete", "identity":identity,
                "identity_sha256":digest(identity_path), "predictions_jsonl":str(journal), "predictions_sha256":digest(journal)}
        write_json(report_path, report)
        return report_path, manifest, journal, rows

    def run_analysis(self, report, manifest):
        return analyze(report, manifest, category_yaml=self.categories)

    def reseal_journal(self, report, journal, rows):
        write_rows(journal, rows)
        data=json.loads(report.read_text())
        data["predictions_sha256"]=digest(journal)
        write_json(report, data)

    def test_global_matching_does_not_award_same_prediction_to_two_categories(self):
        report, manifest, _, _=self.bundle()
        result=self.run_analysis(report, manifest)
        boxes=result["by_iou"]["0.50"]["boxes"]
        categories=boxes["by_group"]["category"]
        self.assertEqual(categories["groups"]["1 Elipse"]["recall"], 1)
        self.assertEqual(categories["groups"]["6 thorn"]["recall"], 0)
        self.assertEqual(categories["macro_recall"], 0.5)
        self.assertIsNone(categories["groups"]["2 cloude"]["recall"])
        self.assertEqual(boxes["global"]["unmatched_false_positives"], 1)
        self.assertNotIn("f1", categories["groups"]["1 Elipse"])
        self.assertEqual(list(result["by_iou"]), ["0.50"])

    def test_geometry_memberships_are_identical_at_640_and_1600(self):
        left, manifest, _, _=self.bundle(imgsz=640, name="small")
        right, _, _, _=self.bundle(imgsz=1600, name="large")
        first, second=self.run_analysis(left, manifest), self.run_analysis(right, manifest)
        self.assertEqual(first["membership_sha256"], second["membership_sha256"])
        self.assertEqual(first["by_iou"], second["by_iou"])
        self.assertNotEqual(first["inference_imgsz"], second["inference_imgsz"])

    def test_fixed_size_aspect_and_page_boundaries(self):
        names={1:"bubble"}
        for short, expected in ((15.999, "lt16"), (16, "16_to_lt24"), (24, "24_to_lt32"), (32, "ge32")):
            ann={"category_id":1, "bbox":[0, 0, short, 40]}
            self.assertEqual(geometry_groups(ann, {"width":768, "height":768}, names)["short_side_at_768"], expected)
        for width, expected in ((9, "lt1"), (10, "1_to_3"), (30, "1_to_3"), (31, "gt3")):
            ann={"category_id":1, "bbox":[0, 0, width, 10]}
            self.assertEqual(geometry_groups(ann, {"width":768, "height":768}, names)["width_height_ratio"], expected)
        for side, expected in ((1024, "le1024"), (1025, "1025_to_2048"), (2048, "1025_to_2048"), (2049, "gt2048")):
            ann={"category_id":1, "bbox":[0, 0, 20, 20]}
            self.assertEqual(geometry_groups(ann, {"width":side, "height":768}, names)["source_page_long_side"], expected)

    def test_segmentation_has_published_class_and_only_attested_iou_thresholds(self):
        report, manifest, _, _=self.bundle(segmentation=True)
        result=self.run_analysis(report, manifest)
        self.assertEqual(set(result["by_iou"]), {"0.50", "0.75"})
        groups=result["by_iou"]["0.75"]["masks"]["by_group"]["category"]["groups"]
        self.assertEqual(list(groups), ["26 Comic Bubble"])
        self.assertEqual(groups["26 Comic Bubble"]["support"], 2)
        self.assertEqual(groups["26 Comic Bubble"]["mean_matched_iou"], 1)

    def test_manifest_and_journal_hash_changes_fail(self):
        report, manifest, journal, _=self.bundle()
        original=journal.read_bytes()
        journal.write_bytes(original+b"\n")
        with self.assertRaisesRegex(ValueError, "journal hash mismatch"):
            self.run_analysis(report, manifest)
        journal.write_bytes(original)
        manifest.write_bytes(manifest.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "Manifest hash mismatch"):
            self.run_analysis(report, manifest)

    def test_reordered_rows_fail_even_after_journal_hash_is_updated(self):
        report, manifest, journal, rows=self.bundle()
        self.reseal_journal(report, journal, rows[::-1])
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            self.run_analysis(report, manifest)

    def test_non_global_reassignment_fails_despite_one_to_one_pairs(self):
        report, manifest, journal, rows=self.bundle()
        pair=rows[0]["matches"][0]
        pair.update({"target_index":1, "annotation_id":2, "iou":360/440})
        self.reseal_journal(report, journal, rows)
        with self.assertRaisesRegex(ValueError, "global_box_match"):
            self.run_analysis(report, manifest)

    def test_duplicate_mask_assignment_is_rejected(self):
        report, manifest, journal, rows=self.bundle(segmentation=True)
        rows[0]["matches"]["0.50"]["masks"].append({"target_index":1, "annotation_id":2, "prediction_index":0, "iou":0.8})
        self.reseal_journal(report, journal, rows)
        with self.assertRaisesRegex(ValueError, "Repeated instance match"):
            self.run_analysis(report, manifest)

    def test_report_aggregate_mismatch_and_partial_evaluation_fail(self):
        report, manifest, _, _=self.bundle()
        data=json.loads(report.read_text())
        data["boxes"]["tp"]=2
        write_json(report, data)
        with self.assertRaisesRegex(ValueError, "Mismatch report.boxes.tp"):
            self.run_analysis(report, manifest)
        data["complete"]=False
        write_json(report, data)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.run_analysis(report, manifest)


if __name__=="__main__":
    unittest.main()
