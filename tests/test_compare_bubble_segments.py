"""Reject tampered or incomparable evidence and enforce frozen joint quality gates."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from training_scripts.compare_bubble_segments import compare, digest, read_json, read_rows
from training_scripts.evaluate_bubble_segments import aggregate_rows, crop_mask, score_image


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        self.source=self.root/"synthetic_evaluator.py"
        self.source.write_text("# frozen synthetic evaluator source\n")
        self.protocol=self.root/"protocol.json"
        self.manifest=self.root/"manifest.jsonl"
        self.paths={name:self.root/f"{name}.json" for name in ("baseline", "small", "mayocream")}
        self.runs={name:self.root/f"{name}_run.json" for name in ("small", "mayocream")}
        self.records=[]
        annotations=[]
        for index in range(100):
            x=2*index+1
            annotations.append({"id":f"a{index}", "image_id":"one", "category_id":5,
                                "bbox":[x, 1, 1, 1], "area":1, "iscrowd":0,
                                "segmentation":{"synthetic":"geometry supplied directly to score helper"}})
        self.records=[{"image_id":"one", "book":"issue_a", "image_path":"first.png",
                       "source_sha256":"a"*64, "width":202, "height":4,
                       "annotations":annotations, "group_type":"publication_issue"},
                      {"image_id":"two", "book":"issue_b", "image_path":"negative.png",
                       "source_sha256":"b"*64, "width":202, "height":4,
                       "annotations":[], "group_type":"publication_issue"}]
        # Supply valid compressed COCO masks so the real scorer builds journals.
        from pycocotools import mask as coco_mask
        for annotation in annotations:
            mask=np.zeros((4, 202), dtype=np.uint8, order="F")
            mask[1, annotation["bbox"][0]]=1
            rle=coco_mask.encode(mask)
            annotation["segmentation"]={"size":rle["size"], "counts":rle["counts"].decode("ascii")}
        self.write_manifest()
        for name, correct in (("baseline", 50), ("small", 91), ("mayocream", 90)):
            self.make_report(name, correct)

    def write_manifest(self):
        self.manifest.write_text("".join(json.dumps(row)+"\n" for row in self.records), encoding="utf-8")
        write_json(self.protocol, {"manifest":str(self.manifest), "manifest_sha256":digest(self.manifest),
                                  "images":len(self.records), "confidence":0.35, "iou":0.5, "nms_iou":0.5,
                                  "agnostic_nms":True, "imgsz":768, "precision":"FP32",
                                  "mask_resolution":"original_image", "no_tuning_on_external_data":True,
                                  "target":{"box_f1":0.9, "mask_f1_conservative":0.9},
                                  "comparability":{"max_absolute_box_f1_difference":0.01,
                                                   "max_absolute_mask_f1_difference":0.01}})

    def make_report(self, name, correct):
        path=self.paths[name]
        model=self.root/f"{name}.pt"
        model.write_bytes(name.encode())
        settings={"imgsz":768, "device":"cpu", "half":False, "rect":False, "retina_masks":True,
                  "conf":0.35, "iou":0.5, "agnostic_nms":True, "max_det":300, "verbose":False, "save":False}
        ordered=hashlib.sha256("one\ntwo\n".encode()).hexdigest()
        identity={"manifest":{"path":str(self.manifest), "sha256":digest(self.manifest), "images":2,
                               "ordered_image_ids_sha256":ordered},
                  "model":{"path":str(model), "sha256":digest(model), "mayocream":name=="baseline"},
                  "settings":settings, "match_iou":[0.5, 0.75], "limit":None,
                  "source_sha256":{str(self.source):digest(self.source)},
                  "versions":{"synthetic":"1"}, "method":"synthetic frozen fixture",
                  "missing_mask_policy":"explicit unknown masks with conservative bound"}
        identity_path=path.with_suffix(".identity.json")
        write_json(identity_path, identity)
        identity_hash=digest(identity_path)
        predictions=[]
        for index in range(100):
            x=2*index+1
            y=1 if index<correct else 2
            predictions.append({"bbox":[x, y, x+1, y+1], "confidence":0.9,
                                "mask":crop_mask(np.ones((1, 1)), (x, y))})
        rows=[]
        for index, record in enumerate(self.records):
            row=score_image(record, predictions if index==0 else [], (0.5, 0.75))
            row.update({"index":index, "identity_sha256":identity_hash})
            rows.append(row)
        journal=path.with_suffix(".predictions.jsonl")
        journal.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        report=aggregate_rows(rows, (0.5, 0.75))
        groups=defaultdict(list)
        for row in rows:
            groups[row["book"]].append(row)
        report.update({"complete":True, "status":"complete", "target_f1":0.9,
                       "development_target_reached":min(report["boxes"]["f1"], report["mask_f1_lower_bound"])>=0.9,
                       "identity":identity, "identity_sha256":identity_hash,
                       "predictions_jsonl":str(journal), "predictions_sha256":digest(journal),
                       "books":[{"book":book, **aggregate_rows(values, (0.5, 0.75))}
                                for book, values in sorted(groups.items())]})
        write_json(path, report)
        if name in self.runs:
            write_json(self.runs[name], {"track":name, "arguments":{"track":name},
                                        "status":"needs_review", "stages":{"segments":{
                                            "checkpoint":str(model), "sha256":digest(model)}}})

    def comparison(self, output=None):
        return compare(self.paths["baseline"], self.paths["small"], self.paths["mayocream"],
                       self.protocol, output, self.runs["small"], self.runs["mayocream"])

    def rewrite_journal(self, role, rows):
        path=self.paths[role]
        journal=path.with_suffix(".predictions.jsonl")
        journal.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        report=read_json(path)
        report["predictions_sha256"]=digest(journal)
        write_json(path, report)

    def test_complete_reports_recompute_counts_and_accept_inclusive_one_point_difference(self):
        output=self.root/"comparison.json"
        result=self.comparison(output)
        self.assertEqual(result["scores"]["small"]["box_f1"], 0.91)
        self.assertEqual(result["scores"]["mayocream"]["mask_f1_conservative"], 0.9)
        self.assertTrue(result["practically_comparable"])
        self.assertTrue(result["both_retrained_models_pass_and_comparable"])
        self.assertFalse(result["ai4va_subset_target_reached"]["baseline"])
        self.assertFalse(result["architecture_superiority_established"])
        self.assertEqual(result["observed_joint_quality_preference"], "small")
        self.assertAlmostEqual(result["gains_vs_baseline"]["small"]["box_f1"], 0.41)
        self.assertEqual(read_json(output), result)
        with self.assertRaises(FileExistsError):
            self.comparison(output)

    def test_more_than_one_point_is_not_comparable_even_if_both_pass(self):
        self.make_report("small", 92)
        result=self.comparison()
        self.assertFalse(result["practically_comparable"])
        self.assertTrue(result["ai4va_subset_target_reached"]["small"])
        self.assertTrue(result["ai4va_subset_target_reached"]["mayocream"])
        self.assertFalse(result["both_retrained_models_pass_and_comparable"])

    def test_conservative_mask_gate_cannot_be_replaced_by_perfect_known_mask_f1(self):
        for annotation in self.records[0]["annotations"][-11:]:
            annotation["segmentation"]=None
            annotation["mask_status"]="missing_source_polygon"
        self.write_manifest()
        for role in self.paths:
            self.make_report(role, 100)
        result=self.comparison()
        self.assertEqual(result["scores"]["small"]["known_mask_f1"], 1.0)
        self.assertEqual(result["scores"]["small"]["mask_f1_conservative"], 0.89)
        self.assertFalse(result["ai4va_subset_target_reached"]["small"])

    def test_declared_noninferiority_allows_better_mayo_without_claiming_equivalence(self):
        protocol=read_json(self.protocol)
        protocol["mayocream_noninferiority_tolerance"]=0.01
        write_json(self.protocol, protocol)
        self.make_report("mayocream", 96)
        result=self.comparison()
        self.assertFalse(result["practically_comparable"])
        self.assertTrue(result["both_retrained_models_meet_declared_quality_requirement"])
        self.make_report("small", 99)
        self.assertFalse(self.comparison()["both_retrained_models_meet_declared_quality_requirement"])

    def test_noninferiority_cannot_excuse_below_target_or_widen_the_frozen_margin(self):
        protocol=read_json(self.protocol)
        protocol["mayocream_noninferiority_tolerance"]=0.01
        write_json(self.protocol, protocol)
        self.make_report("small", 89)
        result=self.comparison()
        self.assertTrue(result["mayocream_meets_declared_comparison"])
        self.assertFalse(result["both_retrained_models_meet_declared_quality_requirement"])
        protocol["mayocream_noninferiority_tolerance"]=0.05
        write_json(self.protocol, protocol)
        with self.assertRaisesRegex(ValueError, "tolerance must"):
            self.comparison()

    def test_report_aggregate_tampering_is_rejected(self):
        report=read_json(self.paths["small"])
        report["boxes"]["tp"]+=1
        write_json(self.paths["small"], report)
        with self.assertRaisesRegex(ValueError, "report.boxes.tp"):
            self.comparison()

    def test_rehashed_journal_count_tampering_still_fails(self):
        rows=read_rows(self.paths["small"].with_suffix(".predictions.jsonl"))
        rows[0]["counts"]["0.50"]["masks_conservative"]["fn"]+=1
        self.rewrite_journal("small", rows)
        with self.assertRaisesRegex(ValueError, "journal.*masks_conservative.fn"):
            self.comparison()

    def test_rehashed_missing_page_or_wrong_identity_cannot_pass(self):
        original=read_rows(self.paths["small"].with_suffix(".predictions.jsonl"))
        self.rewrite_journal("small", original[:1])
        with self.assertRaisesRegex(ValueError, "cover every"):
            self.comparison()
        original[0]["identity_sha256"]="0"*64
        self.rewrite_journal("small", original)
        with self.assertRaisesRegex(ValueError, "index/identity"):
            self.comparison()

    def test_raw_journal_hash_mismatch_is_rejected(self):
        journal=self.paths["mayocream"].with_suffix(".predictions.jsonl")
        with journal.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "journal hash"):
            self.comparison()

    def test_incomplete_report_and_changed_source_are_rejected(self):
        report=read_json(self.paths["small"])
        report["complete"]=False
        write_json(self.paths["small"], report)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.comparison()
        report["complete"]=True
        write_json(self.paths["small"], report)
        self.source.write_text("# changed after evaluation\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.comparison()

    def test_changed_operating_point_with_fresh_identity_hash_is_rejected(self):
        path=self.paths["small"]
        report=read_json(path)
        report["identity"]["settings"]["imgsz"]=1600
        sidecar=path.with_suffix(".identity.json")
        write_json(sidecar, report["identity"])
        report["identity_sha256"]=digest(sidecar)
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "settings.imgsz"):
            self.comparison()

    def test_duplicate_model_roles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "three distinct model hashes"):
            compare(self.paths["baseline"], self.paths["small"], self.paths["small"], self.protocol)

    def test_swapped_baseline_and_candidate_loaders_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Baseline must use"):
            compare(self.paths["small"], self.paths["baseline"], self.paths["mayocream"], self.protocol)

    def test_swapped_candidate_reports_fail_selected_run_hash_verification(self):
        with self.assertRaisesRegex(ValueError, "Selected run checkpoint hash differs"):
            compare(self.paths["baseline"], self.paths["mayocream"], self.paths["small"],
                    self.protocol, small_run=self.runs["small"], mayocream_run=self.runs["mayocream"])

    def test_swapped_candidate_run_roles_and_incomplete_training_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Run track"):
            compare(self.paths["baseline"], self.paths["small"], self.paths["mayocream"],
                    self.protocol, small_run=self.runs["mayocream"], mayocream_run=self.runs["small"])
        run=read_json(self.runs["small"])
        run["status"]="running"
        write_json(self.runs["small"], run)
        with self.assertRaisesRegex(ValueError, "completed training"):
            self.comparison()

    def test_no_run_metadata_cannot_claim_both_retrained_models_pass(self):
        result=compare(self.paths["baseline"], self.paths["small"], self.paths["mayocream"], self.protocol)
        self.assertTrue(result["both_candidate_models_pass_and_comparable"])
        self.assertFalse(result["both_retrained_models_pass_and_comparable"])
        self.assertFalse(result["training_lineage"]["verified"])


if __name__=="__main__":
    unittest.main()
