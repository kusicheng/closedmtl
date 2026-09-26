"""Reject fabricated, incomplete, or internally inconsistent acceptance reports."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training_scripts"))
from ocr_compare import compare
from ocr_data import digest
from ocr_text import post_process


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/"test.jsonl"
        rows=[{"id":"a", "text":"abc"}, {"id":"b", "text":"xyz"}]
        self.path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        self.report={"samples":2, "manifest_sha256":digest(self.path),
                     "sample_ids_sha256":hashlib.sha256(b"a\nb").hexdigest(),
                     "device":"cpu", "ram_peak_bytes":900000000,
                     "ram_peak_method":"windows_peak_working_set", "cer":0.0, "exact_match":1.0,
                     "predictions":[{"id":row["id"], "target":post_process(row["text"]),
                                     "prediction":post_process(row["text"]), "edits":0} for row in rows]}

    def test_valid_complete_evidence_passes(self):
        self.assertTrue(compare(self.report, self.report, self.path)["passed"])

    def test_wrong_predictions_and_aggregates_fail(self):
        for name in ("truncated", "ids", "targets", "edits", "hash", "cer", "exact", "nan"):
            with self.subTest(name=name):
                broken=deepcopy(self.report)
                if name=="truncated":
                    broken["predictions"].pop()
                elif name=="ids":
                    broken["predictions"][0]["id"]="wrong"
                elif name=="targets":
                    broken["predictions"][0]["target"]="changed"
                elif name=="edits":
                    broken["predictions"][0]["edits"]=100
                elif name=="hash":
                    broken["sample_ids_sha256"]="WRONG"
                elif name=="cer":
                    broken["cer"]=0.1
                elif name=="exact":
                    broken["exact_match"]=0.9
                else:
                    broken["cer"]=float("nan")
                self.assertFalse(compare(broken, broken, self.path)["passed"])

    def test_invalid_memory_fails(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                broken={**self.report, "ram_peak_bytes":value}
                self.assertFalse(compare(broken, broken, self.path)["passed"])
        broken={**self.report, "ram_peak_method":"final_rss"}
        self.assertFalse(compare(broken, broken, self.path)["passed"])

    def test_empty_manifest_fails(self):
        self.path.write_text("", encoding="utf-8")
        self.assertFalse(compare(self.report, self.report, self.path)["passed"])

    def test_gpu_teacher_allows_absolute_cpu_ram_but_not_cross_device_ratio(self):
        teacher={**self.report, "device":"cuda", "ram_peak_bytes":4000000000}
        self.assertTrue(compare(teacher, self.report, self.path)["passed"])
        student={**self.report, "ram_peak_bytes":1100000000}
        self.assertFalse(compare(teacher, student, self.path)["passed"])
        self.assertFalse(compare(self.report, {**self.report, "device":"cuda"}, self.path)["passed"])


if __name__=="__main__":
    unittest.main()
