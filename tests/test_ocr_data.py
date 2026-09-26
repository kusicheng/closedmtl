"""Check split isolation, crop boundaries, and incomplete payload detection."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training_scripts"))
from ocr_data import prepare, read_rows
from ocr_benchmark import edit_distance
from ocr_compare import compare
from ocr_data import digest


class DataTests(unittest.TestCase):
    def fixture(self, root):
        (root/"annotations").mkdir()
        (root/"books.txt").write_text("A\nB\nC\n", encoding="utf-8")
        (root/"readme.txt").write_text("Fixture", encoding="utf-8")
        for book in "ABC":
            folder=root/"images"/book
            folder.mkdir(parents=True)
            Image.new("RGB", (20, 30), "white").save(folder/"000.jpg")
            (root/"annotations"/(book+".xml")).write_text(
                '<book><pages><page index="0" width="20" height="30">'
                '<text id="1" xmin="1" ymin="2" xmax="19" ymax="25">abc</text>'
                '</page><page index="1" width="20" height="30"/></pages></book>', encoding="utf-8")

    def test_splits_and_unlabeled_missing_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root, root/"out")
            rows=[read_rows(root/"out"/(split+".jsonl")) for split in ("train", "validation", "test")]
            self.assertEqual([len(group) for group in rows], [1, 1, 1])
            self.assertEqual(len({group[0]["book"] for group in rows}), 3)
            self.assertEqual(rows[0][0]["box"], [0, 0, 20, 30])
            report=json.loads((root/"out/provenance.json").read_text())
            self.assertEqual(len(report["missing_unlabeled_pages"]), 3)

    def test_missing_labeled_image_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            (root/"images/A/000.jpg").unlink()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                prepare(root, root/"out")

    def test_cer_edit_count(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("same", "same"), 0)

    def test_gate_rejects_partial_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/"test.jsonl"
            manifest.write_text('{"id":"a","text":"a"}\n{"id":"b","text":"b"}\n', encoding="utf-8")
            item={"manifest_sha256":digest(manifest), "samples":1,
                  "sample_ids_sha256":"same", "device":"cpu", "cer":0.05,
                  "exact_match":0.9, "ram_peak_bytes":900000000}
            result=compare(item, item, manifest)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["full_same_manifest"])


if __name__=="__main__":
    unittest.main()
