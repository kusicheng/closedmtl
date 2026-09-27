"""Check annotation-only issue isolation, exclusions and complete exact holdouts."""

from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

from PIL import Image

from training_scripts import organize_ai4va_additional as organizer
from training_scripts.prepare_ai4va_adaptation import assign_issues, inspect_sources, prepare
from training_scripts.prepare_ai4va_evaluation import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def accepted_polygon(mask, bbox, image, minimum_iou):
    return "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8", {"iou":1.0}


class SplitTests(unittest.TestCase):
    def test_seeded_stratification_keeps_issues_disjoint_and_holds_required_masks(self):
        stats={f"issue_{i:03d}":{"pages":2, "masks":0 if i<9 else 20, "negative_pages":1}
               for i in range(76)}
        result=assign_issues(stats)
        self.assertEqual(result, assign_issues(dict(reversed(list(stats.items())))))
        self.assertEqual([result["counts"][split]["issues"] for split in ("train", "calibration", "test")], [46, 15, 15])
        self.assertEqual(Counter(result["issue_to_split"].values()), {"train":46, "calibration":15, "test":15})
        self.assertEqual(len(result["issue_to_split"]), 76)
        for split in ("calibration", "test"):
            self.assertGreaterEqual(result["counts"][split]["masks"], 200)
            self.assertEqual(sum(stats[i]["masks"]==0 for i in result["issue_splits"][split]), 2)

    def test_insufficient_capacity_fails_without_lowering_heldout_requirement(self):
        stats={str(i):{"pages":1, "masks":1, "negative_pages":0} for i in range(76)}
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            assign_issues(stats)


class OrganizedFixture(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        self.dataset=self.root/"additional"
        self.dataset.mkdir()
        self.frozen_path=self.root/"frozen_mapping.json"
        write_json(self.frozen_path, {"images":[{"issue_id":"frozen_issue", "sha256":"reserved"}]})
        self.source={"images":[], "annotations":[], "categories":[{"id":26, "name":"Comic Bubble"}, {"id":1, "name":"Frame"}]}
        self.mapping={"status":"complete", "images":[]}
        self.audit={"source_path":str(self.root/"official_train.json"), "source_sha256":None,
                    "frozen_mapping_path":str(self.frozen_path), "frozen_mapping_sha256":digest(self.frozen_path), "pages":[]}
        for ident in (1, 2, 3, 4, 5, 6, 154, 272):
            issue_number=1 if ident==154 else 2 if ident==272 else ident
            issue=f"vaillant_{issue_number:04d}_1954_01_01"
            filename=f"Vaillant_{issue_number:04d}_1954_01_01-{ident:02d}.png"
            path=self.dataset/"images"/issue/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (ident%255, ident//255, 99)).save(path)
            self.source["images"].append({"id":ident, "file_name":filename, "width":64, "height":48})
            annotation={"id":ident, "image_id":ident, "category_id":1 if ident==6 else 26,
                        "bbox":[5, 5, 45, 35], "area":1575, "iscrowd":int(ident==272),
                        "segmentation":[] if ident==154 else [[5, 5, 50, 5, 50, 40, 5, 40]]}
            self.source["annotations"].append(annotation)
            self.mapping["images"].append({"source_split":"train", "image_id":ident, "source_file_name":filename,
                                          "issue_id":issue, "publication_date":"1954-01-01", "page_number":str(ident),
                                          "organized_path":str(path), "sha256":digest(path), "width":64, "height":48})
            self.audit["pages"].append({"image_id":ident, "file_name":filename, "issue_id":issue,
                                        "overlaps_frozen_evaluation_issue":False, "annotation_count":1,
                                        "bubble_count":int(ident!=6), "known_nonempty_masks":int(ident not in (6, 154))})
        self.audit_path=self.root/"audit.json"
        self.write_inputs()

    def write_inputs(self):
        write_json(self.dataset/"annotations/source_train.json", self.source)
        write_json(Path(self.audit["source_path"]), self.source)
        self.audit["source_sha256"]=digest(Path(self.audit["source_path"]))
        write_json(self.dataset/"mapping_manifest.json", self.mapping)
        write_json(self.audit_path, self.audit)

    def test_excludes_missing_and_crowd_pages_before_split_and_retains_negative(self):
        rows, exclusions, stats, hashes=inspect_sources(self.dataset, self.audit_path)
        self.assertEqual({row["source_image_id"] for row in exclusions}, {154, 272})
        self.assertEqual(sum(row["masks"] for row in stats.values()), 5)
        self.assertEqual(sum(row["pages"] for row in stats.values()), 6)
        self.assertEqual(sum(row["negative_pages"] for row in stats.values()), 1)
        self.assertEqual(next(row for row in rows if row["source_image_id"]==6)["annotations"], [])
        self.assertTrue(all(Path(row["image_path"]).is_file() for row in exclusions))

    def test_frozen_issue_and_hash_overlap_fail(self):
        write_json(self.frozen_path, {"images":[{"issue_id":self.mapping["images"][0]["issue_id"], "sha256":"other"}]})
        self.audit["frozen_mapping_sha256"]=digest(self.frozen_path)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "overlaps frozen"):
            inspect_sources(self.dataset, self.audit_path)
        write_json(self.frozen_path, {"images":[{"issue_id":"frozen_issue", "sha256":self.mapping["images"][0]["sha256"]}]})
        self.audit["frozen_mapping_sha256"]=digest(self.frozen_path)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "bytes overlap"):
            inspect_sources(self.dataset, self.audit_path)

    def test_missing_source_annotations_do_not_become_negative_and_hashes_are_enforced(self):
        self.source["annotations"]=[ann for ann in self.source["annotations"] if ann["image_id"]!=6]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Unannotated page"):
            inspect_sources(self.dataset, self.audit_path)
        self.mapping["images"][0]["sha256"]="wrong"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            inspect_sources(self.dataset, self.audit_path)

    def test_absent_unannotated_source_eval_page_does_not_reserve_an_unused_issue(self):
        frozen=json.loads(self.frozen_path.read_text())
        frozen["images"].append({"issue_id":self.mapping["images"][0]["issue_id"], "sha256":None,
                                 "availability":"missing_from_official_archive", "organized_path":None})
        write_json(self.frozen_path, frozen)
        self.audit["frozen_mapping_sha256"]=digest(self.frozen_path)
        self.write_inputs()
        rows, _, stats, _=inspect_sources(self.dataset, self.audit_path)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(stats), 6)

    def test_duplicate_image_bytes_cannot_cross_frozen_adaptation_splits(self):
        _, _, stats, _=inspect_sources(self.dataset, self.audit_path)
        plan=assign_issues(stats, minimum_masks=1)
        first=next(row for row in self.mapping["images"] if row["issue_id"]==plan["issue_splits"]["train"][0])
        second=next(row for row in self.mapping["images"] if row["issue_id"]==plan["issue_splits"]["test"][0])
        Path(second["organized_path"]).write_bytes(Path(first["organized_path"]).read_bytes())
        second["sha256"]=first["sha256"]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "cross prospective split"):
            prepare(self.dataset, self.audit_path, minimum_masks=1)
        self.assertFalse((self.dataset/"prepared").exists())

    def test_freezes_before_polygon_rejection_and_retains_complete_exact_test(self):
        rows, _, stats, _=inspect_sources(self.dataset, self.audit_path)
        expected=assign_issues(stats, minimum_masks=1)
        reject_issue=expected["issue_splits"]["calibration"][0]
        observed=[]
        def converter(mask, bbox, image, minimum_iou):
            observed.append(image["split"])
            self.assertTrue((self.dataset/"prepared/split_plan.json").is_file())
            if image["issue_id"]==reject_issue:
                return None, {"iou":0.8}
            return accepted_polygon(mask, bbox, image, minimum_iou)
        report=prepare(self.dataset, self.audit_path, minimum_masks=1, polygon_converter=converter)
        self.assertEqual(report["issue_splits"], expected["issue_splits"])
        self.assertNotIn("test", observed)
        exact=[row for split in ("train", "calibration", "test")
               for row in read_rows(self.dataset/f"prepared/full_{split}_masks.jsonl")]
        self.assertEqual({row["source_image_id"] for row in exact}, {1, 2, 3, 4, 5, 6})
        self.assertEqual(len(report["polygon_rejected_pages"]), 1)
        self.assertTrue(report["polygon_rejected_pages"][0]["exact_manifest_retained"])
        self.assertEqual(report["counts"]["calibration"]["exact_masks"], 1)
        self.assertEqual(report["counts"]["calibration"]["yolo_masks"], 0)
        self.assertFalse(report["yolo_calibration_has_positive_masks"])
        self.assertFalse((self.dataset/"prepared/segments_test.txt").exists())
        self.assertNotIn("test:", (self.dataset/"prepared/segments.yaml").read_text())
        with self.assertRaises(FileExistsError):
            prepare(self.dataset, self.audit_path, minimum_masks=1)

    def test_organizer_selects_only_audited_nonoverlap_and_keeps_bytes_names(self):
        archive_path=self.dataset/"_downloads/train-images-comics.zip"
        archive_path.parent.mkdir()
        with zipfile.ZipFile(archive_path, "w") as archive:
            for row in self.mapping["images"]:
                archive.write(row["organized_path"], "train/"+row["source_file_name"])
            archive.writestr("train/not_selected.png", b"ignored")
            archive.writestr("__MACOSX/train/._unused.png", b"ignored")
        expected_hash=digest(archive_path)
        expected_bytes=archive_path.stat().st_size
        write_json(Path(str(archive_path)+".source.json"), {"bytes":expected_bytes, "sha256":expected_hash,
                   "public_url":"https://example.invalid/fixture", "file_id":"test_fixture"})
        (self.dataset/"mapping_manifest.json").unlink()
        with patch.object(organizer, "ARCHIVE_SHA256", expected_hash), patch.object(organizer, "ARCHIVE_BYTES", expected_bytes):
            result=organizer.organize(self.dataset, self.audit_path)
        self.assertEqual(result["counts"], {"images":8, "issues":6, "needs_annotation_review":2})
        self.assertEqual({row["sha256"] for row in result["images"]}, {row["sha256"] for row in self.mapping["images"]})
        self.assertTrue(all(Path(row["organized_path"]).name==row["source_file_name"] for row in result["images"]))
        self.assertFalse((self.dataset/"images/not_selected.png").exists())


class ArchiveSafetyTests(unittest.TestCase):
    def test_unsafe_paths_duplicate_members_and_symlinks_fail(self):
        for name in ("../page.png", "/page.png", "C:/page.png", "train\\page.png", "train/../page.png"):
            with self.subTest(name=name):
                archive=SimpleNamespace(infolist=lambda:[SimpleNamespace(filename=name, external_attr=0)])
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    organizer.safe_members(archive)
        with BytesIO() as data:
            entry=zipfile.ZipInfo("train/page.png")
            entry.external_attr=(stat.S_IFLNK|0o777)<<16
            with zipfile.ZipFile(data, "w") as archive:
                archive.writestr(entry, b"outside")
            with zipfile.ZipFile(data) as archive, self.assertRaisesRegex(ValueError, "Unsafe"):
                organizer.safe_members(archive)
        member=SimpleNamespace(filename="train/page.png", external_attr=0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            organizer.safe_members(SimpleNamespace(infolist=lambda:[member, member]))


if __name__=="__main__":
    unittest.main()
