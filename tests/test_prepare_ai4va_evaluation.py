"""Check AI4VA source coverage, canonical rasterization and identity preservation."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from training_scripts.prepare_ai4va_evaluation import convert_annotation, digest, prepare
from training_scripts.prepare_bubble_segments import cropped_mask


def bubble(ident, image_id, missing=False):
    return {"id":ident, "image_id":image_id, "category_id":26,
            "bbox":[0, 0, 9, 7], "area":63, "iscrowd":0,
            "segmentation":[] if missing else [[1, 1, 5, 1, 5, 4, 1, 4],
                                               [3, 2, 7, 2, 7, 5, 3, 5]]}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CanonicalPolygonTests(unittest.TestCase):
    def test_multiple_polygons_use_union_and_exact_evaluator_decodes_tight_crop(self):
        source=bubble(7, 1)
        result=convert_annotation(source, {"width":32, "height":24}, "val")
        expected=np.zeros((24, 32), dtype=np.uint8)
        expected[1:4, 1:5]=1
        expected[2:5, 3:7]=1
        decoded=coco_mask.decode(result["segmentation"])
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(result["area"], 20)
        self.assertEqual(result["bbox"], [1, 1, 6, 4])
        self.assertEqual(result["source_bbox"], source["bbox"])
        self.assertEqual(result["source_area"], 63)
        np.testing.assert_array_equal(cropped_mask(result, {"width":32, "height":24}),
                                      expected[1:5, 1:7])

    def test_missing_polygon_keeps_real_source_box_without_fabricating_mask(self):
        source=bubble(27364, 184, missing=True)
        result=convert_annotation(source, {"width":32, "height":24}, "val")
        self.assertIsNone(result["segmentation"])
        self.assertEqual(result["mask_status"], "missing_source_polygon")
        self.assertEqual(result["bbox"], source["bbox"])
        self.assertEqual(result["category_id"], 5)
        self.assertEqual(result["id"], "ai4va:val:annotation:27364")
        self.assertEqual(result["image_id"], "ai4va:val:184")

    def test_bad_coordinates_and_empty_raster_fail(self):
        polygons=([1, 1, float("nan"), 1, 4, 3],
                  [-1, 1, 4, 1, 4, 3], [1, 1, 33, 1, 4, 3],
                  [1, 1, 1.1, 1, 1.1, 1.1])
        for polygon in polygons:
            with self.subTest(polygon=polygon):
                source=bubble(7, 1)
                source["segmentation"]=[polygon]
                with self.assertRaises(ValueError):
                    convert_annotation(source, {"width":32, "height":24}, "val")


class ManifestCoverageTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        (self.root/"annotations").mkdir()
        (self.root/"images").mkdir()
        categories=[{"id":26, "name":"Comic Bubble"}, {"id":1, "name":"Frame"}]
        self.sources={
            "val":{"images":[], "annotations":[bubble(27364, 184, True), bubble(27365, 184),
                                                    bubble(7, 1), {"id":8, "image_id":2, "category_id":1}],
                   "categories":categories},
            "test":{"images":[], "annotations":[bubble(7, 1)], "categories":categories},
        }
        self.mapping={"status":"complete", "images":[]}
        for split, ids in (("val", (184, 1, 2)), ("test", (1, 94))):
            for ident in ids:
                filename=f"{split}_{ident}.png"
                self.sources[split]["images"].append({"id":ident, "file_name":filename,
                                                      "width":32, "height":24})
                path=self.root/"images"/filename
                missing=ident==94
                if not missing:
                    Image.new("RGB", (32, 24), (ident, 0 if split=="val" else 99, 30)).save(path)
                self.mapping["images"].append({
                    "source_split":split, "image_id":ident, "source_file_name":filename,
                    "organized_path":None if missing else str(path),
                    "sha256":None if missing else digest(path),
                    "width":32, "height":24, "issue_id":f"issue_{split}",
                    "publication_date":"1954-01-01", "page_number":str(ident),
                    "availability":"missing_from_official_archive" if missing else "verified",
                })
        self.write_sources()

    def write_sources(self):
        for split, data in self.sources.items():
            (self.root/"annotations"/f"source_{split}.json").write_text(json.dumps(data), encoding="utf-8")
        (self.root/"mapping_manifest.json").write_text(json.dumps(self.mapping), encoding="utf-8")

    def test_retains_missing_mask_page_and_distinguishes_supplied_negatives_from_unlabeled(self):
        source_hashes={split:digest(self.root/"annotations"/f"source_{split}.json") for split in self.sources}
        report=prepare(self.root)
        self.assertEqual(report["counts"]["validation"]["available_annotated_pages"], 3)
        self.assertEqual(report["counts"]["validation"]["bubble_boxes"], 3)
        self.assertEqual(report["counts"]["validation"]["known_bubble_masks"], 2)
        self.assertEqual(report["counts"]["validation"]["missing_bubble_masks"], 1)
        self.assertEqual(report["counts"]["validation"]["supplied_negative_pages"], 1)
        self.assertEqual(report["exclusions"][0]["source_image_id"], 94)
        rows=read_rows(self.root/"evaluation/validation_joint_masks.jsonl")
        missing_page=next(row for row in rows if row["source_image_id"]==184)
        self.assertEqual(len(missing_page["annotations"]), 2)
        self.assertEqual(sum(ann["mask_status"]=="known" for ann in missing_page["annotations"]), 1)
        negative=next(row for row in rows if row["source_image_id"]==2)
        self.assertEqual(negative["annotations"], [])
        self.assertEqual(negative["annotation_status"], "supplied_negative_other_classes_annotated")
        test=read_rows(self.root/"evaluation/test_joint_masks.jsonl")
        self.assertEqual([row["image_id"] for row in test], ["ai4va:test:1"])
        all_ids=[row["image_id"] for row in rows+test]
        self.assertEqual(len(set(all_ids)), len(all_ids))
        boxes=read_rows(self.root/"evaluation/validation_source_boxes.jsonl")
        self.assertTrue(all(ann["bbox"]==ann["source_bbox"] and "segmentation" not in ann
                            for row in boxes for ann in row["annotations"]))
        self.assertEqual(missing_page["group_type"], "publication_issue")
        self.assertIsNone(missing_page["story_series"])
        self.assertFalse(report["acceptance_eligible"])
        for split, expected in source_hashes.items():
            self.assertEqual(digest(self.root/"annotations"/f"source_{split}.json"), expected)
        with self.assertRaises(FileExistsError):
            prepare(self.root)

    def test_cross_split_identical_image_bytes_are_rejected_before_output_creation(self):
        val=next(row for row in self.mapping["images"] if row["source_split"]=="val" and row["image_id"]==1)
        test=next(row for row in self.mapping["images"] if row["source_split"]=="test" and row["image_id"]==1)
        Path(test["organized_path"]).write_bytes(Path(val["organized_path"]).read_bytes())
        test["sha256"]=val["sha256"]
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            prepare(self.root)
        self.assertFalse((self.root/"evaluation").exists())

    def test_unknown_annotation_image_and_duplicate_ids_are_rejected(self):
        original=deepcopy(self.sources)
        self.sources["val"]["annotations"][0]["image_id"]=999
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "unknown image"):
            prepare(self.root)
        self.sources=original
        self.sources["val"]["annotations"].append(deepcopy(self.sources["val"]["annotations"][0]))
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
            prepare(self.root)

    def test_hash_and_actual_dimension_mismatch_are_rejected(self):
        row=self.mapping["images"][0]
        Image.new("RGB", (31, 24)).save(row["organized_path"])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare(self.root)
        row["sha256"]=digest(row["organized_path"])
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Actual image dimensions"):
            prepare(self.root)


if __name__=="__main__":
    unittest.main()
