"""Exercise deterministic replay and exclusion boundaries in joint data composition."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocotools import mask as coco_mask
import yaml

from training_scripts.prepare_bubble_segments import cropped_mask, polygon_from_mask
from training_scripts.prepare_joint_bubble_data import MANGA_POLYGON_PRODUCER, digest, label_path, prepare


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def make_page(path, data, segmentation=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode())
    label=label_path(path)
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9\n" if segmentation else "0 0.5 0.5 0.8 0.8\n", encoding="utf-8")


class JointDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.ai=self.root/"ai/prepared"
        self.ai.mkdir(parents=True)
        self.manga=self.root/"manga"
        self.manga.mkdir()
        self.frozen=self.root/"frozen.json"
        write_json(self.frozen, {"images":[{"issue_id":"old_issue", "sha256":"old_image_hash"}]})
        self.plan={"issue_splits":{split:[f"issue_{split}"] for split in ("train", "calibration", "test")},
                   "issue_to_split":{f"issue_{split}":split for split in ("train", "calibration", "test")}}
        write_json(self.ai/"split_plan.json", self.plan)
        self.ai_pages={}
        for ident, split in enumerate(("train", "calibration", "test"), 1):
            name=f"ai4va_train_{ident:04d}_page_{ident}.png"
            path=self.ai/"segments/images"/split/name
            make_page(path, f"ai4va_{split}")
            row={"image_id":f"ai4va:train:{ident}", "source_image_id":ident, "split":split,
                 "source_split":"train", "issue_id":f"issue_{split}", "source_file_name":f"page_{ident}.png",
                 "image_path":path.as_posix(), "source_sha256":digest(path), "annotations":[{"id":ident}],
                 "polygon_conversion_status":"not_required_for_untouched_test" if split=="test" else "accepted"}
            write_rows(self.ai/f"full_{split}_masks.jsonl", [row])
            self.ai_pages[split]=path
            if split!="test":
                (self.ai/f"segments_{split}.txt").write_text(path.as_posix()+"\n", encoding="utf-8")
        self.ai_prov={"status":"complete", "issue_splits":self.plan["issue_splits"],
                      "split_plan_sha256":digest(self.ai/"split_plan.json"),
                      "source_hashes":{str(self.frozen):digest(self.frozen)}}
        self.refresh_ai()
        self.manga_prov={"status":"complete", "split_books":{split:[f"book_{split}"] for split in ("train", "validation", "test")},
                         "manifest_sha256":{}, "exact_mask_sha256":{}, "current_boxes":{},
                         "source_hashes":{str(MANGA_POLYGON_PRODUCER):digest(MANGA_POLYGON_PRODUCER)},
                         "polygon_quality":{"minimum_required_iou":0.98}}
        mask=np.zeros((10, 10), dtype=np.uint8)
        mask[1:9, 1:9]=1
        rle=coco_mask.encode(np.asfortranarray(mask))
        rle["counts"]=rle["counts"].decode("ascii")
        self.manga_pages={}
        for split in ("train", "validation"):
            rows=[]
            paths=[]
            for ident in range(6):
                path=self.manga/"segments/images"/split/f"manga_{ident}.jpg"
                make_page(path, f"manga_{split}_{ident}")
                paths.append(path)
                annotation={"id":ident, "bbox":[1, 1, 8, 8], "area":64, "segmentation":rle, "iscrowd":0}
                row={"image_id":ident, "book":f"book_{split}", "image_path":path.as_posix(),
                     "source_sha256":digest(path), "width":10, "height":10, "annotations":[annotation]}
                line, _=polygon_from_mask(cropped_mask(annotation, row), annotation["bbox"], row)
                label_path(path).write_text(line+"\n", encoding="utf-8")
                rows.append(row)
            self.manga_pages[split]=paths
            manifest=self.manga/f"{split}_masks.jsonl"
            write_rows(manifest, rows)
            self.manga_prov["exact_mask_sha256"][manifest.name]=digest(manifest)
            file=self.manga/f"segments_{split}.txt"
            file.write_text("".join(path.as_posix()+"\n" for path in paths), encoding="utf-8")
            self.manga_prov["manifest_sha256"][file.name]=digest(file)
            box=self.manga/"boxes/images"/f"current_{split}"/"current.jpg"
            make_page(box, f"current_{split}", False)
            file=self.manga/f"current_{split}.txt"
            file.write_text(box.as_posix()+"\n", encoding="utf-8")
            self.manga_prov["manifest_sha256"][file.name]=digest(file)
            self.manga_prov["current_boxes"][split]={"images":1, "boxes":1}
        config={"path":self.manga.as_posix(), "train":"current_train.txt", "val":"current_validation.txt",
                "test":"current_test.txt", "names":{0:"balloon"}}
        (self.manga/"current_boxes.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        write_json(self.manga/"provenance.json", self.manga_prov)

    def refresh_ai(self):
        self.ai_prov["artifact_sha256"]={path.relative_to(self.ai).as_posix():digest(path)
                                         for path in self.ai.rglob("*") if path.is_file() and path.name!="provenance.json"}
        write_json(self.ai/"provenance.json", self.ai_prov)

    def run_prepare(self, name="out"):
        return prepare(self.ai, self.manga, self.frozen, self.root/name, replay_count=2, seed=20260927)

    def test_deterministic_selection_and_groups_without_test_yaml_entries(self):
        first=self.run_prepare()
        second=self.run_prepare("second")
        self.assertEqual(first["selected_manga_books"], second["selected_manga_books"])
        for name in ("train.txt", "val.txt", "image_label_manifest.jsonl"):
            self.assertEqual((self.root/"out"/name).read_bytes(), (self.root/"second"/name).read_bytes())
        for name in ("segments.yaml", "current_boxes.yaml"):
            config=yaml.safe_load((self.root/"out"/name).read_text())
            self.assertNotIn("test", config)
        groups=json.loads((self.root/"out/validation_groups.json").read_text())
        self.assertEqual(list(groups.values()).count("ai4va"), 1)
        self.assertEqual(list(groups.values()).count("manga"), 2)
        self.assertTrue(all(Path(path).is_absolute() for path in groups))
        self.assertEqual(first["counts"]["train"]["current_boxes"], {"images":1, "targets":1})
        self.assertEqual(first["cross_split_path_and_hash_overlap"], 0)
        self.assertIn("image_label_manifest.jsonl", first["artifact_sha256"])
        self.assertTrue(first["manga_polygon_geometry_verified"])

    def test_missing_label_fails_before_output_creation(self):
        label_path(self.ai_pages["train"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_prepare()
        self.assertFalse((self.root/"out").exists())

    def test_modified_polygon_is_rejected_even_with_same_target_count(self):
        label_path(self.ai_pages["train"]).write_text("0 0.2 0.2 0.8 0.2 0.8 0.8\n")
        with self.assertRaisesRegex(ValueError, "Source hash changed"):
            self.run_prepare()

    def test_changed_manga_geometry_is_rejected_even_with_same_valid_label_count(self):
        for image in self.manga_pages["train"]:
            label_path(image).write_text("0 0.2 0.2 0.8 0.2 0.8 0.8\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Manga polygon geometry differs from audited RLE"):
            self.run_prepare()
        self.assertFalse((self.root/"out").exists())

    def test_changed_original_polygon_producer_is_rejected(self):
        self.manga_prov["source_hashes"][str(MANGA_POLYGON_PRODUCER)]="0"*64
        write_json(self.manga/"provenance.json", self.manga_prov)
        with self.assertRaisesRegex(ValueError, "Source hash changed"):
            self.run_prepare()

    def test_reserved_test_page_in_train_list_is_rejected(self):
        path=self.ai/"segments_train.txt"
        path.write_text(self.ai_pages["test"].as_posix()+"\n", encoding="utf-8")
        self.refresh_ai()
        with self.assertRaisesRegex(ValueError, "list differs"):
            self.run_prepare()

    def test_duplicate_bytes_across_train_and_validation_fail(self):
        path=self.manga/"boxes/images/current_validation/current.jpg"
        path.write_bytes((self.manga/"boxes/images/current_train/current.jpg").read_bytes())
        with self.assertRaisesRegex(ValueError, "Training/validation overlap"):
            self.run_prepare()

    def test_frozen_external_image_hash_is_excluded(self):
        write_json(self.frozen, {"images":[{"issue_id":"old_issue", "sha256":digest(self.ai_pages["train"])}]})
        self.ai_prov["source_hashes"][str(self.frozen)]=digest(self.frozen)
        self.refresh_ai()
        with self.assertRaisesRegex(ValueError, "overlaps frozen evaluation"):
            self.run_prepare()

    def test_source_book_groups_must_be_disjoint(self):
        self.manga_prov["split_books"]["validation"]=["book_train"]
        write_json(self.manga/"provenance.json", self.manga_prov)
        with self.assertRaisesRegex(ValueError, "Overlapping Manga books"):
            self.run_prepare()

    def test_reusing_output_fails(self):
        self.run_prepare()
        with self.assertRaises(FileExistsError):
            self.run_prepare()


if __name__=="__main__":
    unittest.main()
