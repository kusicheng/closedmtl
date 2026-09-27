"""Check hash binding and exact loader coverage without importing a model runtime."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.verify_joint_bubble_data import REQUIRED_ARTIFACTS, sha256, verify_joint_data


class JointDataVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base=Path(self.temporary.name)
        self.root=base/"composed"
        self.root.mkdir()
        self.external=base/"original"
        self.external.mkdir()
        self.mask_rows=[]
        self.box_rows=[]
        for split in ("train", "validation"):
            for dataset in ("ai4va", "manga", "current_boxes"):
                image=base/"inputs"/dataset/"images"/split/"page.png"
                label=base/"inputs"/dataset/"labels"/split/"page.txt"
                image.parent.mkdir(parents=True)
                label.parent.mkdir(parents=True)
                image.write_bytes(f"synthetic image bytes {dataset} {split}".encode())
                label.write_text("0 0.5 0.5 0.5 0.5\n" if dataset=="current_boxes"
                                 else "0 0.2 0.2 0.8 0.2 0.8 0.8\n", encoding="utf-8")
                row={"image_path":image.as_posix(), "label_path":label.as_posix(),
                     "image_sha256":sha256(image), "label_sha256":sha256(label),
                     "split":split, "dataset":dataset, "targets":1,
                     "label_kind":"box" if dataset=="current_boxes" else "polygon"}
                (self.box_rows if dataset=="current_boxes" else self.mask_rows).append(row)
        self.write_rows("image_label_manifest.jsonl", self.mask_rows)
        self.write_rows("current_boxes_image_label_manifest.jsonl", self.box_rows)
        for split, name in (("train", "train.txt"), ("validation", "val.txt")):
            (self.root/name).write_text("\n".join(row["image_path"] for row in self.mask_rows
                                                if row["split"]==split)+"\n", encoding="utf-8")
            (self.external/f"current_{split}.txt").write_text(
                "\n".join(row["image_path"] for row in self.box_rows if row["split"]==split)+"\n",
                encoding="utf-8")
        self.write_yaml("segments.yaml", {"path":self.root.as_posix(), "train":"train.txt", "val":"val.txt", "names":{0:"balloon"}})
        self.write_yaml("current_boxes.yaml", {"path":self.external.as_posix(),
                        "train":(self.external/"current_train.txt").as_posix(),
                        "val":(self.external/"current_validation.txt").as_posix(), "names":{0:"balloon"}})
        self.write_json("validation_groups.json", {row["image_path"]:row["dataset"]
                                                   for row in self.mask_rows if row["split"]=="validation"})
        self.provenance={"status":"complete", "artifact_sha256":{},
                         "source_hashes":{path.as_posix():sha256(path) for path in self.external.glob("*.txt")}}
        self.refresh_artifacts()

    def write_json(self, name, value):
        (self.root/name).write_text(json.dumps(value), encoding="utf-8")

    def write_yaml(self, name, value):
        (self.root/name).write_text(yaml.safe_dump(value), encoding="utf-8")

    def write_rows(self, name, rows):
        (self.root/name).write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")

    def refresh_artifacts(self):
        self.provenance["artifact_sha256"]={name:sha256(self.root/name) for name in REQUIRED_ARTIFACTS}
        self.write_json("provenance.json", self.provenance)

    def test_complete_snapshot_attests_all_hashes_and_is_read_only(self):
        files=[path for path in Path(self.temporary.name).rglob("*") if path.is_file()]
        before={path:sha256(path) for path in files}
        first=verify_joint_data(self.root)
        second=verify_joint_data(self.root)
        self.assertEqual(first["status"], "verified")
        self.assertEqual(first["manifest_rows"], {"segmentation":4, "current_boxes":2})
        self.assertEqual(first["verified_file_count"], 22)
        self.assertEqual(first["verified_inventory_sha256"], second["verified_inventory_sha256"])
        self.assertEqual(first["effective_lists"]["current_boxes"]["train"]["images"], 1)
        self.assertEqual(first["provenance_sha256"], sha256(self.root/"provenance.json"))
        self.assertEqual(before, {path:sha256(path) for path in Path(self.temporary.name).rglob("*") if path.is_file()})

    def test_every_list_yaml_and_manifest_requires_its_recorded_hash(self):
        for name in REQUIRED_ARTIFACTS:
            path=self.root/name
            original=path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(original+b" ")
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    verify_joint_data(self.root)
                path.write_bytes(original)

    def test_all_mask_and_box_images_and_labels_are_hashed(self):
        for row in self.mask_rows+self.box_rows:
            for key in ("image_path", "label_path"):
                path=Path(row[key])
                original=path.read_bytes()
                with self.subTest(path=path):
                    path.write_bytes(original+b"changed")
                    with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                        verify_joint_data(self.root)
                    path.write_bytes(original)

    def test_original_current_list_hashes_are_checked_outside_composed_root(self):
        path=self.external/"current_train.txt"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            verify_joint_data(self.root)

    def test_hashed_mask_list_still_requires_exact_manifest_coverage(self):
        path=self.root/"train.txt"
        original=path.read_text(encoding="utf-8").splitlines()
        for lines in (original[:1], original+[original[0]], original+[self.mask_rows[-1]["image_path"]]):
            with self.subTest(lines=lines):
                path.write_text("\n".join(lines)+"\n", encoding="utf-8")
                self.refresh_artifacts()
                with self.assertRaisesRegex(ValueError, "coverage differs"):
                    verify_joint_data(self.root)

    def test_hashed_external_current_list_still_requires_exact_coverage(self):
        path=self.external/"current_train.txt"
        extra=next(row["image_path"] for row in self.box_rows if row["split"]=="validation")
        path.write_text(extra+"\n", encoding="utf-8")
        self.provenance["source_hashes"][path.as_posix()]=sha256(path)
        self.refresh_artifacts()
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            verify_joint_data(self.root)

    def test_unrecorded_alternative_current_list_is_rejected(self):
        alternate=self.external/"alternate.txt"
        alternate.write_bytes((self.external/"current_train.txt").read_bytes())
        config=yaml.safe_load((self.root/"current_boxes.yaml").read_text())
        config["train"]=alternate.as_posix()
        self.write_yaml("current_boxes.yaml", config)
        self.refresh_artifacts()
        with self.assertRaisesRegex(ValueError, "lacks recorded provenance"):
            verify_joint_data(self.root)

    def test_test_entry_is_rejected_even_when_empty_and_correctly_hashed(self):
        for name in ("segments.yaml", "current_boxes.yaml"):
            config=yaml.safe_load((self.root/name).read_text())
            self.write_yaml(name, {**config, "test":""})
            self.refresh_artifacts()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "test entry"):
                verify_joint_data(self.root)
            self.write_yaml(name, config)

    def test_validation_group_swapping_cannot_change_source_gate(self):
        groups={row["image_path"]:"manga" for row in self.mask_rows if row["split"]=="validation"}
        self.write_json("validation_groups.json", groups)
        self.refresh_artifacts()
        with self.assertRaisesRegex(ValueError, "group assignments"):
            verify_joint_data(self.root)

    def test_loader_label_path_and_duplicate_manifest_rows_are_rejected(self):
        original=json.loads(json.dumps(self.mask_rows))
        self.mask_rows[0]["label_path"]=self.mask_rows[1]["label_path"]
        self.mask_rows[0]["label_sha256"]=self.mask_rows[1]["label_sha256"]
        self.write_rows("image_label_manifest.jsonl", self.mask_rows)
        self.refresh_artifacts()
        with self.assertRaisesRegex(ValueError, "label path differs"):
            verify_joint_data(self.root)
        self.write_rows("image_label_manifest.jsonl", original+[original[0]])
        self.refresh_artifacts()
        with self.assertRaisesRegex(ValueError, "Repeated image"):
            verify_joint_data(self.root)

    def test_cross_split_duplicate_image_bytes_are_rejected(self):
        first=self.mask_rows[0]
        last=self.mask_rows[-1]
        Path(last["image_path"]).write_bytes(Path(first["image_path"]).read_bytes())
        last["image_sha256"]=first["image_sha256"]
        self.write_rows("image_label_manifest.jsonl", self.mask_rows)
        self.refresh_artifacts()
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            verify_joint_data(self.root)

    def test_missing_provenance_artifact_and_path_escape_are_rejected(self):
        del self.provenance["artifact_sha256"]["segments.yaml"]
        self.write_json("provenance.json", self.provenance)
        with self.assertRaisesRegex(ValueError, "Required composed artifacts"):
            verify_joint_data(self.root)
        self.refresh_artifacts()
        self.provenance["artifact_sha256"]["../outside.json"]="0"*64
        self.write_json("provenance.json", self.provenance)
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify_joint_data(self.root)


if __name__=="__main__":
    unittest.main()
