"""Verify sampling eligibility, source labels, and private deterministic draws."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.joint_sampling import build_sampling_policy, make_weighted_loader, weights_for_dataset
from training_scripts.verify_joint_bubble_data import sha256


class IndexDataset(Dataset):
    def __init__(self, paths):
        self.im_files=list(paths)

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        return index


class JointSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.masks=[]
        self.boxes=[]
        self.categories=[]
        for name, dataset, targets in (("ai_positive", "ai4va", 2), ("ai_negative", "ai4va", 0),
                                        ("manga_positive", "manga", 4), ("rectangle", "current_boxes", 1),
                                        ("ordinary", "current_boxes", 1)):
            image=self.root/(name+".png")
            image.write_bytes(name.encode())
            row={"image_path":image.as_posix(), "image_sha256":sha256(image), "split":"train",
                 "dataset":dataset, "targets":targets}
            if dataset=="current_boxes":
                label=self.root/(name+".txt")
                label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
                row.update(label_path=label.as_posix(), label_sha256=sha256(label))
                self.boxes.append(row)
                self.categories.append({"image_path":image.as_posix(), "source_sha256":row["image_sha256"],
                                        "width":100, "height":200, "annotations":[
                                            {"category_id":4 if name=="rectangle" else 1, "bbox":[25, 50, 50, 100]}]})
            else:
                self.masks.append(row)
        # These rows point to nonexistent files: no sampling lookup may read them.
        self.masks.extend({"image_path":str(self.root/(split+".png")), "split":split,
                           "dataset":"ai4va", "targets":999} for split in ("validation", "test"))
        self.boxes.append({"image_path":str(self.root/"box_validation.png"), "split":"validation",
                           "dataset":"current_boxes", "targets":999})
        self.source=self.root/"current_train_boxes.jsonl"
        self.write_manifests()

    def write_rows(self, path, rows):
        path.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")

    def write_manifests(self):
        self.write_rows(self.root/"image_label_manifest.jsonl", self.masks)
        self.write_rows(self.root/"current_boxes_image_label_manifest.jsonl", self.boxes)
        self.write_rows(self.source, self.categories)

    def policy(self, ai_weight=3, box_weight=3):
        return build_sampling_policy(self.root, ai_weight, box_weight, self.source, seed=411)

    def test_only_positive_ai_pages_and_actual_rectangle_pages_are_boosted(self):
        policy=self.policy()
        masks={Path(row["image_path"]).stem:row["weight"] for row in policy["groups"]["masks"]["pages"]}
        boxes={Path(row["image_path"]).stem:row["weight"] for row in policy["groups"]["current_boxes"]["pages"]}
        self.assertEqual(masks, {"ai_positive":3.0, "ai_negative":1.0, "manga_positive":1.0})
        self.assertEqual(boxes, {"rectangle":3.0, "ordinary":1.0})
        self.assertEqual(policy["groups"]["masks"]["num_samples"], 3)
        self.assertEqual(policy["source_sha256"][str(self.source.resolve())], sha256(self.source))

    def test_default_policy_disables_sampling_and_does_not_require_category_source(self):
        self.source.unlink()
        policy=self.policy(1, 1)
        self.assertNotIn(str(self.source), policy["source_sha256"])
        for group in policy["groups"].values():
            self.assertFalse(group["enabled"])
            self.assertEqual(group["boosted_pages"], 0)
            self.assertTrue(all(page["weight"]==1 for page in group["pages"]))

    def test_actual_dataset_order_controls_weight_mapping(self):
        group=self.policy()["groups"]["masks"]
        dataset=IndexDataset([str(self.root/(name+".png")) for name in ("manga_positive", "ai_positive", "ai_negative")])
        self.assertEqual(weights_for_dataset(dataset, group), [1.0, 3.0, 1.0])
        dataset.im_files[0]=dataset.im_files[1]
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            weights_for_dataset(dataset, group)

    def test_loader_draws_are_private_seeded_and_preserve_epoch_length(self):
        group=self.policy()["groups"]["masks"]
        dataset=IndexDataset([row["image_path"] for row in group["pages"]])
        state=torch.random.get_rng_state().clone()
        first=make_weighted_loader(dataset, 2, 0, group)
        second=make_weighted_loader(dataset, 2, 0, group)
        self.assertIsInstance(first.sampler, WeightedRandomSampler)
        self.assertTrue(first.sampler.replacement)
        self.assertEqual(first.sampler.num_samples, len(dataset))
        self.assertEqual(len(first), 2)
        for _ in range(3):
            left=[item for batch in first for item in batch.tolist()]
            right=[item for batch in second for item in batch.tolist()]
            self.assertEqual(left, right)
            self.assertEqual(len(left), len(dataset))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        changed=deepcopy(group)
        changed["seed"]+=1
        third=make_weighted_loader(dataset, 2, 0, changed)
        self.assertNotEqual(first.sampler.generator.initial_seed(), third.sampler.generator.initial_seed())

    def test_replacement_draws_repeat_pages_without_duplicating_dataset(self):
        group=self.policy(1e12, 1)["groups"]["masks"]
        dataset=IndexDataset([row["image_path"] for row in group["pages"]])
        loader=make_weighted_loader(dataset, 1, 0, group)
        indices=[batch.item() for batch in loader]
        boosted=next(index for index, row in enumerate(group["pages"]) if row["weight"]>1)
        self.assertEqual(indices, [boosted]*len(dataset))
        self.assertEqual(len(set(dataset.im_files)), 3)

    def test_source_hash_and_label_geometry_mismatches_are_rejected(self):
        self.categories[0]["source_sha256"]="0"*64
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "image hash"):
            self.policy()
        self.categories[0]["source_sha256"]=self.boxes[0]["image_sha256"]
        self.categories[0]["annotations"][0]["bbox"][0]=20
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "one-class box labels"):
            self.policy()

    def test_rectangle_source_requires_exact_training_coverage(self):
        original=deepcopy(self.categories)
        variants=(original[:1], original+[original[0]], original+[
            {**original[0], "image_path":str(self.root/"box_validation.png")}])
        for rows in variants:
            with self.subTest(rows=rows):
                self.write_rows(self.source, rows)
                with self.assertRaisesRegex(ValueError, "coverage differs"):
                    self.policy()

    def test_changed_label_bytes_and_wrong_source_category_are_rejected(self):
        label=Path(self.boxes[0]["label_path"])
        label.write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "label hash"):
            self.policy()
        self.boxes[0]["label_sha256"]=sha256(label)
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "one-class box labels"):
            self.policy()
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        self.boxes[0]["label_sha256"]=sha256(label)
        self.categories[0]["annotations"][0]["category_id"]=0
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "Unknown original"):
            self.policy()

    def test_invalid_weights_and_repeated_training_rows_are_rejected(self):
        for value in (0, 0.5, float("nan"), float("inf")):
            with self.subTest(weight=value), self.assertRaisesRegex(ValueError, "multipliers"):
                self.policy(value, 1)
        self.masks.append(deepcopy(self.masks[0]))
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "repeated training"):
            self.policy()


if __name__=="__main__":
    unittest.main()
