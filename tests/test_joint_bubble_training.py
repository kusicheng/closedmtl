"""Source-specific stopping gates must not accept pooled or incomplete scores."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.bubble_metrics import count_metrics
from training_scripts.train_joint_bubbles import grouped_gate as source_gate
from training_scripts.train_joint_bubbles import JointTrainer, run
from training_scripts.train_bubble_segment import save_network
from training_scripts.verify_joint_bubble_data import REQUIRED_ARTIFACTS, sha256, verify_joint_data


def page(path, tp=90, fp=10, fn=10):
    return {"image":path, "targets":tp+fn,
            "boxes":count_metrics(tp, fp, fn), "masks":count_metrics(tp, fp, fn)}


def current(tp=965, fp=35, fn=35):
    return {"valid":True, "images":1, "targets":tp+fn,
            "boxes":count_metrics(tp, fp, fn),
            "image_rows":[page("current/calibration.png", tp, fp, fn)]}


def grouped_gate(rows, assignments, current_report, current_paths=("current/calibration.png",)):
    return source_gate(rows, assignments, current_report, current_paths)


class JointTrainingGateTests(unittest.TestCase):
    def setUp(self):
        self.assignments={"calibration/ai.png":"ai4va", "calibration/manga.png":"manga"}
        self.rows=[page(path) for path in self.assignments]

    def test_exact_targets_pass_without_rounding(self):
        result=grouped_gate(self.rows, self.assignments, current())
        self.assertTrue(result["target_reached"])
        self.assertEqual(result["fitness"], 1.0)
        self.assertEqual(result["targets"]["current_box_f1"], 0.965)

    def test_no_source_or_output_can_hide_behind_stronger_pooled_scores(self):
        for index in range(2):
            for kind in ("boxes", "masks"):
                with self.subTest(source=index, kind=kind):
                    rows=[page(path, 10000, 0, 0) for path in self.assignments]
                    rows[index]=page(rows[index]["image"], 100, 0, 0)
                    rows[index][kind]=count_metrics(80, 20, 20)
                    pooled=count_metrics(*(sum(row[kind][name] for row in rows)
                                           for name in ("tp", "fp", "fn")))
                    self.assertGreater(pooled["f1"], 0.99)
                    result=grouped_gate(rows, self.assignments, current(1000, 0, 0))
                    self.assertFalse(result["target_reached"])
                    self.assertEqual(result["fitness"], 0.8/0.9)

    def test_current_boxes_must_pass_even_when_both_mask_sources_are_perfect(self):
        rows=[page(path, 1000, 0, 0) for path in self.assignments]
        result=grouped_gate(rows, self.assignments, current(96, 4, 4))
        self.assertFalse(result["target_reached"])
        self.assertEqual(result["fitness"], 0.96/0.965)

    def test_mask_f1_that_rounds_to_target_still_fails(self):
        counts=count_metrics(449999, 50000, 50000)
        self.assertEqual(round(counts["f1"], 5), 0.9)
        self.rows[0]["masks"]=counts
        self.rows[0]["targets"]=counts["tp"]+counts["fn"]
        result=grouped_gate(self.rows, self.assignments, current())
        self.assertFalse(result["target_reached"])
        self.assertLess(result["fitness"], 1.0)

    def test_current_box_f1_that_rounds_to_target_still_fails(self):
        boxes=current(964999, 35000, 35000)
        self.assertEqual(round(boxes["boxes"]["f1"], 5), 0.965)
        result=grouped_gate(self.rows, self.assignments, boxes)
        self.assertFalse(result["target_reached"])
        self.assertLess(result["fitness"], 1.0)

    def test_source_metrics_recomputed_from_counts_instead_of_cached_f1(self):
        self.rows[0]["masks"]=count_metrics(80, 20, 20)
        self.rows[0]["masks"]["f1"]=1.0
        result=grouped_gate(self.rows, self.assignments, current())
        self.assertEqual(result["sources"]["ai4va"]["masks"]["f1"], 0.8)
        self.assertFalse(result["target_reached"])

    def test_micro_counts_include_verified_negative_pages(self):
        self.assignments["calibration/negative.png"]="ai4va"
        self.rows.append(page("calibration/negative.png", 0, 100, 0))
        result=grouped_gate(self.rows, self.assignments, current())
        score=result["sources"]["ai4va"]
        self.assertEqual(score["images"], 2)
        self.assertEqual(score["targets"], 100)
        self.assertEqual(score["masks"], count_metrics(90, 110, 10))
        self.assertFalse(result["target_reached"])

    def test_missing_positive_or_negative_page_is_rejected(self):
        self.assignments["calibration/negative.png"]="ai4va"
        complete=self.rows+[page("calibration/negative.png", 0, 0, 0)]
        for missing in range(len(complete)):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "every assigned"):
                grouped_gate(complete[:missing]+complete[missing+1:], self.assignments, current())

    def test_repeated_or_unexpected_validation_page_is_rejected(self):
        for extra in (deepcopy(self.rows[0]), page("calibration/unexpected.png")):
            with self.subTest(image=extra["image"]), self.assertRaisesRegex(ValueError, "repeated"):
                grouped_gate(self.rows+[extra], self.assignments, current())

    def test_case_or_relative_alias_cannot_repeat_a_page(self):
        for alias in ("CALIBRATION/AI.PNG", "calibration/../calibration/ai.png"):
            rows=self.rows+[page(alias)]
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, "repeated"):
                grouped_gate(rows, self.assignments, current())
            assignments={**self.assignments, alias:"ai4va"}
            with self.subTest(assignment=alias), self.assertRaisesRegex(ValueError, "unique"):
                grouped_gate(self.rows, assignments, current())

    def test_missing_or_unknown_source_is_rejected(self):
        for assignments in ({self.rows[0]["image"]:"ai4va"},
                            {**self.assignments, "calibration/other.png":"other"}):
            with self.subTest(assignments=assignments), self.assertRaisesRegex(ValueError, "both mask sources"):
                grouped_gate(self.rows, assignments, current())

    def test_empty_source_and_invalid_current_box_report_are_rejected(self):
        empty=[page(self.rows[0]["image"], 0, 0, 0), self.rows[1]]
        with self.assertRaisesRegex(ValueError, "positive targets"):
            grouped_gate(empty, self.assignments, current())
        for report in ({**current(), "valid":False}, current(0, 0, 0)):
            with self.subTest(report=report), self.assertRaisesRegex(ValueError, "Current box"):
                grouped_gate(self.rows, self.assignments, report)

    def test_missing_repeated_or_unexpected_current_page_is_rejected(self):
        for rows in ([], current()["image_rows"]*2, [page("current/unexpected.png")]):
            report={**current(), "image_rows":rows, "images":len(rows)}
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                grouped_gate(self.rows, self.assignments, report)

    def test_current_expected_path_alias_and_report_count_mismatch_are_rejected(self):
        for paths in ([], ["current/calibration.png", "CURRENT/CALIBRATION.PNG"]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                grouped_gate(self.rows, self.assignments, current(), paths)
        report={**current(), "images":2}
        with self.assertRaises(ValueError):
            grouped_gate(self.rows, self.assignments, report)

    def test_gate_does_not_mutate_inputs(self):
        boxes=current()
        before=deepcopy((self.rows, self.assignments, boxes))
        grouped_gate(self.rows, self.assignments, boxes)
        self.assertEqual((self.rows, self.assignments, boxes), before)


class JointTrainerCpuIntegrationTest(unittest.TestCase):
    def test_real_trainer_loaders_loss_final_validation_and_plain_export(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root=Path(directory)/"composed"
            root.mkdir()
            inputs=Path(directory)/"input_data"
            validation=[]
            manifests={"segments":[], "current_boxes":[]}
            source_hashes={}
            original_box_rows=[]
            for dataset_index, dataset in enumerate(("segments", "current_boxes")):
                lists={}
                for split_index, split in enumerate(("train", "val")):
                    images=inputs/dataset/"images"/split
                    labels=inputs/dataset/"labels"/split
                    images.mkdir(parents=True)
                    labels.mkdir(parents=True)
                    paths=[]
                    for index in range(2):
                        image=images/f"page_{index}.png"
                        # Every source/split/page has distinct image bytes.
                        width=96 if split=="val" else 64
                        pixels=np.full((64, width, 3), 180+dataset_index*20+split_index*6+index, dtype=np.uint8)
                        cv2.rectangle(pixels, (width//4, 16), (3*width//4, 48), (30, 30, 30), 2)
                        self.assertTrue(cv2.imwrite(str(image), pixels))
                        label="0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
                        if dataset=="current_boxes":
                            label="0 0.5 0.5 0.5 0.5\n"
                        label_path=labels/f"page_{index}.txt"
                        label_path.write_text(label, encoding="utf-8")
                        paths.append(image.as_posix())
                        manifests[dataset].append({"image_path":image.as_posix(), "label_path":label_path.as_posix(),
                                                   "image_sha256":sha256(image), "label_sha256":sha256(label_path),
                                                   "split":"train" if split=="train" else "validation",
                                                   "dataset":("ai4va", "manga")[index] if dataset=="segments" else "current_boxes",
                                                   "label_kind":"polygon" if dataset=="segments" else "box", "targets":1})
                        if dataset=="current_boxes" and split=="train":
                            original_box_rows.append({"image_path":image.as_posix(), "source_sha256":sha256(image),
                                                      "width":width, "height":64, "annotations":[
                                                          {"category_id":4 if index==0 else 1,
                                                           "bbox":[width/4, 16, width/2, 32]}]})
                        if dataset=="segments" and split=="val":
                            validation.append(image.as_posix())
                    if dataset=="segments":
                        list_path=root/("train.txt" if split=="train" else "val.txt")
                    else:
                        list_path=inputs/dataset/("current_train.txt" if split=="train" else "current_validation.txt")
                    list_path.write_text("\n".join(paths)+"\n", encoding="utf-8")
                    lists[split]=list_path.as_posix()
                    if dataset=="current_boxes":
                        source_hashes[list_path.as_posix()]=sha256(list_path)
                config={"path":(root if dataset=="segments" else inputs/dataset).as_posix(), **lists,
                        "names":{0:"balloon"}, "nc":1}
                (root/f"{dataset}.yaml").write_text(json.dumps(config), encoding="utf-8")
            for dataset, name in (("segments", "image_label_manifest.jsonl"),
                                  ("current_boxes", "current_boxes_image_label_manifest.jsonl")):
                (root/name).write_text("\n".join(json.dumps(row) for row in manifests[dataset])+"\n", encoding="utf-8")
            (root/"validation_groups.json").write_text(
                json.dumps(dict(zip(validation, ("ai4va", "manga")))), encoding="utf-8")
            rectangle_source=inputs/"current_boxes/current_train_boxes.jsonl"
            rectangle_source.write_text("\n".join(json.dumps(row) for row in original_box_rows)+"\n", encoding="utf-8")
            source_hashes[str(rectangle_source)]=sha256(rectangle_source)
            provenance={"status":"complete", "fixture":"synthetic CPU contract test", "source_hashes":source_hashes,
                        "artifact_sha256":{name:sha256(root/name) for name in REQUIRED_ARTIFACTS}}
            (root/"provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            verified_before=verify_joint_data(root)
            self.assertEqual(verified_before["verified_file_count"], 26)
            source=root/"initial.pt"
            network=SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
            network.names={0:"balloon"}
            save_network(network, source, "segment")
            args=SimpleNamespace(track="mayocream", model=str(source), data_root=str(root),
                                 output=str(root/"output"), box_weight=1.0, mask_target=0.9,
                                 box_target=0.965, epochs=1, imgsz=64, batch=1, workers=0,
                                 box_batch=1, box_workers=0, device="cpu", seed=20260927,
                                 lr=0.0001, patience=2, ai4va_positive_weight=3.0, rectangle_weight=3.0,
                                 rectangle_source=str(rectangle_source))
            captured=io.StringIO()
            original_setup=JointTrainer._setup_train

            def check_actual_validation_geometry(trainer):
                original_setup(trainer)
                for loader in (trainer.test_loader, trainer.box_validation_loader):
                    self.assertFalse(loader.dataset.rect)
                    self.assertEqual(tuple(loader.dataset[0]["img"].shape[-2:]), (64, 64))
                for loader in (trainer.train_loader, trainer.box_loader):
                    self.assertIsInstance(loader.sampler, WeightedRandomSampler)
                    self.assertEqual(loader.sampler.num_samples, len(loader.dataset))
                    self.assertEqual(sorted(loader.sampler.weights.tolist()), [1.0, 3.0])

            try:
                with redirect_stdout(captured), redirect_stderr(captured), \
                     patch.object(JointTrainer, "_setup_train", check_actual_validation_geometry):
                    run(args)
            except Exception:
                print(captured.getvalue())
                raise
            result=json.loads((root/"output/run.json").read_text(encoding="utf-8"))
            checkpoint=Path(result["stages"]["segments"]["checkpoint"])
            calibration=result["stages"]["segments"]["calibration"]
            self.assertEqual(calibration["box_batches_seen"], 2)
            self.assertEqual(calibration["current_boxes"]["images"], 2)
            self.assertEqual(set(calibration["sources"]), {"ai4va", "manga"})
            self.assertEqual(calibration["validation_rect"], {"masks":False, "current_boxes":False})
            attestation=result["data_verification"]
            self.assertTrue(attestation["unchanged"])
            for when in ("before", "after"):
                self.assertEqual(attestation[when]["status"], "verified")
                self.assertEqual(attestation[when]["verified_inventory_sha256"], verified_before["verified_inventory_sha256"])
            checker=Path(__file__).resolve().parents[1]/"training_scripts/verify_joint_bubble_data.py"
            self.assertEqual(result["source_sha256"][str(checker)], sha256(checker))
            sampling=result["sampling"]
            policy_path=Path(sampling["policy_path"])
            self.assertEqual(sampling["policy_sha256"], sha256(policy_path))
            self.assertEqual(result["source_sha256"][str(policy_path)], sha256(policy_path))
            self.assertEqual(result["source_sha256"][str(rectangle_source.resolve())], sha256(rectangle_source))
            self.assertEqual(sampling["groups"]["masks"]["boosted_pages"], 1)
            self.assertEqual(sampling["groups"]["current_boxes"]["boosted_pages"], 1)
            restored=YOLO(str(checkpoint)).model.float().eval()
            self.assertIs(type(restored), SegmentationModel)
            with torch.no_grad():
                decoded, raw=restored(torch.zeros(1, 3, 64, 64))
            self.assertEqual(decoded[0].shape[0], 1)
            self.assertEqual(raw["proto"].shape[0], 1)
            verified_after=verify_joint_data(root)
            self.assertEqual(verified_before["verified_inventory_sha256"], verified_after["verified_inventory_sha256"])


if __name__=="__main__":
    unittest.main()
