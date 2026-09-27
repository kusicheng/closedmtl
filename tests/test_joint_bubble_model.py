"""Actual YOLO11 CPU loss, gradient, and checkpoint contracts."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from ultralytics import YOLO, __version__
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8DetectionLoss, v8SegmentationLoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.joint_bubble_model import (
    JointBubbleModel, _detection_predictions, _segmentation_slots,
    _slice_predictions, joint_model_from_segment,
)


def model():
    torch.manual_seed(20260927)
    network=SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
    network.args=get_cfg(overrides={"overlap_mask":False})
    network.names={0:"balloon"}
    return network.train()


def batch(count=1, masks=True, seed=73):
    generator=torch.Generator().manual_seed(seed)
    result={"img":torch.rand(count, 3, 96, 96, generator=generator),
            "batch_idx":torch.arange(count, dtype=torch.float32),
            "cls":torch.zeros(count, 1),
            "bboxes":torch.tensor([[0.5, 0.5, 0.5, 0.5]]).repeat(count, 1)}
    if masks:
        result["masks"]=torch.zeros(count, 24, 24)
        result["masks"][:, 6:18, 6:18]=1
    return result


class JointBubbleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_factory_preserves_parameter_objects_and_all_values(self):
        original=model()
        parameters={key:id(value) for key, value in original.named_parameters()}
        before={key:value.clone() for key, value in original.state_dict().items()}
        converted=joint_model_from_segment(original, 0.7)
        self.assertIs(converted, original)
        self.assertEqual(converted.box_loss_weight, 0.7)
        self.assertEqual(parameters, {key:id(value) for key, value in converted.named_parameters()})
        for key, value in converted.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_mask_only_loss_and_gradients_match_native_model_exactly(self):
        original=model()
        joint=joint_model_from_segment(deepcopy(original))
        sample=batch(count=2)
        expected, expected_items=original.loss(sample)
        actual, actual_items=joint.loss(sample)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_items, expected_items))
        expected.sum().backward()
        actual.sum().backward()
        for (name, left), (_, right) in zip(original.named_parameters(), joint.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad), name)

    def test_joint_loss_uses_one_forward_and_native_weighted_loss(self):
        network=joint_model_from_segment(model(), 0.7)
        sample=batch(count=2)
        sample["box_batch"]=batch(count=1, masks=False, seed=79)
        captured=[]
        hook=network.model[-1].register_forward_hook(lambda _, args, output:captured.append(output))
        actual, items=network.loss(sample)
        hook.remove()
        self.assertEqual(len(captured), 1)
        raw=captured[0]
        self.assertEqual(raw["boxes"].shape[0], 3)
        masks=_slice_predictions(raw, 0, 2, 3)
        boxes=_slice_predictions(raw, 2, 3, 3)
        segment_loss, _=v8SegmentationLoss(network)(masks, sample)
        box_loss, _=v8DetectionLoss(network)(_detection_predictions(boxes), sample["box_batch"])
        expected=segment_loss+0.7*_segmentation_slots(box_loss)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(items, actual.detach()/2))
        self.assertEqual(tuple(items.shape), (5,))
        self.assertTrue(torch.isfinite(actual).all())
        actual.sum().backward()
        for group in (network.model[0], network.model[-1].cv2, network.model[-1].cv3,
                      network.model[-1].cv4, network.model[-1].proto):
            gradients=[parameter.grad for parameter in group.parameters() if parameter.requires_grad]
            self.assertTrue(any(value is not None and value.abs().sum()>0 for value in gradients))

    def test_box_objective_has_no_gradient_path_to_mask_parameters(self):
        network=joint_model_from_segment(model())
        mask_batch, box_batch=batch(), batch(masks=False, seed=79)
        raw=network(torch.cat((mask_batch["img"], box_batch["img"])))
        box_predictions=_slice_predictions(raw, 1, 2, 2)
        loss, _=v8DetectionLoss(network)(_detection_predictions(box_predictions), box_batch)
        parameters=[(name, value) for name, value in network.named_parameters() if value.requires_grad]
        gradients=torch.autograd.grad(loss.sum(), [value for _, value in parameters], allow_unused=True)
        head_index=len(network.model)-1
        mask_prefixes=(f"model.{head_index}.cv4.", f"model.{head_index}.proto.")
        mask_count=0
        for (name, _), gradient in zip(parameters, gradients):
            if name.startswith(mask_prefixes):
                mask_count+=1
                self.assertIsNone(gradient, name)
        self.assertGreater(mask_count, 0)
        for prefix in ("model.0.", f"model.{head_index}.cv2.", f"model.{head_index}.cv3."):
            self.assertTrue(any(name.startswith(prefix) and gradient is not None and gradient.abs().sum()>0
                                for (name, _), gradient in zip(parameters, gradients)), prefix)

    def test_checkpoint_round_trip_preserves_inference(self):
        network=joint_model_from_segment(model(), 0.75).eval()
        sample=batch()["img"]
        with torch.no_grad():
            expected=network(sample)[0]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"joint.pt"
            arguments={**vars(network.args), "task":"segment"}
            torch.save({"model":network, "ema":None, "epoch":-1,
                        "train_args":arguments, "version":__version__}, path)
            restored=YOLO(str(path), task="segment").model.float().eval()
            self.assertIsInstance(restored, JointBubbleModel)
            self.assertEqual(restored.box_loss_weight, 0.75)
            with torch.no_grad():
                actual=restored(sample)[0]
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_invalid_mixed_batches_are_rejected_before_forward(self):
        network=joint_model_from_segment(model())
        sample=batch()
        sample["box_batch"]=batch(masks=True)
        with self.assertRaisesRegex(ValueError, "no masks"):
            network.loss(sample)
        sample["box_batch"]=batch(masks=False)
        sample["box_batch"]["img"]=torch.zeros(1, 3, 64, 64)
        with self.assertRaisesRegex(ValueError, "height/width"):
            network.loss(sample)
        sample["box_batch"]["img"]=torch.zeros(1, 3, 96, 96, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "dtype"):
            network.loss(sample)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            joint_model_from_segment(network, float("nan"))

    def test_supplied_predictions_must_cover_both_batches(self):
        network=joint_model_from_segment(model())
        sample=batch()
        sample["box_batch"]=batch(masks=False)
        insufficient=network(sample["img"])
        with self.assertRaisesRegex(ValueError, "combined batch"):
            network.loss(sample, insufficient)


if __name__=="__main__":
    unittest.main()
