"""Check that rounded library metrics cannot satisfy the training target."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_scripts.bubble_metrics import count_metrics
from training_scripts.train_bubble_segment import FixedTrainerMixin


class FakeValidator:
    def __init__(self, boxes, masks=None, valid=True):
        self.fixed_report={"valid":valid, "boxes":boxes}
        if masks is not None:
            self.fixed_report["masks"]=masks

    def __call__(self, trainer):
        result={"metrics/fixed_f1(B)":round(self.fixed_report["boxes"]["f1"], 5)}
        if "masks" in self.fixed_report:
            result["metrics/fixed_f1(M)"]=round(self.fixed_report["masks"]["f1"], 5)
        return result


class TrainingGateTest(unittest.TestCase):
    def check_gate(self, boxes, masks=None, valid=True):
        with TemporaryDirectory() as directory:
            trainer=FixedTrainerMixin()
            trainer.args=SimpleNamespace(task="segment" if masks is not None else "detect")
            trainer.validator=FakeValidator(boxes, masks, valid)
            trainer.best_fitness=None
            trainer.stop=False
            trainer.epoch=0
            trainer.save_dir=Path(directory)
            _, fitness=trainer.validate()
            return trainer.stop, fitness

    def test_rounded_below_target_does_not_stop(self):
        counts=count_metrics(449999, 50000, 50000)
        self.assertEqual(round(counts["f1"], 5), 0.9)
        stopped, fitness=self.check_gate(counts)
        self.assertFalse(stopped)
        self.assertLess(fitness, 0.9)

    def test_both_outputs_required(self):
        stopped, fitness=self.check_gate(count_metrics(95, 5, 5), count_metrics(85, 15, 15))
        self.assertFalse(stopped)
        self.assertEqual(fitness, 0.85)

    def test_exact_target_stops(self):
        stopped, fitness=self.check_gate(count_metrics(90, 10, 10), count_metrics(90, 10, 10))
        self.assertTrue(stopped)
        self.assertEqual(fitness, 0.9)

    def test_invalid_evaluation_never_stops(self):
        stopped, _=self.check_gate(count_metrics(90, 10, 10), valid=False)
        self.assertFalse(stopped)


if __name__=="__main__":
    unittest.main()
