"""Prediction durability and accidental replacement protection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training_scripts"))
from ocr_journal import PredictionJournal


class JournalTests(unittest.TestCase):
    def test_rows_visible_before_close_and_existing_run_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"result.json"
            journal=PredictionJournal(output, {"model":"test"})
            row={"id":"sample", "prediction":"日本語"}
            try:
                journal.append(row)
                self.assertEqual(json.loads(journal.path.read_text(encoding="utf-8")), row)
                with self.assertRaises(FileExistsError):
                    PredictionJournal(output, {"model":"replacement"})
                self.assertEqual(json.loads(output.with_suffix(".journal.json").read_text()), {"model":"test"})
            finally:
                journal.close()


if __name__=="__main__":
    unittest.main()
