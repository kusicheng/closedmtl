"""Persist each prediction independently of the final aggregate report."""

import json
from pathlib import Path


class PredictionJournal:
    def __init__(self, output, metadata):
        output=Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.path=output.with_suffix(".predictions.jsonl")
        self.stream=self.path.open("x", encoding="utf-8", buffering=1)
        output.with_suffix(".journal.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")

    def append(self, prediction):
        self.stream.write(json.dumps(prediction, ensure_ascii=False)+"\n")

    def close(self):
        self.stream.close()
