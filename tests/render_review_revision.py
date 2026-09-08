"""Re-render saved OCR/translations locally, retaining all earlier review versions."""

import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_networking_components import image_pipeline
sys.path.insert(0, str(ROOT/"tests"))
from run_ocr_review_batches import review_files


def main():
    root=Path(sys.argv[1]).resolve()
    for batch in sorted(root.glob("batch_*_images")):
        previous=batch/"revision_03/result"
        revision=batch/"revision_04"
        output=revision/"result"
        output.mkdir(parents=True, exist_ok=False)
        for name in ("ocr.json", "translation_responses.json"):
            shutil.copy2(previous/name, output/name)
        shutil.copy2(batch/"sources.json", revision/"sources.json")
        entries=json.loads((output/"ocr.json").read_text(encoding="utf-8"))["images"]
        paths=[Path(entry["path"]) for entry in entries]
        # This command cannot send a request, even if its saved cache is incomplete.
        with patch.object(image_pipeline, "translate_regions", side_effect=AssertionError("Local rendering cannot call an API")):
            image_pipeline.process_images(paths, output, detector=None, ocr=None,
                                          font_path=Path("C:/Windows/Fonts/arial.ttf"), prepared=True)
        review_files(revision, paths)
        print(f"Local rendering complete: {batch.name}", flush=True)


if __name__=="__main__":
    main()
