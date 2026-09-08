"""Offline pipeline and random review batch checks; no model or network calls."""

import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_networking_components import image_pipeline as pipeline
from api_networking_components import text_replacement as replacement
import run_ocr_review_batches as batches


FONT=Path("C:/Windows/Fonts/arial.ttf")


def raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception as exc:
        return exc
    raise AssertionError(f"Expected {exception.__name__}")


def source_image(root):
    path=root/"source.png"
    with Image.new("RGB", (180, 120), "white") as image:
        image.save(path)
    return path


def fake_extract(path, detector, ocr, image_id):
    return [{"id": image_id+"_r1", "image_id": image_id, "box": [10, 10, 170, 50],
             "text": "こんにちは", "label": "speech"},
            {"id": image_id+"_r2", "image_id": image_id, "box": [10, 65, 170, 110],
             "text": "ありがとう", "label": "speech"}]


def fake_translate(rows, target_language, key, model="minimax/minimax-m3:free"):
    assert target_language=="en"
    assert model=="minimax/minimax-m3:free"
    return {"regions": [{"id": row["id"], "text": "Hello" if row["id"].endswith("r1") else "Thank you"}
                         for row in reversed(rows)],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30}, "model": "minimax/minimax-m3:free"}


def test_box_ids_preserve_sources_and_map_reordered_translation():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        source=source_image(root)
        original=source.read_bytes()
        output=root/"result"
        with patch.object(pipeline, "extract_regions", side_effect=fake_extract) as extract:
            with patch.object(pipeline, "translate_regions", side_effect=fake_translate) as translate:
                result=pipeline.process_images([source], output, detector=None, ocr=None, font_path=FONT)
        extract.assert_called_once_with(source, None, None, "image_0001")
        translate.assert_called_once()
        rows=json.loads((output/"translated_regions.json").read_text(encoding="utf-8"))["images"][0]["regions"]
        assert [(row["id"], row["text"], row["source_text"]) for row in rows]==[
            ("image_0001_r1", "Hello", "こんにちは"), ("image_0001_r2", "Thank you", "ありがとう")]
        assert rows[0]["box"]==[10, 10, 170, 50] and rows[0]["label"]=="speech"
        assert result["status"]=="complete" and result["regions"]==2
        assert result["usage"]==[{"prompt_tokens": 100, "completion_tokens": 30}]
        assert source.read_bytes()==original
        with zipfile.ZipFile(result["zip"]) as archive:
            assert archive.testzip() is None
            names=[name for name in archive.namelist() if name.endswith(".png")]
            assert len(names)==1
            with Image.open(io.BytesIO(archive.read(names[0]))) as rendered, Image.open(source) as original:
                assert rendered.size==(180, 120)
                assert rendered.convert("RGB").tobytes()!=original.convert("RGB").tobytes()
        assert not list(output.glob(".replacement-*"))


def test_no_accepted_regions_skips_translation_and_preserves_pixels():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        source=source_image(root)
        with patch.object(pipeline, "extract_regions", return_value=[]):
            with patch.object(pipeline, "translate_regions") as translate:
                result=pipeline.process_images([source], root/"result", detector=None, ocr=None, font_path=FONT)
        translate.assert_not_called()
        assert result["regions"]==0 and result["usage"]==[]
        with zipfile.ZipFile(result["zip"]) as archive:
            name=next(name for name in archive.namelist() if name.endswith(".png"))
            with Image.open(io.BytesIO(archive.read(name))) as rendered, Image.open(source) as original:
                assert rendered.convert("RGB").tobytes()==original.convert("RGB").tobytes()


def test_prepared_resume_uses_saved_ocr_without_models():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        source=source_image(root)
        output=root/"result"
        output.mkdir()
        pipeline.save_json(output/"ocr.json", {"complete": True, "images": [{"path": str(source.resolve()), "image_id": "saved",
                            "regions": fake_extract(source, None, None, "saved")}]})
        before=(output/"ocr.json").read_bytes()
        with patch.object(pipeline, "extract_regions", side_effect=AssertionError("OCR must not run")) as extract:
            with patch.object(pipeline, "translate_regions", side_effect=fake_translate):
                result=pipeline.process_images([source], output, detector=None, ocr=None, font_path=FONT, prepared=True)
        extract.assert_not_called()
        assert result["images"]==1 and result["regions"]==2
        assert (output/"ocr.json").read_bytes()==before


def test_partial_ocr_resume_continues_only_remaining_sources():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        first=source_image(root)
        second=root/"second.png"
        second.write_bytes(first.read_bytes())
        output=root/"result"
        with patch.object(pipeline, "extract_regions", side_effect=[fake_extract(first, None, None, "image_0001"),
                                                                   RuntimeError("Interrupted OCR")]):
            raises(RuntimeError, pipeline.prepare_images, [first, second], output, detector=None, ocr=None)
        saved=json.loads((output/"ocr.json").read_text(encoding="utf-8"))
        assert saved["complete"] is False and len(saved["images"])==1
        with patch.object(pipeline, "extract_regions", side_effect=fake_extract) as extract:
            entries=pipeline.prepare_images([first, second], output, detector=None, ocr=None)
        extract.assert_called_once_with(second, None, None, "image_0002")
        assert len(entries)==2 and entries[0]==saved["images"][0]
        assert json.loads((output/"ocr.json").read_text(encoding="utf-8"))["complete"] is True


def test_prepare_and_prepared_process_reject_mismatched_source_lists():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        source=source_image(root)
        other=root/"other.png"
        other.write_bytes(source.read_bytes())
        output=root/"result"
        with patch.object(pipeline, "extract_regions", side_effect=fake_extract):
            pipeline.prepare_images([source], output, detector=None, ocr=None)
        before=(output/"ocr.json").read_bytes()
        with patch.object(pipeline, "extract_regions") as extract, patch.object(pipeline, "translate_regions") as translate:
            for paths in ([other], []):
                raises(ValueError, pipeline.prepare_images, paths, output, detector=None, ocr=None)
                raises(ValueError, pipeline.process_images, paths, output, detector=None, ocr=None,
                       font_path=FONT, prepared=True)
            extract.assert_not_called()
            translate.assert_not_called()
        assert (output/"ocr.json").read_bytes()==before


def test_prepared_translation_cache_avoids_api_and_keeps_usage():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        source=source_image(root)
        output=root/"result"
        with patch.object(pipeline, "extract_regions", side_effect=fake_extract):
            entries=pipeline.prepare_images([source], output, detector=None, ocr=None)
        response=fake_translate(entries[0]["regions"], "en", None)
        pipeline.save_json(output/"translation_responses.json", [response])
        with patch.object(pipeline, "extract_regions") as extract, patch.object(pipeline, "translate_regions") as translate:
            result=pipeline.process_images([source], output, detector=None, ocr=None, font_path=FONT, prepared=True)
            extract.assert_not_called()
            translate.assert_not_called()
        assert result["status"]=="complete" and result["usage"]==[response["usage"]]
        rows=json.loads((output/"translated_regions.json").read_text(encoding="utf-8"))["images"][0]["regions"]
        assert rows[0]["text"]=="Hello" and rows[0]["source_text"]=="こんにちは"


def test_sparse_ink_wiping_preserves_pixels_between_rectangles():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        source=source_image(root)
        with Image.open(source) as image:
            image=image.convert("RGB")
            draw=ImageDraw.Draw(image)
            draw.rectangle((20, 20, 29, 29), fill="black")
            draw.rectangle((60, 20, 69, 29), fill="black")
            draw.rectangle((40, 20, 49, 29), fill="red")
            image.save(source)
        region={"box": [15, 15, 75, 35], "layout_box": [10, 10, 100, 80], "text": "",
                "wipe_rects": [[20, 20, 30, 30], [60, 20, 70, 30]]}
        output=root/"wiped.png"
        replacement.replace_text(source, [region], output, font_path=FONT)
        with Image.open(output) as rendered:
            rendered=rendered.convert("RGB")
            assert rendered.getpixel((25, 25))==(255, 255, 255)
            assert rendered.getpixel((65, 25))==(255, 255, 255)
            assert rendered.getpixel((45, 25))==(255, 0, 0)
        region["wipe_rects"]=[[0, 0, 30, 30]]
        raises(ValueError, replacement.replace_text, source, [region], root/"invalid.png", font_path=FONT)


def test_zip_traversal_and_duplicate_basenames_use_safe_unique_names():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        image=source_image(root).read_bytes()
        upload=root/"upload.zip"
        with zipfile.ZipFile(upload, "w") as archive:
            for name in ("../escaped.png", "first/panel.png", "second/panel.png", "C:\\outside.png"):
                archive.writestr(name, image)
            archive.writestr("ignore.txt", "not an image")
        original=upload.read_bytes()
        extracted=pipeline.extract_zip(upload, root/"extracted")
        assert len(extracted)==4 and len(set(extracted))==4
        for path in extracted:
            assert path.parent==(root/"extracted").resolve()
            assert path.name.isascii() and path.read_bytes()==image
            with Image.open(path) as opened:
                opened.verify()
        assert not (root/"escaped.png").exists()
        assert upload.read_bytes()==original


def test_invalid_zip_and_unsupported_archive_preserve_upload():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        corrupt=root/"corrupt.zip"
        corrupt.write_bytes(b"not a ZIP")
        raises(zipfile.BadZipFile, pipeline.extract_zip, corrupt, root/"bad_extract")
        assert corrupt.read_bytes()==b"not a ZIP"
        upload=root/"no_images.zip"
        with zipfile.ZipFile(upload, "w") as archive:
            archive.writestr("readme.txt", "documentation")
        original=upload.read_bytes()
        raises(ValueError, pipeline.extract_zip, upload, root/"empty_extract")
        assert upload.read_bytes()==original


def test_random_plan_unique_sizes_images_and_groups():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        for split in ("train", "valid"):
            (root/split).mkdir()
            for group in range(25):
                for page in range(8):
                    (root/split/f"ja_chapter{group}_{page}_png_jpg.rf.{split}.jpg").touch()
        plan=batches.make_plan(root, 5, 48219)
        assert plan==batches.make_plan(root, 5, 48219)
        sizes=[item["size"] for item in plan]
        assert len(sizes)==len(set(sizes))==5 and all(1<=size<=20 for size in sizes)
        identities=[]
        for item in plan:
            assert len(item["sources"])==item["size"]
            names=[Path(path).name.split(".rf.")[0] for path in item["sources"]]
            groups=[re.sub(r"_\d+_png_jpg$", "", name) for name in names]
            assert len(groups)==len(set(groups))
            identities.extend(names)
        assert len(identities)==len(set(identities))


def test_impossible_sampling_and_invalid_batch_count_are_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        raises(ValueError, batches.make_plan, Path(temporary), 5, 12)
    for count in ("0", "21", "-1"):
        with patch.object(sys, "argv", ["run_ocr_review_batches.py", "--batches", count]):
            with patch.object(batches, "load_models") as models:
                with contextlib.redirect_stderr(io.StringIO()):
                    exc=raises(SystemExit, batches.main)
                assert exc.code==2
                models.assert_not_called()


if __name__=="__main__":
    tests=[unittest.FunctionTestCase(function) for name, function in sorted(globals().items())
           if name.startswith("test_")]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
    sys.exit(not result.wasSuccessful())
