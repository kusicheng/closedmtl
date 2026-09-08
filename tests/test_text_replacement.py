"""Offline image rendering and archive contract checks; run this file directly."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_networking_components import text_replacement as replacement


FONT=Path("C:/Windows/Fonts/arial.ttf")


def raises(exception, callback, *args, **kwargs):
    try:
        callback(*args, **kwargs)
    except exception as exc:
        return exc
    raise AssertionError(f"Expected {exception}")


def source_image(directory, name="panel.png"):
    path=directory/name
    with Image.new("RGB", (180, 120), (20, 80, 140)) as image:
        image.save(path)
    return path


def read_image(archive, name):
    with Image.open(io.BytesIO(archive.read(name))) as image:
        return image.convert("RGB")


def test_region_replacement_and_source_preservation():
    assert FONT.is_file(), f"Test font is missing: {FONT}"
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        original=source.read_bytes()
        output=directory/"translated.zip"
        box=[20, 20, 160, 100]
        images=[{"path": source, "regions": [{"box": box,
                 "text": "D\u00e9j\u00e0 pr\u00eat !"}]}]
        result=replacement.replace_and_zip(images, output, font_path=FONT)
        assert result==output.resolve()
        assert source.read_bytes()==original
        assert set(directory.iterdir())=={source, output}
        with zipfile.ZipFile(output) as archive:
            assert set(archive.namelist())=={"000001_panel.png", "manifest.json"}
            assert archive.testzip() is None
            json.loads(archive.read("manifest.json"))
            with read_image(archive, "000001_panel.png") as rendered:
                assert rendered.size==(180, 120)
                inside=[]
                for y in range(120):
                    for x in range(180):
                        pixel=rendered.getpixel((x, y))
                        if 20<=x<160 and 20<=y<100:
                            inside.append(pixel)
                        else:
                            assert pixel==(20, 80, 140), (x, y, pixel)
                assert (0, 0, 0) in inside, "Translated glyphs must be drawn"
                assert (255, 255, 255) in inside, "Old text background must be cleared"
                assert (20, 80, 140) not in inside


def test_no_regions_preserves_pixels():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        output=directory/"unchanged.zip"
        replacement.replace_and_zip([{"path": str(source), "regions": []}],
                                    output, font_path=str(FONT))
        with zipfile.ZipFile(output) as archive:
            with read_image(archive, "000001_panel.png") as rendered:
                with Image.open(source) as original:
                    assert rendered.tobytes()==original.tobytes()
        assert set(directory.iterdir())=={source, output}


def test_51_images_across_batch_boundary_have_unique_names():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        original=source.read_bytes()
        output=directory/"all_images.zip"
        images=[{"path": source, "regions": []} for _ in range(51)]
        replacement.replace_and_zip(images, output, font_path=FONT, batch_size=50)
        with zipfile.ZipFile(output) as archive:
            expected={f"{number:06d}_panel.png" for number in range(1, 52)}
            assert set(archive.namelist())==expected|{"manifest.json"}
            assert len(archive.namelist())==52
            assert archive.testzip() is None
            for name in expected:
                with read_image(archive, name) as rendered:
                    assert rendered.size==(180, 120)
                    assert rendered.getpixel((0, 0))==(20, 80, 140)
        assert source.read_bytes()==original
        assert set(directory.iterdir())=={source, output}


def test_output_collision_preserves_existing_archive():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        output=directory/"existing.zip"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("keep.txt", "Prior work")
        original=output.read_bytes()
        raises(FileExistsError, replacement.replace_and_zip,
               [{"path": source, "regions": []}], output, font_path=FONT)
        assert output.read_bytes()==original
        assert set(directory.iterdir())=={source, output}


def test_bad_region_or_unfittable_text_retains_work_and_source():
    for box, text in (([-1, 0, 160, 100], "Bonjour"),
                      ([20, 20, 181, 100], "Bonjour"),
                      ([20, 20, 21, 21], "A long translation "*100)):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            source=source_image(directory)
            original=source.read_bytes()
            output=directory/"failed.zip"
            images=[{"path": source, "regions": []},
                    {"path": source, "regions": [{"box": box, "text": text}]}]
            raises((ValueError, RuntimeError), replacement.replace_and_zip,
                   images, output, font_path=FONT)
            assert source.read_bytes()==original
            assert not output.exists()
            staging=[path for path in directory.iterdir() if path.is_dir()]
            assert len(staging)==1
            assert (staging[0]/"000001_panel.png").is_file()


def test_zip_failure_keeps_rendered_images():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        original=source.read_bytes()
        output=directory/"failed.zip"
        images=[{"path": source, "regions": []}]
        with patch.object(zipfile.ZipFile, "write", side_effect=OSError("Disk write failed")):
            raises((OSError, RuntimeError), replacement.replace_and_zip,
                   images, output, font_path=FONT)
        assert source.read_bytes()==original
        assert not output.exists()
        staging=[path for path in directory.iterdir() if path.is_dir()]
        assert len(staging)==1
        assert (staging[0]/"000001_panel.png").is_file()


def test_zip_verification_failure_does_not_publish():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        original=source.read_bytes()
        output=directory/"unverified.zip"
        with patch.object(zipfile.ZipFile, "testzip", return_value="000001_panel.png"):
            error=raises(RuntimeError, replacement.replace_and_zip,
                         [{"path": source, "regions": []}], output, font_path=FONT)
        assert "verification failed" in str(error)
        assert source.read_bytes()==original
        assert not output.exists()
        staging=[path for path in directory.iterdir() if path.is_dir()]
        assert len(staging)==1
        assert (staging[0]/"000001_panel.png").is_file()
        assert (staging[0]/"result.zip").is_file()


def test_publication_collision_preserves_competing_file_and_recovery():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        original=source.read_bytes()
        output=directory/"raced.zip"
        competing=b"Existing output created while images were rendering"

        def competing_publication(staged_archive, destination):
            Path(destination).write_bytes(competing)
            raise FileExistsError("Another job created this output")

        with patch.object(replacement.os, "link", side_effect=competing_publication):
            raises(RuntimeError, replacement.replace_and_zip,
                   [{"path": source, "regions": []}], output, font_path=FONT)
        assert source.read_bytes()==original
        assert output.read_bytes()==competing
        staging=[path for path in directory.iterdir() if path.is_dir()]
        assert len(staging)==1
        assert (staging[0]/"000001_panel.png").is_file()
        with zipfile.ZipFile(staging[0]/"result.zip") as archive:
            assert archive.testzip() is None
            assert "000001_panel.png" in archive.namelist()


def test_invalid_batch_sizes_create_no_outputs():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        output=directory/"invalid.zip"
        for batch_size in (0, 51, True):
            raises(ValueError, replacement.replace_and_zip,
                   [{"path": source, "regions": []}], output,
                   font_path=FONT, batch_size=batch_size)
            assert set(directory.iterdir())=={source}


def test_expanded_layout_clears_only_wipe_box_with_rgb_list():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        output=directory/"cleared.png"
        region={"box": [80, 30, 100, 90], "layout_box": [20, 10, 160, 110],
                "text": "", "background": [245, 240, 230]}
        replacement.replace_text(source, [region], output, font_path=FONT)
        with Image.open(output) as rendered:
            for y in range(120):
                for x in range(180):
                    expected=(245, 240, 230) if 80<=x<100 and 30<=y<90 else (20, 80, 140)
                    assert rendered.getpixel((x, y))[:3]==expected, (x, y)


def test_expanded_layout_preserves_surroundings_except_glyphs():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        output=directory/"lettered.png"
        region={"box": [80, 30, 100, 90], "layout_box": [20, 10, 160, 110],
                "text": "Hello, my friend!", "background": [245, 240, 230],
                "color": [0, 0, 0]}
        replacement.replace_text(source, [region], output, font_path=FONT)
        changed_outside_wipe=0
        preserved_inside_layout=0
        with Image.open(output) as rendered:
            for y in range(120):
                for x in range(180):
                    pixel=rendered.getpixel((x, y))[:3]
                    if not (20<=x<160 and 10<=y<110):
                        assert pixel==(20, 80, 140), (x, y)
                    elif not (80<=x<100 and 30<=y<90):
                        if pixel==(20, 80, 140):
                            preserved_inside_layout+=1
                        else:
                            # Black glyph antialiasing can only darken each channel.
                            assert all(actual<=original for actual, original in zip(pixel, (20, 80, 140)))
                            changed_outside_wipe+=1
        assert changed_outside_wipe>20, "English glyphs should use the wider layout"
        assert preserved_inside_layout>100, "Layout must not flatten the surrounding image"


def test_invalid_layout_boxes_are_rejected_without_output():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        output=directory/"invalid.png"
        for layout in ([81, 10, 160, 110], [20, 31, 160, 110],
                       [20, 10, 99, 110], [20, 10, 160, 89],
                       [-1, 10, 160, 110], [20, 10, 181, 110],
                       [20, 10, 160, 121], [20, 10, 160],
                       [20, 10, 160, 110.0], [False, 10, 160, 110], None):
            raises(ValueError, replacement.replace_text, source,
                   [{"box": [80, 30, 100, 90], "layout_box": layout, "text": "Hello"}],
                   output, font_path=FONT)
            assert not output.exists()


def test_long_english_fits_expanded_narrow_vertical_source():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        output=directory/"english.png"
        region={"box": [85, 20, 95, 100],
                "text": "We should leave this place before the sun goes down."}
        raises(ValueError, replacement.replace_text, source, [region], output, font_path=FONT)
        assert not output.exists()
        region["layout_box"]=[10, 10, 170, 110]
        replacement.replace_text(source, [region], output, font_path=FONT)
        with Image.open(output) as rendered:
            assert rendered.size==(180, 120)
            assert any(rendered.getpixel((x, y))==(0, 0, 0, 255)
                       for y in range(rendered.height) for x in range(rendered.width))


def test_cli_resolves_source_relative_to_manifest():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=source_image(directory)
        manifest=directory/"input.json"
        manifest.write_text(json.dumps({"images": [{"path": source.name,
                            "regions": []}]}), encoding="utf-8")
        output=directory/"cli.zip"
        stdout=io.StringIO()
        with patch.object(sys, "argv", ["text_replacement.py", str(manifest),
                          str(output), "--font", str(FONT)]):
            with contextlib.redirect_stdout(stdout):
                status=replacement.main()
        assert status==0
        assert Path(stdout.getvalue().strip())==output.resolve()
        with zipfile.ZipFile(output) as archive:
            stored=json.loads(archive.read("manifest.json"))
            assert Path(stored["images"][0]["path"])==source.resolve()
            assert "000001_panel.png" in archive.namelist()
        assert not [path for path in directory.iterdir() if path.is_dir()]


if __name__=="__main__":
    tests=[unittest.FunctionTestCase(function) for name, function in sorted(globals().items())
           if name.startswith("test_")]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
    sys.exit(not result.wasSuccessful())
