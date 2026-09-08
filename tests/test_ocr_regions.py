"""Offline boundary and blank-crop checks; no weights or network required."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import Mock

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_networking_components import ocr_regions


def prediction(boxes, confidence):
    coords=Mock()
    coords.cpu.return_value.tolist.return_value=boxes
    scores=Mock()
    scores.cpu.return_value.tolist.return_value=confidence
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=coords, conf=scores))


def test_blank_crop_is_not_sent_to_ocr():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"blank.png"
        Image.new("RGB", (100, 100), "white").save(path)
        detector=Mock()
        detector.predict.return_value=[prediction([[0, 0, 100, 100]], [0.9])]
        ocr=Mock()
        assert ocr_regions.extract_regions(path, detector, ocr, "blank")==[]
        ocr.assert_not_called()


def test_border_excluded_and_duplicates_removed():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"bubble.png"
        image=Image.new("RGB", (120, 120), "white")
        draw=ImageDraw.Draw(image)
        draw.ellipse((5, 5, 114, 114), outline="black", width=3)
        for x in (45, 60):
            for y in (35, 50, 65):
                draw.rectangle((x, y, x+7, y+9), fill="black")
        image.save(path)
        detector=Mock()
        detector.predict.return_value=[prediction([[0, 0, 120, 120], [1, 1, 119, 119]], [0.9, 0.7])]
        ocr=Mock(return_value="こんにちは")
        regions=ocr_regions.extract_regions(path, detector, ocr, "page_1")
        assert len(regions)==1
        assert regions[0]["text"]=="こんにちは"
        assert regions[0]["id"]=="page_1:box_001"
        left, top, right, bottom=regions[0]["box"]
        assert 35<=left<45 and 25<=top<35
        assert 68<=right<80 and 75<=bottom<85
        assert regions[0]["detector_box"]==[0, 0, 120, 120]
        assert ocr.call_count==1


def test_dark_crop_and_empty_ocr_are_skipped():
    image=Image.new("RGB", (100, 100), "black")
    assert ocr_regions._refine_box(image, [0, 0, 100, 100]) is None
    image.close()


def test_reading_order_and_overlap():
    detections=[([5, 10, 25, 40], 0.9), ([60, 12, 85, 45], 0.8), ([10, 70, 25, 95], 0.7)]
    ordered=ocr_regions._reading_order(detections)
    assert ordered==[detections[1], detections[0], detections[2]]
    assert ocr_regions._overlap([0, 0, 10, 10], [2, 2, 8, 8])==1
    assert ocr_regions._overlap([0, 0, 10, 10], [20, 20, 30, 30])==0


def test_full_crop_reaches_ocr_and_edge_column_is_retained():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"edge_column.png"
        image=Image.new("RGB", (200, 150), "white")
        draw=ImageDraw.Draw(image)
        # A right column only 3px from the detector edge used to be lost.
        for x in (90, 187):
            for y in (40, 58, 76):
                draw.rectangle((x, y, x+9, y+11), fill="black")
        image.save(path)
        detector=Mock()
        detector.predict.return_value=[prediction([[0, 0, 200, 150]], [0.9])]
        sizes=[]

        def recognize(crop):
            sizes.append(crop.size)
            assert crop.getpixel((190, 42))==(0, 0, 0)
            return "こんなこと"

        regions=ocr_regions.extract_regions(path, detector, recognize, "edge")
        assert sizes==[(200, 150)]
        assert len(regions)==1
        assert regions[0]["box"][2]>=197
        assert any(rect[0]<=190<rect[2] for rect in regions[0]["wipe_rects"])


def test_sparse_wipe_rectangles_exclude_spikes_and_intercolumn_art():
    image=Image.new("RGB", (240, 200), "white")
    draw=ImageDraw.Draw(image)
    # Disconnected long border spikes must not enlarge the text region.
    draw.polygon([(5, 10), (90, 12), (10, 17)], fill="black")
    draw.polygon([(220, 5), (235, 8), (232, 95)], fill="black")
    for x in (65, 150):
        for y in (65, 85, 105):
            draw.rectangle((x, y, x+9, y+12), fill="black")
    refined, _, rects=ocr_regions._refine_box(image, [0, 0, 240, 200])
    assert refined[0]>50 and refined[1]>50
    assert refined[2]<175 and refined[3]<135
    assert len(rects)==6
    # A rectangular union wipe would clear this gap; sparse wiping does not.
    assert not any(l<=110<r and t<=90<b for l, t, r, b in rects)
    assert all(0<=l<r<=240 and 0<=t<b<=200 for l, t, r, b in rects)
    image.close()


def test_exterior_art_is_not_selected_as_bubble_text():
    image=Image.new("RGB", (220, 180), (170, 170, 170))
    draw=ImageDraw.Draw(image)
    draw.rectangle((25, 5, 210, 175), fill="white", outline="black", width=3)
    for x in (70, 130):
        for y in (50, 75, 100):
            draw.rectangle((x, y, x+10, y+13), fill="black")
    # Small dark artwork outside the enclosed bubble has glyph-like dimensions.
    draw.rectangle((10, 55, 20, 68), fill="black")
    refined, _, rects=ocr_regions._refine_box(image, [0, 0, 220, 180])
    assert refined[0]>60
    assert len(rects)==6
    assert not any(l<=15<r and t<=60<b for l, t, r, b in rects)
    image.close()


if __name__=="__main__":
    tests=[value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} OCR region tests passed.")
