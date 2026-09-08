"""Japanese manga OCR within detected bubbles; no panel segmentation is inferred.

Manga OCR recognizes Japanese only. Detection order is a rough row grouping,
top to bottom and right to left. Coordinates refer to stored image pixels.
"""

import math
import os
from pathlib import Path


def load_models(weights, ocr_model="kha-white/manga-ocr-base"):
    """Load YOLO and Manga OCR lazily in the caller's Python environment."""
    root=Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root/"models/huggingface"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root/"models/ultralytics_config"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    from manga_ocr import MangaOcr
    from ultralytics import YOLO

    if not Path(weights).is_file():
        raise FileNotFoundError(weights)
    return YOLO(str(weights)), MangaOcr(pretrained_model_name_or_path=ocr_model)


def _overlap(first, second):
    left=max(first[0], second[0])
    top=max(first[1], second[1])
    right=min(first[2], second[2])
    bottom=min(first[3], second[3])
    intersection=max(0, right-left)*max(0, bottom-top)
    area1=(first[2]-first[0])*(first[3]-first[1])
    area2=(second[2]-second[0])*(second[3]-second[1])
    return intersection/max(1, min(area1, area2))


def _reading_order(detections):
    rows=[]
    for item in sorted(detections, key=lambda item: item[0][1]):
        box=item[0]
        middle=(box[1]+box[3])/2
        for row in rows:
            anchor=row[0][0]
            anchor_middle=(anchor[1]+anchor[3])/2
            if abs(middle-anchor_middle)<=0.4*min(box[3]-box[1], anchor[3]-anchor[1]):
                row.append(item)
                break
        else:
            rows.append([item])
    return [item for row in rows for item in sorted(row, key=lambda item: -item[0][0])]


def _refine_box(image, box):
    """Find isolated ink on a light interior, excluding border-connected ink.

    This conservative heuristic skips unsuitable crops. It cannot distinguish
    every drawing from text. Keep source images for manual review.
    """
    import cv2
    import numpy as np

    left, top, right, bottom=box
    # Use the whole bubble: proportional insets cut off edge-adjacent columns.
    if right-left<8 or bottom-top<8:
        return None
    array=np.asarray(image.crop((left, top, right, bottom)).convert("RGB"))
    gray=cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # OCR hallucinations on blank crops are common. Require actual isolated ink.
    light=gray>=190
    if float(light.mean())<0.60:
        return None
    background=np.median(array[light], axis=0).astype(int).tolist()
    light_count, light_labels, light_stats, _=cv2.connectedComponentsWithStats(
        light.astype("uint8"), connectivity=8)
    if light_count<2:
        return None
    interior_label=1+int(np.argmax(light_stats[1:, cv2.CC_STAT_AREA]))
    ink=(gray<min(170, int(np.median(gray[light]))-55)).astype("uint8")
    count, labels, stats, _=cv2.connectedComponentsWithStats(ink, connectivity=8)
    height, width=gray.shape
    components=[]
    for index in range(1, count):
        x, y, w, h, area=map(int, stats[index])
        if x==0 or y==0 or x+w==width or y+h==height:
            continue
        if area<3 or area>width*height*0.06 or w>max(30, width*0.3) or h>max(30, height*0.25):
            continue
        near_edge=x<width*0.12 or y<height*0.12 or x+w>width*0.88 or y+h>height*0.88
        if near_edge and max(w, h)>12 and max(w, h)/max(1, min(w, h))>4:
            continue
        # Text is surrounded by the main white bubble interior. Disconnected
        # artwork outside a spiky border can otherwise look like small glyphs.
        surround=np.concatenate((light_labels[y-1, x-1:x+w+1],
                                 light_labels[y+h, x-1:x+w+1],
                                 light_labels[y:y+h, x-1], light_labels[y:y+h, x+w]))
        if float((surround==interior_label).mean())<0.70:
            continue
        components.append((x, y, x+w, y+h, area))
    if not components or sum(part[4] for part in components)<max(10, width*height*0.001):
        return None
    x1=min(part[0] for part in components)
    y1=min(part[1] for part in components)
    x2=max(part[2] for part in components)
    y2=max(part[3] for part in components)
    padding=max(2, round(min(width, height)*0.025))
    refined=[left+max(0, x1-padding), top+max(0, y1-padding),
             left+min(width, x2+padding), top+min(height, y2+padding)]
    # Per-component rectangles preserve artwork between text columns. They are
    # absolute exclusive-edge boxes, not the much larger enclosing wipe box.
    wipe_rects=[[left+max(0, part[0]-1), top+max(0, part[1]-1),
                 left+min(width, part[2]+1), top+min(height, part[3]+1)]
                for part in components]
    return refined, background, wipe_rects


def extract_regions(image_path, detector, ocr, image_id):
    """Return accepted Japanese OCR regions with stable per-image IDs.

    Each result retains the source OCR in ``text`` and the original bubble in
    ``detector_box``. The replacement ``box`` excludes bubble edges where the
    interior heuristic can identify isolated ink. Unsupported/blank crops are
    skipped; an empty list means no accepted text, not proof that none exists.
    """
    from PIL import Image

    with Image.open(image_path) as source:
        image=source.convert("RGB")
    try:
        predictions=detector.predict(source=image, imgsz=768, conf=0.35, iou=0.5, verbose=False)
        detections=[]
        for prediction in predictions:
            if prediction.boxes is None:
                continue
            coordinates=prediction.boxes.xyxy.cpu().tolist()
            confidences=prediction.boxes.conf.cpu().tolist()
            for coordinates, confidence in zip(coordinates, confidences):
                if len(coordinates)!=4 or not all(math.isfinite(value) for value in coordinates):
                    continue
                left, top, right, bottom=coordinates
                box=[max(0, math.floor(left)), max(0, math.floor(top)),
                     min(image.width, math.ceil(right)), min(image.height, math.ceil(bottom))]
                if box[2]-box[0]<12 or box[3]-box[1]<12 or not math.isfinite(confidence):
                    continue
                detections.append((box, float(confidence)))
        selected=[]
        for item in sorted(detections, key=lambda item: -item[1]):
            if not any(_overlap(item[0], previous[0])>0.75 for previous in selected):
                selected.append(item)
        regions=[]
        for position, (detector_box, confidence) in enumerate(_reading_order(selected), 1):
            refined=_refine_box(image, detector_box)
            if refined is None:
                continue
            box, background, wipe_rects=refined
            # Manga OCR needs all columns, including ink that the conservative
            # wiping heuristic may reject. Bubble borders are valid OCR input.
            with image.crop(detector_box) as crop:
                text=ocr(crop)
            if not isinstance(text, str) or not text.strip():
                continue
            regions.append({"id": f"{image_id}:box_{position:03d}", "image_id": str(image_id),
                            "box": box, "text": text.strip(), "confidence": confidence,
                            "detector_box": detector_box, "background": background,
                            "wipe_rects": wipe_rects,
                            "warning": "Bubble-based text bounds and reading order are heuristic; review against source."})
        return regions
    finally:
        image.close()
