"""Export bubble boxes, individual masks, a segmentation layer, and a preview."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"models/ultralytics_config"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_imgsz(value):
    if type(value) is not int or value<=0 or value%32:
        raise ValueError("Image size must be a positive integer multiple of 32")
    return value


def unique_json_object(pairs):
    result={}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate deployment config key: {key}")
        result[key]=value
    return result


def resolve_deployment(model_path, explicit_imgsz=None):
    """Read only hash-bound image size from a strict, optional sibling config."""
    model_path=Path(model_path).resolve(strict=True)
    model_sha256=digest(model_path)
    candidate=model_path.parent/"deployment.json"
    config_path=config_sha256=config_imgsz=None
    if os.path.lexists(candidate):
        if not candidate.is_file():
            raise ValueError(f"Deployment config is not a readable file: {candidate}")
        raw=candidate.read_bytes()
        try:
            config=json.loads(raw.decode("utf-8"), object_pairs_hook=unique_json_object)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Malformed deployment config: {candidate}") from error
        if not isinstance(config, dict) or set(config)!={"model_sha256", "imgsz"}:
            raise ValueError("Deployment config must contain exactly model_sha256 and imgsz")
        expected=config["model_sha256"]
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError("Deployment config model_sha256 must be a SHA256 hexadecimal string")
        if expected.lower()!=model_sha256:
            raise ValueError("Deployment config model_sha256 does not match the checkpoint")
        config_imgsz=valid_imgsz(config["imgsz"])
        config_path=str(candidate)
        config_sha256=hashlib.sha256(raw).hexdigest()
    if explicit_imgsz is not None:
        effective_imgsz=valid_imgsz(explicit_imgsz)
        source="explicit_override"
    elif config_imgsz is not None:
        effective_imgsz=config_imgsz
        source="deployment_config"
    else:
        effective_imgsz=768
        source="legacy_default"
    return {"model_sha256":model_sha256, "config_path":config_path,
            "config_sha256":config_sha256, "config_imgsz":config_imgsz,
            "explicit_imgsz":explicit_imgsz, "effective_imgsz":effective_imgsz,
            "imgsz_source":source}


def verify_deployment_unchanged(deployment):
    if deployment["config_path"] is not None:
        path=Path(deployment["config_path"])
        if not path.is_file() or digest(path)!=deployment["config_sha256"]:
            raise RuntimeError("Deployment config changed; predictions cannot be published")


def predict(args):
    model_path=Path(args.model).resolve(strict=True)
    deployment=resolve_deployment(model_path, args.imgsz)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    image_path=Path(args.image).resolve(strict=True)
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    source_bytes=image_path.read_bytes()
    image_sha256=hashlib.sha256(source_bytes).hexdigest()
    model_sha256=deployment["model_sha256"]
    source=cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("Cannot read source image")
    if args.mayocream:
        from training_scripts.bubble_models import load_mayocream
        network, _=load_mayocream(model_path)
        model=YOLO("yolo11n-seg.yaml", task="segment")
        model.model=network.eval()
        model.overrides["task"]="segment"
    else:
        model=YOLO(str(model_path), task="segment")
    if model.task!="segment" or len(model.names)!=1:
        raise ValueError("Expected a single-class bubble segmentation model")
    settings=dict(imgsz=deployment["effective_imgsz"], device=args.device, half=False, rect=False,
                  retina_masks=True, conf=0.35, iou=0.5, agnostic_nms=True,
                  max_det=300, verbose=False, save=False)
    result=model.predict(source, **settings)[0]
    verify_deployment_unchanged(deployment)
    if len(result.boxes) and (result.masks is None or len(result.masks)!=len(result.boxes)):
        raise ValueError("Boxes and instance masks differ")
    output.mkdir(parents=True)
    (output/"masks").mkdir()
    layer=np.zeros(source.shape[:2], dtype=np.uint8)
    preview=source.copy()
    rows=[]
    for index in range(len(result.boxes)):
        mask=(result.masks.data[index].cpu().numpy()>0.5).astype(np.uint8)*255
        if mask.shape!=source.shape[:2]:
            raise ValueError("Expected a full-resolution mask")
        name=f"masks/bubble_{index+1:04d}.png"
        if not cv2.imwrite(str(output/name), mask):
            raise OSError("Could not save instance mask")
        layer|=mask
        box=result.boxes.xyxy[index].cpu().tolist()
        confidence=float(result.boxes.conf[index])
        rows.append({"id":index+1, "box_xyxy":box, "confidence":confidence,
                     "mask":name, "mask_sha256":digest(output/name),
                     "mask_pixels":int(np.count_nonzero(mask))})
        selected=mask>0
        preview[selected]=(0.7*preview[selected]+0.3*np.array([50, 220, 70])).astype(np.uint8)
    for row in rows:
        x1, y1, x2, y2=[round(value) for value in row["box_xyxy"]]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 150, 240), 2)
        cv2.putText(preview, str(row["id"]), (x1, max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    for name, image in (("segmentation.png", layer), ("preview.jpg", preview)):
        if not cv2.imwrite(str(output/name), image):
            raise OSError(f"Could not save {name}")
    if digest(model_path)!=model_sha256 or digest(image_path)!=image_sha256:
        raise RuntimeError("Image or weights changed; partial exports are not published predictions")
    verify_deployment_unchanged(deployment)
    report={"image":str(image_path), "image_sha256":image_sha256,
            "model":str(model_path), "model_sha256":model_sha256,
            "deployment":deployment,
            "width":source.shape[1], "height":source.shape[0], "settings":settings,
            "segmentation_layer":"segmentation.png", "preview":"preview.jpg",
            "instances":rows, "meaning":"Model predictions, not ground-truth annotations."}
    (output/"boxes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output":str(output), "instances":len(rows)}), flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mayocream", action="store_true")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Positive multiple of 32; otherwise use hash-bound deployment.json or legacy 768")
    parser.add_argument("--device", default="0")
    predict(parser.parse_args())
