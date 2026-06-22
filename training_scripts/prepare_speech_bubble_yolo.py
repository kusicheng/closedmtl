from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def convert_split(source_dir: Path, output_dir: Path, split: str, category_to_index: dict[int, int]) -> None:
    split_dir = source_dir / split
    annotations_path = split_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing required annotation file: {annotations_path}")

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    labels_by_image: dict[int, list[str]] = {image_id: [] for image_id in images_by_id}

    for annotation in data.get("annotations", []):
        image = images_by_id[int(annotation["image_id"])]
        image_width = float(image["width"])
        image_height = float(image["height"])
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        class_index = category_to_index[int(annotation["category_id"])]
        labels_by_image[int(annotation["image_id"])].append(
            " ".join(
                [
                    str(class_index),
                    f"{(x + width / 2) / image_width:.8f}",
                    f"{(y + height / 2) / image_height:.8f}",
                    f"{width / image_width:.8f}",
                    f"{height / image_height:.8f}",
                ]
            )
        )

    image_output_dir = output_dir / "images" / split
    label_output_dir = output_dir / "labels" / split
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    missing_images: list[Path] = []
    for image_id, image in images_by_id.items():
        source_image = split_dir / image["file_name"]
        if not source_image.exists():
            missing_images.append(source_image)
            continue
        shutil.copy2(source_image, image_output_dir / image["file_name"])
        label_path = label_output_dir / f"{Path(image['file_name']).stem}.txt"
        label_path.write_text("\n".join(labels_by_image[image_id]) + "\n", encoding="utf-8")

    if missing_images:
        preview = "\n".join(str(path) for path in missing_images[:10])
        raise FileNotFoundError(f"{len(missing_images)} referenced images are missing:\n{preview}")


def prepare_dataset(source_dir: Path, output_dir: Path) -> Path:
    train_annotations = json.loads((source_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    categories = [
        category
        for category in train_annotations.get("categories", [])
        if int(category.get("id", 0)) != 0
    ]
    categories = sorted(categories, key=lambda category: int(category["id"]))
    if not categories:
        raise ValueError("No speech-bubble categories found in training annotations.")

    category_to_index = {int(category["id"]): index for index, category in enumerate(categories)}
    class_names = [str(category["name"]) for category in categories]

    for split in ("train", "valid", "test"):
        convert_split(source_dir, output_dir, split, category_to_index)

    yaml_path = output_dir / "speech_bubbles.yaml"
    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert speech bubble COCO annotations to YOLO format.")
    parser.add_argument(
        "--source-dir",
        default="training_data/speech-bubbles-detection",
        help="Path to the Roboflow COCO dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="training_data/speech-bubbles-detection-yolo",
        help="Path for the generated YOLO dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    yaml_file = prepare_dataset(Path(args.source_dir), Path(args.output_dir))
    print(f"Wrote YOLO dataset YAML: {yaml_file}")
