"""Read-only verification of composed joint-training inputs and their lineage.

The composed format uses absolute dataset roots and absolute image entries in
text lists. Reject other input forms instead of guessing Ultralytics resolution.
No dataset loader is imported, so verification cannot create caches or download.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import yaml


REQUIRED_ARTIFACTS={"segments.yaml", "current_boxes.yaml", "train.txt", "val.txt",
                    "validation_groups.json", "image_label_manifest.jsonl",
                    "current_boxes_image_label_manifest.jsonl"}


def path_key(path):
    return str(Path(path).resolve()).casefold()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_absolute(value):
    path=Path(value)
    if not path.is_absolute():
        raise ValueError(f"Expected an absolute composed-data path: {value}")
    return path.resolve(strict=True)


class InputVerifier:
    def __init__(self):
        self.inventory={}
        self.stamps={}
        self.control_files={}

    def check(self, path, expected, control=False):
        path=Path(path).resolve(strict=True)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid recorded SHA256 for {path}")
        before=path.stat()
        actual=sha256(path)
        after=path.stat()
        if (before.st_size, before.st_mtime_ns)!=(after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed while hashing: {path}")
        if actual!=expected:
            raise ValueError(f"Input SHA256 differs from provenance: {path}")
        key=path_key(path)
        if key in self.inventory and self.inventory[key]["sha256"]!=actual:
            raise ValueError(f"Conflicting hashes for the same input: {path}")
        self.inventory[key]={"path":path.as_posix(), "sha256":actual, "size_bytes":after.st_size}
        self.stamps[path]=(after.st_size, after.st_mtime_ns)
        if control:
            self.control_files[path]=actual
        return actual

    def finish(self):
        for path, expected in self.stamps.items():
            stat=path.stat()
            if (stat.st_size, stat.st_mtime_ns)!=expected:
                raise ValueError(f"Input changed during verification: {path}")
        for path, expected in self.control_files.items():
            if sha256(path)!=expected:
                raise ValueError(f"Configuration or manifest changed during verification: {path}")
        ordered=[self.inventory[key] for key in sorted(self.inventory)]
        encoded=json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def read_manifest(path, label_kind, verifier):
    groups={"train":{}, "validation":{}}
    rows=[]
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row=json.loads(line)
            if row["split"] not in groups or row["label_kind"]!=label_kind:
                raise ValueError(f"Unexpected split or supervision kind in {path}")
            allowed={"ai4va", "manga"} if label_kind=="polygon" else {"current_boxes"}
            if row["dataset"] not in allowed:
                raise ValueError(f"Unexpected dataset in {path}")
            image=checked_absolute(row["image_path"])
            label=checked_absolute(row["label_path"])
            # Match the image-to-label rule used by the YOLO loader.
            # Path.parts handles Windows drives without string slash assumptions.
            parts=list(image.parts)
            if "images" not in parts:
                raise ValueError(f"Image path has no YOLO images directory: {image}")
            index=len(parts)-1-parts[::-1].index("images")
            parts[index]="labels"
            expected_label=Path(*parts).with_suffix(".txt")
            if path_key(label)!=path_key(expected_label):
                raise ValueError(f"Manifest label path differs from the YOLO loader: {image}")
            key=path_key(image)
            if any(key in group for group in groups.values()):
                raise ValueError(f"Repeated image identity in {path}: {image}")
            verifier.check(image, row["image_sha256"])
            verifier.check(label, row["label_sha256"])
            groups[row["split"]][key]=row
            rows.append(row)
    if any(not values for values in groups.values()):
        raise ValueError(f"Both training and validation rows are required: {path}")
    return groups, rows


def source_hashes(provenance):
    result={}
    for path, expected in provenance.get("source_hashes", {}).items():
        key=path_key(path)
        if key in result and result[key]!=expected:
            raise ValueError(f"Conflicting recorded source hashes: {path}")
        result[key]=expected
    return result


def verify_yaml(path, expected_groups, verifier, original_hashes):
    config=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "test" in config:
        raise ValueError(f"A training YAML must not expose a test entry: {path}")
    allowed={"path", "train", "val", "names", "nc"}
    if set(config)-allowed:
        raise ValueError(f"Unsupported training YAML settings: {path}")
    base=checked_absolute(config["path"])
    result={}
    for key, split in (("train", "train"), ("val", "validation")):
        value=config[key]
        if not isinstance(value, str):
            raise ValueError(f"Expected one text list for YAML {key}: {path}")
        listed=Path(value)
        listed=(listed if listed.is_absolute() else base/listed).resolve(strict=True)
        if listed.suffix.lower()!=".txt" or not listed.is_file():
            raise ValueError(f"Expected a text image list: {listed}")
        identity=path_key(listed)
        recorded=verifier.inventory.get(identity, {}).get("sha256", original_hashes.get(identity))
        if recorded is None:
            raise ValueError(f"Effective YAML list lacks recorded provenance: {listed}")
        verifier.check(listed, recorded, control=True)
        images=[checked_absolute(line.strip()) for line in listed.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        keys=[path_key(image) for image in images]
        if len(keys)!=len(set(keys)) or set(keys)!=set(expected_groups[split]):
            raise ValueError(f"Effective YAML {key} coverage differs from intended manifest: {listed}")
        result[key]={"path":listed.as_posix(), "sha256":recorded, "images":len(keys)}
    return result


def verify_joint_data(data_root):
    """Return a compact hash attestation after checking all effective inputs."""
    started=datetime.now(timezone.utc).isoformat()
    root=Path(data_root).resolve(strict=True)
    provenance_path=root/"provenance.json"
    provenance_bytes=provenance_path.read_bytes()
    provenance_hash=hashlib.sha256(provenance_bytes).hexdigest()
    provenance=json.loads(provenance_bytes)
    if provenance.get("status")!="complete":
        raise ValueError("Joint dataset composition is incomplete")
    artifacts=provenance["artifact_sha256"]
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise ValueError("Required composed artifacts are absent from provenance")
    verifier=InputVerifier()
    verifier.check(provenance_path, provenance_hash, control=True)
    artifact_hashes={}
    for name, expected in artifacts.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Artifact path escapes the composed root: {name}")
        path=(root/relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact path escapes the composed root: {name}")
        artifact_hashes[name]=verifier.check(path, expected, control=True)
    mask_groups, mask_rows=read_manifest(root/"image_label_manifest.jsonl", "polygon", verifier)
    box_groups, box_rows=read_manifest(root/"current_boxes_image_label_manifest.jsonl", "box", verifier)
    seen_paths, seen_hashes={}, {}
    for row in mask_rows+box_rows:
        for value, seen in ((path_key(row["image_path"]), seen_paths), (row["image_sha256"], seen_hashes)):
            previous=seen.setdefault(value, row["split"])
            if previous!=row["split"]:
                raise ValueError("Image content or identity overlaps training and validation")
    groups=json.loads((root/"validation_groups.json").read_text(encoding="utf-8"))
    normalized={path_key(path):group for path, group in groups.items()}
    expected={key:row["dataset"] for key, row in mask_groups["validation"].items()}
    if len(normalized)!=len(groups) or normalized!=expected or set(expected.values())!={"ai4va", "manga"}:
        raise ValueError("Validation group assignments differ from manifest sources")
    original_hashes=source_hashes(provenance)
    lists={"segments":verify_yaml(root/"segments.yaml", mask_groups, verifier, original_hashes),
           "current_boxes":verify_yaml(root/"current_boxes.yaml", box_groups, verifier, original_hashes)}
    inventory_hash=verifier.finish()
    return {"status":"verified", "data_root":root.as_posix(), "started_at_utc":started,
            "finished_at_utc":datetime.now(timezone.utc).isoformat(),
            "provenance_sha256":provenance_hash, "artifact_sha256":artifact_hashes,
            "effective_lists":lists, "verified_file_count":len(verifier.inventory),
            "verified_inventory_sha256":inventory_hash,
            "manifest_rows":{"segmentation":len(mask_rows), "current_boxes":len(box_rows)},
            "checker":{"path":str(Path(__file__).resolve()), "sha256":sha256(__file__)},
            "verification":"Every listed artifact and image/label hash matched; effective training/validation "
                           "lists exactly match manifests; no test YAML entry; source assignments verified."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh attestation path: {output}")
    result=verify_joint_data(args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status":result["status"], "verified_file_count":result["verified_file_count"],
                      "verified_inventory_sha256":result["verified_inventory_sha256"], "output":str(output)}))


if __name__=="__main__":
    main()
