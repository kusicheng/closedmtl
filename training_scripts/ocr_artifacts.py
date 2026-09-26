"""Bind evaluation evidence to local model bytes without loading large files."""

import hashlib
import json
from pathlib import Path


def file_hash(path):
    hasher=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk:=stream.read(1024*1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def fingerprint(directory):
    directory=Path(directory)
    suffixes={".pt", ".bin", ".safetensors", ".json", ".txt"}
    files={path.name:file_hash(path) for path in sorted(directory.iterdir())
           if path.is_file() and path.suffix in suffixes}
    if not files:
        raise ValueError(f"No model artifacts found: {directory}")
    encoded=json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"model_artifact_sha256":hashlib.sha256(encoded).hexdigest(), "model_files_sha256":files}
