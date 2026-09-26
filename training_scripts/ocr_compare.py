"""Check full labeled accuracy and CPU deployment RAM; keep ratios comparable."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from ocr_benchmark import edit_distance
from ocr_data import digest, read_rows
from ocr_text import post_process


def evidence(item, rows, expected_hash):
    """Recalculate quality from a complete ordered set of real predictions."""
    predictions=item.get("predictions", [])
    expected_ids=[row["id"] for row in rows]
    ids_hash=hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    complete=bool(rows) and len(set(expected_ids))==len(rows)
    complete=complete and len(predictions)==len(rows) and item.get("samples")==len(rows)
    complete=complete and item.get("manifest_sha256")==expected_hash
    complete=complete and item.get("sample_ids_sha256")==ids_hash
    edits=characters=exact=0
    if complete:
        for row, prediction in zip(rows, predictions):
            target=post_process(row["text"])
            actual=prediction.get("prediction")
            if prediction.get("id")!=row["id"] or prediction.get("target")!=target or not isinstance(actual, str):
                complete=False
                break
            distance=edit_distance(target, actual)
            if prediction.get("edits")!=distance:
                complete=False
                break
            edits+=distance
            characters+=len(target)
            exact+=actual==target
    cer=edits/max(1, characters) if complete else None
    accuracy=exact/len(rows) if complete else None
    aggregates=complete and all(isinstance(item.get(key), (int, float))
                               and math.isfinite(item[key])
                               and math.isclose(item[key], value, rel_tol=0, abs_tol=1e-12)
                               for key, value in (("cer", cer), ("exact_match", accuracy)))
    memory=item.get("ram_peak_bytes")
    valid_memory=type(memory) is int and memory>0 and item.get("ram_peak_method")=="windows_peak_working_set"
    return {"complete":complete, "aggregates":aggregates, "memory":valid_memory,
            "cer":cer, "exact_match":accuracy}


def compare(teacher, student, manifest, cer_tolerance=0.01, exact_tolerance=0.02):
    if not 0<=cer_tolerance<=1 or not 0<=exact_tolerance<=1:
        raise ValueError("Quality tolerances must be finite fractions in [0, 1]")
    rows=read_rows(manifest)
    count=len(rows)
    expected=digest(manifest)
    first=evidence(teacher, rows, expected)
    second=evidence(student, rows, expected)
    valid_quality=first["complete"] and second["complete"]
    valid_memory=first["memory"] and second["memory"]
    same_cpu=teacher.get("device")==student.get("device")=="cpu"
    checks={
        "full_same_manifest":valid_quality,
        "recomputed_aggregates":first["aggregates"] and second["aggregates"],
        "cpu_deployment":student.get("device")=="cpu" and teacher.get("device") in ("cpu", "cuda"),
        "real_peak_counter":valid_memory,
        "cer":valid_quality and second["cer"]<=first["cer"]+cer_tolerance,
        "exact_match":valid_quality and second["exact_match"]>=first["exact_match"]-exact_tolerance,
        "peak_ram":valid_memory and (student["ram_peak_bytes"]<1000000000 or
                   same_cpu and student["ram_peak_bytes"]<=teacher["ram_peak_bytes"]/3),
    }
    return {"passed":all(checks.values()), "checks":checks,
            "samples_required":count,
            "cer_delta":second["cer"]-first["cer"] if valid_quality else None,
            "exact_match_delta":second["exact_match"]-first["exact_match"] if valid_quality else None,
            "peak_ram_ratio":student["ram_peak_bytes"]/teacher["ram_peak_bytes"] if valid_memory else None,
            "student_private_bytes":student.get("private_bytes"),
            "teacher_device":teacher.get("device"), "student_device":student.get("device"),
            "ratio_acceptance_allowed":same_cpu,
            "cer_tolerance":cer_tolerance, "exact_tolerance":exact_tolerance}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--manifest", default="training_data/manga109_ocr/test.jsonl")
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    teacher=json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    student=json.loads(Path(args.student).read_text(encoding="utf-8"))
    result=compare(teacher, student, args.manifest)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
