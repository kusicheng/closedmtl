"""Launch an OCR evaluation with a log and post-exit lifetime RAM evidence."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

from ocr_memory_watch import observe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command
    if command and command[0]=="--":
        command=command[1:]
    if not command:
        parser.error("An evaluation script and its arguments are required")
    prefix=Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".log").open("x", encoding="utf-8") as log:
        child=subprocess.Popen([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
        created=psutil.Process(child.pid).create_time()
        identity={"pid":child.pid, "create_time":created,
                  "supervisor_pid":psutil.Process().pid, "command":command}
        prefix.with_suffix(".process.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        memory=observe(child.pid, created)
        memory["exit_code"]=child.wait()
        prefix.with_suffix(".memory.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
        return memory["exit_code"]


if __name__=="__main__":
    raise SystemExit(main())
