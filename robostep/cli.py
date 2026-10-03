"""Command-line helpers for frozen recipes and VLM caches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .schema import RewardProgram
from .vlm import CompileRequest, FrameVLMCompiler, FramePacket, FrozenAsyncVLM, build_frame_vlm_prompt
from .benchmarks import final_suite


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m robostep.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile", help="validate an offline recipe or print a compiler prompt")
    compile_parser.add_argument("--input", required=True)
    compile_parser.add_argument("--output", required=True)
    freeze_parser = sub.add_parser("freeze-cache", help="copy a JSONL cache and write a frozen manifest")
    freeze_parser.add_argument("--cache", required=True)
    freeze_parser.add_argument("--output", required=True)
    sub.add_parser("list-tasks", help="print the frozen 50/19/6 task inventory")
    args = parser.parse_args()
    if args.command == "compile":
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if "program" not in payload:
            request = CompileRequest(task_id=payload["task_id"], task_description=payload["task_description"], simulator_state_api=payload["simulator_state_api"], frames=tuple(FramePacket(**item) for item in payload.get("frames", ())), repair_round=int(payload.get("repair_round", 0)), training_diagnostics=payload.get("training_diagnostics", {}), gate_diagnostics=payload.get("gate_diagnostics", {}))
            print(build_frame_vlm_prompt(request))
            raise SystemExit("No program supplied; save the model's JSON and rerun with a program field")
        program = RewardProgram.from_dict(payload["program"])
        errors = program.validate()
        if errors:
            raise SystemExit("; ".join(errors))
        program.to_json(args.output)
        print(f"validated frozen recipe: {args.output}")
    elif args.command == "freeze-cache":
        source = Path(args.cache)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entries = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines() if line.strip()]
        destination.with_suffix(destination.suffix + ".frozen").write_text(json.dumps({"schema": "robostep-frozen-v1", "keys": sorted(item.get("key", "") for item in entries)}, indent=2) + "\n", encoding="utf-8")
        print(f"froze {len(entries)} cache entries: {destination}")
    elif args.command == "list-tasks":
        for item in final_suite():
            print(f"{item.benchmark}\t{item.task_id}\tK={len(item.stages)}\thorizon={item.horizon}\ttype={item.task_type}")


if __name__ == "__main__":
    main()
