#!/usr/bin/env python3
"""Extract one validation prompt's answers across dumped validation steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read validation_data_dir JSONL dumps and track one prompt across steps."
    )
    parser.add_argument("validation_dir", type=Path, help="Directory containing {step}.jsonl validation dumps.")
    parser.add_argument(
        "--query",
        default=None,
        help="Substring used to select the prompt. Defaults to the first prompt in the first dump.",
    )
    parser.add_argument("--max-output-chars", type=int, default=1200, help="Truncate each answer for readability.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the matched prompt text.")
    return parser.parse_args()


def step_from_path(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**18


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    files = sorted(args.validation_dir.glob("*.jsonl"), key=step_from_path)
    if not files:
        raise SystemExit(f"No JSONL files found under {args.validation_dir}")

    query = args.query
    if query is None:
        first_rows = read_jsonl(files[0])
        if not first_rows:
            raise SystemExit(f"No rows in {files[0]}")
        query = first_rows[0]["input"]

    matched_prompt = None
    for path in files:
        rows = read_jsonl(path)
        match = next((row for row in rows if query in row.get("input", "")), None)
        if match is None:
            continue
        matched_prompt = match["input"]
        break

    if matched_prompt is None:
        raise SystemExit(f"No prompt matched query: {query!r}")

    if args.show_prompt:
        print("# Prompt")
        print(matched_prompt)
        print()

    print("| step | score | output |")
    print("|---:|---:|---|")
    for path in files:
        rows = read_jsonl(path)
        match = next((row for row in rows if row.get("input") == matched_prompt), None)
        if match is None:
            continue
        output = " ".join(str(match.get("output", "")).split())
        if len(output) > args.max_output_chars:
            output = output[: args.max_output_chars].rstrip() + " ..."
        output = output.replace("|", "\\|")
        print(f"| {match.get('step', path.stem)} | {match.get('score')} | {output} |")


if __name__ == "__main__":
    main()
