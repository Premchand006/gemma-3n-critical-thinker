"""Run the cognitive-tools agent over a math JSONL file and score exact match.

Examples:
    python scripts/agent_eval.py --data data/sample_math.jsonl
    python scripts/agent_eval.py --data aime2024.jsonl --adapter none --max-turns 12

Each input line is {"id": ..., "question": ..., "answer": ...}. Writes
traces.jsonl and summary.json to --output-dir (default
results/agent_eval/<timestamp>). The summary is computed from the traces; no
score is written by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cogtools.eval.agent_eval import load_problems, run_agent_eval
from solve import add_agent_args, build_agent, tools_from_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="data/sample_math.jsonl", help="JSONL problem file")
    p.add_argument("--limit", type=int, default=None, help="only the first N problems")
    p.add_argument("--output-dir", default=None, help="where to write traces and summary")
    add_agent_args(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = load_problems(args.data, limit=args.limit)
    output_dir = args.output_dir or str(
        Path("results") / "agent_eval" / time.strftime("%Y%m%d-%H%M%S")
    )
    agent = build_agent(args)
    run_config = {
        "data": args.data,
        "limit": args.limit,
        "model": args.model,
        "adapter": None if str(args.adapter).lower() in {"", "none"} else args.adapter,
        "load_in_4bit": not args.no_4bit,
        "tools": tools_from_args(args),
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "tool_max_new_tokens": args.tool_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    summary = run_agent_eval(agent, problems, output_dir, run_config=run_config)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir}/traces.jsonl and {output_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
