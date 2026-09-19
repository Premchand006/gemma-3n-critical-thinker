"""Run the cognitive-tools agent on one question and print the trace.

Examples:
    python scripts/solve.py --question "What is 17 * 23?"
    python scripts/solve.py --question-file q.txt --adapter none --tools understand_question
    python scripts/solve.py --question "..." --enable-code --json-out trace.json

The model needs a GPU for 4-bit loading. Set HF_TOKEN in the environment if
the weights are gated. --enable-code lets the model run Python on this machine
in a child process with a timeout; that is not a security sandbox.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cogtools.tools import ALL_TOOLS, AVAILABLE_TOOLS, USE_CODE, CognitiveToolsAgent, HFBackend

DEFAULT_MODEL = "unsloth/gemma-3n-E4B-it"
DEFAULT_ADAPTER = "anhsirk/gemma-3n-critical-thinker"


def add_agent_args(p: argparse.ArgumentParser) -> None:
    """Model and agent options shared with scripts/agent_eval.py."""
    p.add_argument("--model", default=DEFAULT_MODEL, help="base model id or path")
    p.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="LoRA adapter id or path; 'none' for the base model only",
    )
    p.add_argument("--no-4bit", action="store_true", help="load full-precision weights")
    p.add_argument(
        "--tools",
        default=",".join(ALL_TOOLS),
        help=f"comma-separated tools to offer (choices: {', '.join(AVAILABLE_TOOLS)})",
    )
    p.add_argument(
        "--enable-code",
        action="store_true",
        help="also offer use_code, which runs model-written Python locally",
    )
    p.add_argument("--max-turns", type=int, default=8, help="cap on main-model generations")
    p.add_argument("--max-new-tokens", type=int, default=1024, help="tokens per main turn")
    p.add_argument("--tool-max-new-tokens", type=int, default=1024, help="tokens per tool call")
    p.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--code-timeout", type=float, default=10.0, help="seconds per use_code run")


def tools_from_args(args: argparse.Namespace) -> list[str]:
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    if args.enable_code and USE_CODE not in tools:
        tools.append(USE_CODE)
    return tools


def build_agent(args: argparse.Namespace) -> CognitiveToolsAgent:
    adapter = None if str(args.adapter).lower() in {"", "none"} else args.adapter
    backend = HFBackend(
        args.model,
        adapter=adapter,
        load_in_4bit=not args.no_4bit,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return CognitiveToolsAgent(
        backend,
        enabled_tools=tools_from_args(args),
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        tool_max_new_tokens=args.tool_max_new_tokens,
        code_timeout=args.code_timeout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument("--question", help="question text")
    q.add_argument("--question-file", help="file holding the question text")
    add_agent_args(p)
    p.add_argument("--json-out", help="also write the trace as JSON to this path")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    question = args.question
    if args.question_file:
        question = Path(args.question_file).read_text(encoding="utf-8").strip()
    agent = build_agent(args)
    trace = agent.solve(question)
    if args.json_out:
        Path(args.json_out).write_text(trace.to_json(indent=2) + "\n", encoding="utf-8")
    print(trace.to_json(indent=2) if args.format == "json" else trace.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
