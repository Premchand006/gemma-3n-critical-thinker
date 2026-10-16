"""Score the model with lighteval 0.11.0.

Examples:
    python scripts/evaluate.py --config configs/eval_tasks.yaml
    python scripts/evaluate.py --max-samples 2 --tasks "leaderboard|gsm8k|5"
    python scripts/evaluate.py --adapter none          # base model only

Writes results/<timestamp>/results.json and prints a summary table.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import yaml

from cogtools.config import load_eval_config
from cogtools.eval.lighteval_runner import (
    DEFAULT_COMPUTE_DTYPE,
    DEFAULT_MAX_LENGTH,
    run_lighteval,
)
from cogtools.eval.summarize import summarize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/eval_tasks.yaml", help="eval config YAML")
    p.add_argument(
        "--domains",
        default="configs/eval_domains.yaml",
        help="domain grouping YAML for the summary",
    )
    p.add_argument("--model", default=None, help="override the base model id or path")
    p.add_argument(
        "--adapter",
        default=None,
        help="override the LoRA adapter; 'none' runs the base model alone",
    )
    p.add_argument(
        "--max-samples",
        default=None,
        help="per-task sample cap; 'none' for the full test sets",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="comma-separated task specs replacing the config list, e.g. 'leaderboard|gsm8k|5'",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument(
        "--compute-dtype",
        default=DEFAULT_COMPUTE_DTYPE,
        choices=["float16", "bfloat16", "float32"],
        help="bitsandbytes compute dtype; float16 on a T4 (watch for NaN output)",
    )
    return p.parse_args(argv)


def _none_or(value: str | None) -> str | None:
    if value is None or value.strip().lower() in {"none", "null", ""}:
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_eval_config(args.config)

    overrides: dict = {}
    if args.model is not None:
        overrides["model"] = args.model
    if args.adapter is not None:
        overrides["adapter"] = _none_or(args.adapter)
    if args.max_samples is not None:
        value = _none_or(args.max_samples)
        overrides["max_samples"] = None if value is None else int(value)
    if args.tasks is not None:
        overrides["tasks"] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    record = run_lighteval(cfg, max_length=args.max_length, compute_dtype=args.compute_dtype)

    domains_path = Path(args.domains)
    if domains_path.exists():
        domains = yaml.safe_load(domains_path.read_text(encoding="utf-8")) or {}
    else:
        domains = getattr(cfg, "domains", None) or {"all": list(cfg.tasks)}
    print(summarize(record, domains))
    return 0


if __name__ == "__main__":
    sys.exit(main())
