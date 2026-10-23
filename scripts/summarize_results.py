"""Print a markdown summary of a lighteval results file.

Examples:
    python scripts/summarize_results.py results/20261117T120000Z/results.json
    python scripts/summarize_results.py results/<run>/results.json --out results/<run>/summary.md

Accepts the results.json written by scripts/evaluate.py or the results_*.json
file lighteval writes itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from cogtools.eval.summarize import summarize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("results", help="path to a results JSON file")
    p.add_argument("--domains", default="configs/eval_domains.yaml", help="domain grouping YAML")
    p.add_argument("--out", default=None, help="also write the markdown to this file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    domains = yaml.safe_load(Path(args.domains).read_text(encoding="utf-8")) or {}
    text = summarize(results, domains)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
