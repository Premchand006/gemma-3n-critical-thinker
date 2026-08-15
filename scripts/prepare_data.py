"""Select the OpenThoughts3 training subset described by a train config and save it.

Example:
    python scripts/prepare_data.py --config configs/train_gemma3n_e4b.yaml \\
        --out data/openthoughts3_subset.jsonl

The saved file can be passed to training with ``--prepared-data``. Rows keep
the dataset's original columns (``difficulty``, ``source``, ``domain``,
``conversations``); conversion to chat format happens at training time.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from cogtools.config import load_train_config
from cogtools.data import compute_quotas, domain_counts, load_openthoughts_subset, save_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, type=Path, help="train config YAML")
    p.add_argument("--out", required=True, type=Path, help="output file, .jsonl or .parquet")
    p.add_argument("--total-samples", type=int, help="override total_samples")
    p.add_argument("--sampling", choices=["random", "first_n"], help="override sampling")
    p.add_argument("--shuffle-seed", type=int, help="override shuffle_seed")
    p.add_argument("--max-token-length", type=int, help="override max_token_length")
    p.add_argument("--max-stream-rows", type=int, help="override max_stream_rows")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_train_config(args.config)
    overrides = {
        "total_samples": args.total_samples,
        "sampling": args.sampling,
        "shuffle_seed": args.shuffle_seed,
        "max_token_length": args.max_token_length,
        "max_stream_rows": args.max_stream_rows,
    }
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

    quotas = compute_quotas(cfg.total_samples, cfg.domain_ratios, method=cfg.quota_rounding)
    print(f"quotas: {quotas} (total {sum(quotas.values())})")
    dataset = load_openthoughts_subset(cfg)
    print(f"selected {len(dataset)} rows: {domain_counts(dataset)}")
    path = save_rows(dataset, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
