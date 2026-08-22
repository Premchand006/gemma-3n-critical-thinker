"""Train a Gemma 3n LoRA adapter from a YAML config.

Example (exact settings of the original run):
    python scripts/train.py --config configs/train_gemma3n_e4b_original.yaml

Command-line options override the matching config keys. To push the adapter,
set ``--push-to-hub-repo`` and export ``HF_TOKEN``; without the token the push
is skipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from cogtools.config import load_train_config

# CLI option -> TrainConfig field.
_OVERRIDES = {
    "model_name": str,
    "max_seq_length": int,
    "lora_r": int,
    "lora_alpha": int,
    "total_samples": int,
    "max_steps": int,
    "num_train_epochs": float,
    "learning_rate": float,
    "warmup_steps": int,
    "seed": int,
    "shuffle_seed": int,
    "sampling": str,
    "max_token_length": int,
    "output_dir": str,
    "push_to_hub_repo": str,
    "prepared_data_path": str,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, type=Path, help="train config YAML")
    for name, typ in _OVERRIDES.items():
        flag = "--" + name.replace("_", "-")
        if name == "prepared_data_path":
            flag = "--prepared-data"
        p.add_argument(flag, dest=name, type=typ, help=f"override {name}")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved config and exit without training",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_train_config(args.config)
    overrides = {k: getattr(args, k) for k in _OVERRIDES if getattr(args, k) is not None}
    cfg = dataclasses.replace(cfg, **overrides)
    if args.dry_run:
        for key, value in cfg.to_dict().items():
            print(f"{key}: {value}")
        return 0

    from cogtools.train import run_training

    out = run_training(cfg)
    print(f"adapter and run_record.json written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
