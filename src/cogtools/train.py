"""LoRA fine-tuning of Gemma 3n with Unsloth and TRL.

The steps follow Unsloth's Gemma3N_(4B)-Conversational notebook, which the
original run used: ``FastModel.from_pretrained``, ``FastModel.get_peft_model``,
``get_chat_template("gemma-3")``, ``standardize_data_formats``, ``SFTTrainer``
with ``SFTConfig``, then ``train_on_responses_only`` so the loss covers only
the model turns.

Unsloth, torch, TRL and datasets are imported inside :func:`run_training`, so
importing this module does not need a GPU.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from cogtools import __version__
from cogtools.config import TrainConfig

log = logging.getLogger(__name__)

#: Markers the Gemma 3 chat template puts before user and model turns.
INSTRUCTION_PART = "<start_of_turn>user\n"
RESPONSE_PART = "<start_of_turn>model\n"

_VERSIONED_PACKAGES = (
    "cogtools",
    "unsloth",
    "unsloth_zoo",
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "xformers",
    "timm",
)


def run_training(cfg: TrainConfig) -> Path:
    """Train a LoRA adapter as described by ``cfg`` and return the output directory.

    The output directory receives the adapter and tokenizer files, a model
    card (``README.md``, license ``gemma``) and ``run_record.json`` with the
    config, git commit, package versions, domain counts and trainer metrics.

    The adapter is pushed to ``cfg.push_to_hub_repo`` only if that is set and
    the ``HF_TOKEN`` environment variable is set. The token is read from the
    environment and never written to disk or to the log.
    """
    # Unsloth patches transformers and trl, so it is imported first.
    from unsloth import FastModel
    from unsloth.chat_templates import (
        get_chat_template,
        standardize_data_formats,
        train_on_responses_only,
    )

    # isort: split
    import torch
    from trl import SFTConfig, SFTTrainer

    from cogtools.data import domain_counts, load_openthoughts_subset, load_prepared

    started = datetime.now(timezone.utc)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "started_utc": started.isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": _package_versions(),
    }
    log.info("config: %s", json.dumps(record["config"], sort_keys=True))
    log.info("git commit: %s", record["git_commit"])
    _write_record(output_dir, record)

    with contextlib.suppress(AttributeError):
        # The original install cell raised this limit for Gemma 3n.
        torch._dynamo.config.recompile_limit = 64

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model_name,
        dtype=None,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        random_state=cfg.seed,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    if cfg.prepared_data_path:
        dataset = load_prepared(cfg.prepared_data_path)
    else:
        dataset = load_openthoughts_subset(cfg, tokenizer=tokenizer)
    record["dataset"] = {
        "num_rows": len(dataset),
        "domain_counts": domain_counts(dataset) if "domain" in dataset.column_names else None,
        "source": cfg.prepared_data_path or cfg.dataset_name,
    }
    log.info("dataset: %s", record["dataset"])

    dataset = standardize_data_formats(dataset)

    def format_rows(batch: dict[str, list]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")  # the processor adds <bos> again
            for convo in batch["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(format_rows, batched=True)

    sft_args: dict[str, Any] = {
        "dataset_text_field": "text",
        "max_length": cfg.max_seq_length,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_steps": cfg.warmup_steps,
        "max_steps": cfg.max_steps if cfg.max_steps is not None else -1,
        "learning_rate": cfg.learning_rate,
        "logging_steps": cfg.logging_steps,
        "optim": cfg.optim,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "seed": cfg.seed,
        "output_dir": str(output_dir / "checkpoints"),
        "report_to": cfg.report_to,
    }
    if cfg.num_train_epochs is not None:
        sft_args["num_train_epochs"] = cfg.num_train_epochs
    if cfg.save_steps is not None:
        sft_args.update(save_strategy="steps", save_steps=cfg.save_steps)
    else:
        sft_args["save_strategy"] = "no"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(**sft_args),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record["gpu"] = {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 3),
            "reserved_before_train_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

    t0 = time.perf_counter()
    train_output = trainer.train()
    record["wall_time_s"] = round(time.perf_counter() - t0, 1)
    record["trainer_metrics"] = dict(train_output.metrics)
    record["log_history"] = list(trainer.state.log_history)
    if torch.cuda.is_available():
        record["gpu"]["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    card_path = output_dir / "README.md"
    card_path.write_text(model_card(cfg), encoding="utf-8")

    record["pushed_to"] = _maybe_push(model, tokenizer, card_path, cfg)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_record(output_dir, record)
    log.info("saved adapter and run_record.json to %s", output_dir)
    return output_dir


def model_card(cfg: TrainConfig) -> str:
    """Return a short Hugging Face model card for the adapter.

    The license is ``gemma`` because the adapter is a derivative of Gemma 3n,
    which is released under the Gemma Terms of Use.
    """
    if cfg.max_steps is not None:
        length = f"max_steps={cfg.max_steps}"
    else:
        length = f"num_train_epochs={cfg.num_train_epochs}"
    seen = ""
    if cfg.max_steps is not None:
        n = cfg.max_steps * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        seen = f" At most {n} examples were seen during training."
    return f"""---
base_model: {cfg.model_name}
library_name: peft
license: gemma
datasets:
- {cfg.dataset_name}
language:
- en
pipeline_tag: text-generation
tags:
- gemma3n
- lora
- unsloth
- trl
- sft
---

# LoRA adapter for {cfg.model_name}

LoRA adapter (r={cfg.lora_r}, alpha={cfg.lora_alpha}) trained with Unsloth and TRL
on a {cfg.total_samples}-row subset of {cfg.dataset_name}
({cfg.sampling} sampling per domain), with {length},
max_seq_length={cfg.max_seq_length} and learning rate {cfg.learning_rate}.{seen}
Examples longer than max_seq_length tokens were truncated.

Use of this adapter is subject to the Gemma Terms of Use
(https://ai.google.dev/gemma/terms). Training code: cogtools {__version__}.
Benchmark results: not yet measured.
"""


def _maybe_push(model: Any, tokenizer: Any, card_path: Path, cfg: TrainConfig) -> str | None:
    if not cfg.push_to_hub_repo:
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("push_to_hub_repo is set but HF_TOKEN is not; skipping the push")
        return None
    repo = cfg.push_to_hub_repo
    model.push_to_hub(repo, token=token, private=cfg.hub_private)
    tokenizer.push_to_hub(repo, token=token, private=cfg.hub_private)

    from huggingface_hub import HfApi

    # Upload the card last so it replaces any README written by the push.
    HfApi(token=token).upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo,
        commit_message="Add model card",
    )
    log.info("pushed adapter to %s", repo)
    return repo


def _git_commit() -> str | None:
    """Return the current git commit (with a ``-dirty`` suffix if needed), or None."""
    cwd = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _VERSIONED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_record(output_dir: Path, record: dict[str, Any]) -> None:
    path = output_dir / "run_record.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
