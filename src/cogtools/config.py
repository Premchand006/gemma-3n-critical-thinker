"""Training and evaluation configs, loaded from YAML.

Each YAML file maps one-to-one onto a dataclass. Unknown keys raise
``ValueError`` so that a typo in a config file fails before a GPU job starts.
"""

from __future__ import annotations

import math
import types
import typing
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Union

import yaml

#: Domain shares of OpenThoughts3-1.2M (850k math, 250k code, 100k science).
OPENTHOUGHTS3_RATIOS: dict[str, float] = {
    "math": 850 / 1200,
    "code": 250 / 1200,
    "science": 100 / 1200,
}

#: Allowed distance of ``sum(domain_ratios)`` from 1.0. The original run used
#: 0.708 + 0.208 + 0.083 = 0.999, which this tolerance accepts.
RATIO_SUM_TOLERANCE = 0.005

SAMPLING_METHODS = ("random", "first_n")
QUOTA_ROUNDING = ("largest_remainder", "floor")


@dataclass
class TrainConfig:
    """Settings for one LoRA fine-tuning run of Gemma 3n with Unsloth.

    The fields up to ``push_to_hub_repo`` follow the repo plan. The fields after
    it are the remaining knobs the original notebook hard-coded, plus the
    sampling options added in this repo.

    ``max_seq_length`` is the truncation length: Unsloth loads the model with it
    and it is passed to ``SFTConfig.max_length``. Longer examples are cut off at
    this many tokens, so the end of the reasoning trace and the final answer are
    lost. Use ``max_token_length`` to drop such rows instead.

    If ``max_steps`` is set it takes precedence over ``num_train_epochs`` (the
    Hugging Face Trainer behaviour).
    """

    model_name: str = "unsloth/gemma-3n-E4B-it"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    dataset_name: str = "open-thoughts/OpenThoughts3-1.2M"
    total_samples: int = 2500
    domain_ratios: dict[str, float] = field(default_factory=lambda: dict(OPENTHOUGHTS3_RATIOS))
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int | None = None
    num_train_epochs: float | None = 1.0
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    seed: int = 3407
    output_dir: str = "outputs/gemma-3n-critical-thinker"
    push_to_hub_repo: str | None = None

    # Data selection.
    dataset_split: str = "train"
    sampling: str = "random"
    quota_rounding: str = "largest_remainder"
    stream_shuffle_buffer: int = 10_000
    pool_factor: int = 4
    max_stream_rows: int | None = None
    max_token_length: int | None = None
    shuffle_seed: int = 42
    prepared_data_path: str | None = None

    # LoRA and optimiser settings that the original notebook hard-coded.
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    logging_steps: int = 1
    save_steps: int | None = None
    report_to: str = "none"
    hub_private: bool = True

    def __post_init__(self) -> None:
        _check_types(self)
        positive = (
            "max_seq_length",
            "lora_r",
            "lora_alpha",
            "total_samples",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "stream_shuffle_buffer",
            "pool_factor",
            "logging_steps",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("max_steps", "max_stream_rows", "max_token_length", "save_steps"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or null, got {value}")
        if self.num_train_epochs is not None and self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be > 0 or null, got {self.num_train_epochs}")
        if self.max_steps is None and self.num_train_epochs is None:
            raise ValueError("set max_steps or num_train_epochs (or both)")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.sampling not in SAMPLING_METHODS:
            raise ValueError(f"sampling must be one of {SAMPLING_METHODS}, got {self.sampling!r}")
        if self.quota_rounding not in QUOTA_ROUNDING:
            raise ValueError(
                f"quota_rounding must be one of {QUOTA_ROUNDING}, got {self.quota_rounding!r}"
            )
        validate_ratios(self.domain_ratios)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (for logging and run records)."""
        return asdict(self)


@dataclass
class EvalConfig:
    """Settings for scoring a model (optionally with a LoRA adapter) with Lighteval.

    ``tasks`` holds Lighteval task spec strings. In YAML, ``tasks`` may be a
    list, or a mapping of domain name to list. With a mapping, the lists are
    concatenated into ``tasks`` in file order and the grouping is kept in
    ``domains`` for the results summary.
    """

    model: str
    adapter: str | None
    tasks: list[str]
    max_samples: int | None = 100
    batch_size: int = 1
    load_in_4bit: bool = True
    output_dir: str = "results"
    domains: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_types(self)
        if not self.model:
            raise ValueError("model must be a non-empty string")
        if not self.tasks:
            raise ValueError("tasks must list at least one Lighteval task")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"max_samples must be > 0 or null, got {self.max_samples}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (for logging and run records)."""
        return asdict(self)


def validate_ratios(ratios: dict[str, float]) -> None:
    """Raise ``ValueError`` unless ``ratios`` is non-empty, non-negative and sums to 1.

    The sum may differ from 1 by at most ``RATIO_SUM_TOLERANCE``.
    """
    if not ratios:
        raise ValueError("domain_ratios must not be empty")
    for name, value in ratios.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"domain_ratios[{name!r}] must be a number, got {value!r}")
        if value < 0 or math.isnan(value):
            raise ValueError(f"domain_ratios[{name!r}] must be >= 0, got {value}")
    total = sum(ratios.values())
    if abs(total - 1.0) > RATIO_SUM_TOLERANCE:
        raise ValueError(
            f"domain_ratios must sum to 1 (within {RATIO_SUM_TOLERANCE}), got {total:.6f}"
        )


def load_train_config(path: str | Path) -> TrainConfig:
    """Read a YAML file into a :class:`TrainConfig`."""
    data = _read_yaml(path)
    _reject_unknown_keys(TrainConfig, data, path)
    return TrainConfig(**data)


def load_eval_config(path: str | Path) -> EvalConfig:
    """Read a YAML file into an :class:`EvalConfig`."""
    data = _read_yaml(path)
    _reject_unknown_keys(EvalConfig, data, path)
    data.setdefault("adapter", None)
    tasks = data.get("tasks")
    if isinstance(tasks, dict):
        domains: dict[str, list[str]] = {}
        for domain, specs in tasks.items():
            if not isinstance(specs, list):
                raise ValueError(f"{path}: tasks[{domain!r}] must be a list of task specs")
            domains[str(domain)] = list(specs)
        data["domains"] = domains
        data["tasks"] = [spec for specs in domains.values() for spec in specs]
    return EvalConfig(**data)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _reject_unknown_keys(cls: type, data: dict[str, Any], path: str | Path) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; allowed keys: {sorted(allowed)}")


def _check_types(obj: Any) -> None:
    """Check each dataclass field against its annotation.

    Ints are accepted (and converted) for float fields. Numeric strings are
    converted for float fields too, because PyYAML reads ``2e-4`` (no dot) as a
    string.
    """
    hints = typing.get_type_hints(type(obj))
    for f in fields(obj):
        expected = hints[f.name]
        value = getattr(obj, f.name)
        ok, converted = _matches(value, expected)
        if not ok:
            raise ValueError(
                f"{type(obj).__name__}.{f.name}: expected {_type_name(expected)}, got {value!r}"
            )
        if converted is not value:
            setattr(obj, f.name, converted)


def _matches(value: Any, expected: Any) -> tuple[bool, Any]:
    origin = typing.get_origin(expected)
    args = typing.get_args(expected)
    if origin is Union or origin is types.UnionType:
        for arg in args:
            ok, converted = _matches(value, arg)
            if ok:
                return True, converted
        return False, value
    if expected is type(None):
        return value is None, value
    if expected is bool:
        return isinstance(value, bool), value
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool), value
    if expected is float:
        if isinstance(value, bool):
            return False, value
        if isinstance(value, (int, float)):
            return True, float(value)
        if isinstance(value, str):
            try:
                return True, float(value)
            except ValueError:
                return False, value
        return False, value
    if expected is str:
        return isinstance(value, str), value
    if origin is list:
        if not isinstance(value, list):
            return False, value
        out = []
        for item in value:
            ok, converted = _matches(item, args[0])
            if not ok:
                return False, value
            out.append(converted)
        return True, out
    if origin is dict:
        if not isinstance(value, dict):
            return False, value
        out_d = {}
        for key, item in value.items():
            ok_k, key_c = _matches(key, args[0])
            ok_v, item_c = _matches(item, args[1])
            if not (ok_k and ok_v):
                return False, value
            out_d[key_c] = item_c
        return True, out_d
    return isinstance(value, expected), value


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)
