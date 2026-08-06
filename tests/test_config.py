from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from cogtools.config import (
    EvalConfig,
    TrainConfig,
    load_eval_config,
    load_train_config,
    validate_ratios,
)

REPO = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_valid() -> None:
    cfg = TrainConfig()
    assert cfg.model_name == "unsloth/gemma-3n-E4B-it"
    assert abs(sum(cfg.domain_ratios.values()) - 1.0) < 1e-9


def test_original_config_matches_the_run() -> None:
    cfg = load_train_config(REPO / "configs" / "train_gemma3n_e4b_original.yaml")
    assert cfg.max_steps == 60
    assert cfg.max_seq_length == 1024
    assert (cfg.lora_r, cfg.lora_alpha) == (8, 8)
    assert cfg.learning_rate == pytest.approx(2e-4)
    assert cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 4
    assert cfg.sampling == "first_n"
    assert cfg.seed == 3407
    assert cfg.shuffle_seed == 42


def test_recommended_config_loads() -> None:
    cfg = load_train_config(REPO / "configs" / "train_gemma3n_e4b.yaml")
    assert cfg.sampling == "random"
    assert cfg.max_steps is None
    assert cfg.num_train_epochs == 1.0
    assert cfg.max_token_length is not None


def test_ratios_must_sum_to_one(tmp_path: Path) -> None:
    path = write(tmp_path, "domain_ratios: {math: 0.5, code: 0.2}\n")
    with pytest.raises(ValueError, match="sum to 1"):
        load_train_config(path)


def test_negative_ratio_rejected() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        validate_ratios({"math": 1.2, "code": -0.2})


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "lora_rank: 8\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_train_config(path)


def test_wrong_type_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "max_seq_length: '1024'\n")
    with pytest.raises(ValueError, match="max_seq_length"):
        load_train_config(path)


def test_float_written_without_dot_is_accepted(tmp_path: Path) -> None:
    # PyYAML reads 2e-4 as a string; the loader converts it.
    path = write(tmp_path, "learning_rate: 2e-4\n")
    assert load_train_config(path).learning_rate == pytest.approx(2e-4)


def test_needs_max_steps_or_epochs() -> None:
    with pytest.raises(ValueError, match="max_steps or num_train_epochs"):
        TrainConfig(max_steps=None, num_train_epochs=None)


def test_bad_sampling_rejected() -> None:
    with pytest.raises(ValueError, match="sampling"):
        TrainConfig(sampling="first")


def test_replace_revalidates() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(TrainConfig(), lora_r=0)


def test_eval_config_list(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "model: unsloth/gemma-3n-E4B-it\ntasks: ['lighteval|gsm8k|0']\nmax_samples: null\n",
    )
    cfg = load_eval_config(path)
    assert cfg.adapter is None
    assert cfg.tasks == ["lighteval|gsm8k|0"]
    assert cfg.max_samples is None
    assert cfg.domains == {}


def test_eval_config_grouped_by_domain(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "model: m\nadapter: a\ntasks:\n  math: [t1, t2]\n  code: [t3]\n",
    )
    cfg = load_eval_config(path)
    assert cfg.tasks == ["t1", "t2", "t3"]
    assert cfg.domains == {"math": ["t1", "t2"], "code": ["t3"]}


def test_eval_config_requires_tasks() -> None:
    with pytest.raises(ValueError, match="tasks"):
        EvalConfig(model="m", adapter=None, tasks=[])
