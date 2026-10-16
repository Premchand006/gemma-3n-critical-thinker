"""Score the base model, with or without the LoRA adapter, using lighteval 0.11.0.

The model is loaded here instead of through lighteval's own loader, for two
reasons found in the lighteval 0.11.0 source:

* ``AdapterModelConfig`` reads ``self.config.pretrained``, a field that does not
  exist on the config, and the ``adapters`` extra pins ``peft==0.3.0``.
* ``dtype="4bit"`` hard-codes the bitsandbytes compute dtype to float16.

So the base model is loaded in 4-bit with ``BitsAndBytesConfig``, the adapter is
attached with ``PeftModel.from_pretrained`` (not merged, since merging into a
4-bit base is lossy), and the result is wrapped with
``TransformersModel.from_model``.

torch, transformers, peft and lighteval are imported inside functions so this
module can be imported on a CPU-only machine without them.

Untested: the full path (Gemma 3n in 4-bit, PeftModel, lighteval 0.11.0 on a
T4) has not been run end to end.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cogtools.config import EvalConfig

logger = logging.getLogger(__name__)

# lighteval 0.11.0 reads max_length from the top-level model config. Gemma 3n
# keeps it in the nested text config, so lighteval falls back to 2048 tokens,
# which truncates 25-shot ARC and 5-shot MMLU prompts. Set it explicitly.
DEFAULT_MAX_LENGTH = 4096

# The T4 has no bfloat16 support, so float16 is the default compute dtype.
# Gemma models can overflow in float16 (NaN logits or garbage text). If that
# happens, try "float32" (slower, more memory) or "bfloat16" on an A100/L4.
DEFAULT_COMPUTE_DTYPE = "float16"

_VERSION_PACKAGES = ("lighteval", "transformers", "torch", "peft", "bitsandbytes", "accelerate")


def package_versions() -> dict[str, str | None]:
    """Return installed versions of the packages that affect scores."""
    versions: dict[str, str | None] = {}
    for name in _VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_dtype(name: str) -> Any:
    import torch

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in dtypes:
        raise ValueError(f"compute_dtype must be one of {sorted(dtypes)}, got {name!r}")
    return dtypes[name]


def build_model(
    cfg: EvalConfig,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE,
) -> Any:
    """Load the base model (4-bit if ``cfg.load_in_4bit``), attach the adapter,
    and return a lighteval ``TransformersModel``.

    Args:
        cfg: eval config. Uses ``model``, ``adapter``, ``batch_size`` and ``load_in_4bit``.
        max_length: context length lighteval uses to truncate prompts.
        compute_dtype: "float16", "bfloat16" or "float32".
    """
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.transformers.transformers_model import (
        TransformersModel,
        TransformersModelConfig,
    )
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = _torch_dtype(compute_dtype)
    load_kwargs: dict[str, Any] = {"device_map": {"": 0}, "torch_dtype": dtype}
    if cfg.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    # In transformers 4.57.0 AutoModelForCausalLM maps gemma3n to
    # Gemma3nForConditionalGeneration, so the audio and vision towers are loaded too.
    logger.info("Loading base model %s (4-bit=%s)", cfg.model, cfg.load_in_4bit)
    hf_model = AutoModelForCausalLM.from_pretrained(cfg.model, **load_kwargs)

    if cfg.adapter:
        from peft import PeftModel

        logger.info("Attaching LoRA adapter %s (not merged)", cfg.adapter)
        hf_model = PeftModel.from_pretrained(hf_model, cfg.adapter)
    hf_model.eval()

    model_config = TransformersModelConfig(
        model_name=cfg.model,  # tokenizer is loaded from the base model id
        batch_size=cfg.batch_size,
        max_length=max_length,
        # temperature=0 means greedy decoding. Sampling metrics with more than
        # one sample raise an error at this temperature (see eval_tasks.yaml).
        generation_parameters=GenerationParameters(temperature=0.0),
    )
    # No accelerator is passed, so lighteval does not call .to(device) on the
    # 4-bit model.
    return TransformersModel.from_model(hf_model, model_config)


def run_lighteval(
    cfg: EvalConfig,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE,
) -> dict:
    """Run the configured lighteval tasks and save the results.

    Writes lighteval's own output under ``<output_dir>/<timestamp>/lighteval/``
    and a combined record to ``<output_dir>/<timestamp>/results.json`` with the
    model, adapter, package versions, task list, max_samples, timestamp and the
    dict returned by ``Pipeline.get_results()``.

    Returns:
        The combined record that was written to results.json.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    if not cfg.tasks:
        raise ValueError("EvalConfig.tasks is empty")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, max_length=max_length, compute_dtype=compute_dtype)

    tracker = EvaluationTracker(output_dir=str(run_dir / "lighteval"), save_details=True)
    params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=cfg.max_samples,
    )
    pipeline = Pipeline(
        tasks=",".join(cfg.tasks),
        pipeline_parameters=params,
        evaluation_tracker=tracker,
        model=model,
    )

    # evaluate() returns None in lighteval 0.11.0; results come from get_results().
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()
    lighteval_results = pipeline.get_results()

    record: dict[str, Any] = {
        "model": cfg.model,
        "adapter": cfg.adapter,
        "tasks": list(cfg.tasks),
        "max_samples": cfg.max_samples,
        "batch_size": cfg.batch_size,
        "load_in_4bit": cfg.load_in_4bit,
        "max_length": max_length,
        "compute_dtype": compute_dtype,
        "timestamp": timestamp,
        "versions": package_versions(),
        "config": dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else None,
        "lighteval_results": lighteval_results,
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    # Round-trip through JSON so the returned dict matches the file contents.
    return json.loads(out_path.read_text(encoding="utf-8"))
