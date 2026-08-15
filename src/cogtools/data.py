"""Select a per-domain subset of OpenThoughts3 and convert it to Gemma chat format.

OpenThoughts3-1.2M is stored sorted by domain (code rows first, then math,
then science). The original notebook took the first N rows of each domain in
file order, so its subset came from three narrow row ranges. Here the default
is ``sampling="random"``: the stream is shuffled with a seeded buffer (which
also shuffles shard order) and rows are reservoir-sampled within each domain.
``sampling="first_n"`` reproduces the original behaviour.

Only the standard library is imported at module level. ``datasets`` and
``transformers`` are imported inside the functions that need them.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datasets

    from cogtools.config import TrainConfig

#: ShareGPT speaker names in OpenThoughts3 mapped to chat-template roles.
ROLE_MAP: dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
}


def compute_quotas(
    total: int,
    ratios: Mapping[str, float],
    method: str = "largest_remainder",
) -> dict[str, int]:
    """Split ``total`` rows across domains in proportion to ``ratios``.

    ``method="largest_remainder"`` normalises the ratios by their sum, floors
    each share, then gives the leftover rows to the domains with the largest
    fractional parts (ties go to the domain listed first). The quotas always
    sum to ``total``.

    ``method="floor"`` is ``int(total * ratio)`` per domain, as in the original
    notebook. With ratios 0.708/0.208/0.083 and total 2500 it gives
    1770/520/207 = 2497.
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if not ratios:
        raise ValueError("ratios must not be empty")
    if any(r < 0 for r in ratios.values()):
        raise ValueError(f"ratios must be >= 0, got {dict(ratios)}")

    if method == "floor":
        return {name: int(total * r) for name, r in ratios.items()}
    if method != "largest_remainder":
        raise ValueError(f"unknown quota method {method!r}")

    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("ratios must have a positive sum")
    exact = {name: total * r / ratio_sum for name, r in ratios.items()}
    quotas = {name: math.floor(x) for name, x in exact.items()}
    leftover = total - sum(quotas.values())
    order = list(ratios)
    by_remainder = sorted(order, key=lambda n: (-(exact[n] - quotas[n]), order.index(n)))
    for name in by_remainder[:leftover]:
        quotas[name] += 1
    return quotas


def stratified_sample(
    rows: Iterable[dict],
    quotas: Mapping[str, int],
    domain_key: str = "domain",
    *,
    seed: int = 42,
    method: str = "random",
    pool_factor: int | None = None,
    keep: Callable[[dict], bool] | None = None,
    max_rows: int | None = None,
) -> list[dict]:
    """Pick ``quotas[domain]`` rows per domain from ``rows``.

    Args:
        rows: Any iterable of dicts, including a streaming dataset.
        quotas: Rows wanted per domain. Rows from other domains are skipped.
        domain_key: Key that holds the domain name in each row.
        seed: Seed for the random choice. Same seed and same input order give
            the same output.
        method: ``"random"`` keeps a uniform random sample per domain of the
            rows read (reservoir sampling, Algorithm R). ``"first_n"`` keeps the
            first ``quota`` rows of each domain, as the original notebook did.
        pool_factor: Only for ``"random"``. Stop reading once every domain has
            had at least ``pool_factor * quota`` eligible rows, so the sample is
            uniform over that pool, not over the whole input. ``None`` reads
            everything.
        keep: Optional filter. Rows for which it returns False are ignored and
            do not count towards the pool.
        max_rows: Stop after reading this many rows in total, filled or not.

    Returns:
        The chosen rows, grouped by domain in the order of ``quotas``. Callers
        that train on the result should shuffle it.

    A ``UserWarning`` is issued for each domain that ends with fewer rows than
    its quota.
    """
    if method not in ("random", "first_n"):
        raise ValueError(f"unknown sampling method {method!r}")
    if pool_factor is not None and pool_factor < 1:
        raise ValueError(f"pool_factor must be >= 1, got {pool_factor}")

    wanted = {d: q for d, q in quotas.items() if q > 0}
    rng = random.Random(seed)
    chosen: dict[str, list[dict]] = {d: [] for d in wanted}
    seen: dict[str, int] = {d: 0 for d in wanted}
    if method == "random" and pool_factor is not None:
        targets = {d: q * pool_factor for d, q in wanted.items()}
    else:
        targets = dict(wanted)

    def done() -> bool:
        if method == "random" and pool_factor is None:
            return False
        return all(seen[d] >= targets[d] for d in wanted)

    if not wanted:
        return []

    for read, row in enumerate(rows, start=1):
        domain = row.get(domain_key)
        if domain in wanted and (keep is None or keep(row)):
            quota = wanted[domain]
            seen[domain] += 1
            bucket = chosen[domain]
            if method == "first_n":
                if len(bucket) < quota:
                    bucket.append(row)
            elif len(bucket) < quota:
                bucket.append(row)
            else:
                j = rng.randrange(seen[domain])
                if j < quota:
                    bucket[j] = row
            if done():
                break
        if max_rows is not None and read >= max_rows:
            break

    for domain, quota in wanted.items():
        if len(chosen[domain]) < quota:
            warnings.warn(
                f"domain {domain!r}: found {len(chosen[domain])} rows, quota was {quota}",
                stacklevel=2,
            )
    return [row for domain in wanted for row in chosen[domain]]


def to_gemma_chat(conversations: list[dict]) -> list[dict]:
    """Convert ShareGPT turns (``from``/``value``) to chat turns (``role``/``content``).

    ``human`` becomes ``user`` and ``gpt`` becomes ``assistant``; the Gemma
    chat template renders ``assistant`` as ``model``. Turns that already use
    ``role``/``content`` are passed through with the role normalised. The
    mapping is the same one ``unsloth.chat_templates.standardize_data_formats``
    applies.
    """
    out: list[dict] = []
    for i, turn in enumerate(conversations):
        if "from" in turn:
            speaker, text = turn["from"], turn.get("value")
        elif "role" in turn:
            speaker, text = turn["role"], turn.get("content")
        else:
            raise ValueError(f"turn {i} has neither 'from' nor 'role': {sorted(turn)}")
        role = ROLE_MAP.get(str(speaker).lower())
        if role is None:
            raise ValueError(f"turn {i}: unknown speaker {speaker!r}")
        if not isinstance(text, str):
            raise ValueError(f"turn {i}: content must be a string, got {type(text).__name__}")
        out.append({"role": role, "content": text})
    return out


def make_length_filter(
    max_tokens: int,
    count_tokens: Callable[[list[dict]], int],
    conversations_key: str = "conversations",
) -> Callable[[dict], bool]:
    """Return a ``keep`` predicate that drops rows longer than ``max_tokens``.

    ``count_tokens`` receives the row's turns in role/content form and returns
    a token count. Use :func:`chat_token_counter` to build one from a tokenizer.
    """

    def keep(row: dict) -> bool:
        return count_tokens(to_gemma_chat(row[conversations_key])) <= max_tokens

    return keep


def chat_token_counter(tokenizer: Any) -> Callable[[list[dict]], int]:
    """Count tokens of a conversation rendered with the tokenizer's chat template.

    Works with a plain tokenizer (returns a list of ids) and with a processor
    (returns a mapping whose ``input_ids`` may be batched).
    """

    def count(messages: list[dict]) -> int:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if len(ids) > 0 and not isinstance(ids[0], int):
            ids = ids[0]  # batched output: one conversation in, one row out
        return len(ids)

    return count


def domain_counts(rows: Iterable[Mapping[str, Any]], domain_key: str = "domain") -> dict[str, int]:
    """Count rows per domain."""
    return dict(Counter(row[domain_key] for row in rows))


def load_openthoughts_subset(cfg: TrainConfig, tokenizer: Any = None) -> datasets.Dataset:
    """Stream ``cfg.dataset_name`` and return a per-domain subset as a ``datasets.Dataset``.

    With ``cfg.sampling == "random"`` the stream is first shuffled with
    ``IterableDataset.shuffle(seed=cfg.shuffle_seed, buffer_size=cfg.stream_shuffle_buffer)``.
    That shuffles the order of the parquet shards and mixes rows within a
    buffer. Rows are then reservoir-sampled per domain until every domain has
    had ``cfg.pool_factor`` times its quota. Because each shard holds a single
    domain, the sample comes from the shards read first, which change with the
    seed; it is not uniform over the full 1.2M rows.

    With ``cfg.sampling == "first_n"`` the stream is read in file order and the
    first rows of each domain are kept, as in the original run. That reads
    about 1.1M rows (most of the 28 GB download) before science fills.

    If ``cfg.max_token_length`` is set, rows whose chat-formatted length is
    longer are skipped. ``tokenizer`` is used for counting; if it is None, the
    tokenizer of ``cfg.model_name`` is loaded with ``transformers``.

    The result is shuffled with ``cfg.shuffle_seed``.
    """
    from datasets import Dataset, load_dataset

    stream = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)
    if cfg.sampling == "random":
        stream = stream.shuffle(seed=cfg.shuffle_seed, buffer_size=cfg.stream_shuffle_buffer)

    keep = None
    if cfg.max_token_length is not None:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        keep = make_length_filter(cfg.max_token_length, chat_token_counter(tokenizer))

    quotas = compute_quotas(cfg.total_samples, cfg.domain_ratios, method=cfg.quota_rounding)
    rows = stratified_sample(
        stream,
        quotas,
        seed=cfg.shuffle_seed,
        method=cfg.sampling,
        pool_factor=cfg.pool_factor if cfg.sampling == "random" else None,
        keep=keep,
        max_rows=cfg.max_stream_rows,
    )
    if not rows:
        raise RuntimeError("no rows selected; check domain_ratios, filters and max_stream_rows")
    return Dataset.from_list(rows).shuffle(seed=cfg.shuffle_seed)


def save_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write rows to ``.jsonl`` or ``.parquet`` (chosen by the file suffix).

    Parquet needs ``datasets`` installed. Returns the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    elif suffix == ".parquet":
        from datasets import Dataset

        data = rows if hasattr(rows, "to_parquet") else Dataset.from_list([dict(r) for r in rows])
        data.to_parquet(str(path))
    else:
        raise ValueError(f"unsupported file type {suffix!r}; use .jsonl or .parquet")
    return path


def load_prepared(path: str | Path) -> datasets.Dataset:
    """Load a subset written by :func:`save_rows` as a ``datasets.Dataset``."""
    from datasets import load_dataset

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        builder = "json"
    elif suffix == ".parquet":
        builder = "parquet"
    else:
        raise ValueError(f"unsupported file type {suffix!r}; use .jsonl or .parquet")
    return load_dataset(builder, data_files=str(path), split="train")
