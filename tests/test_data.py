from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cogtools.data import (
    compute_quotas,
    domain_counts,
    make_length_filter,
    save_rows,
    stratified_sample,
    to_gemma_chat,
)


def make_rows(per_domain: int = 200) -> list[dict]:
    """Rows sorted by domain, like OpenThoughts3 (code, then math, then science)."""
    rows = []
    for domain in ("code", "math", "science"):
        for i in range(per_domain):
            rows.append(
                {
                    "domain": domain,
                    "idx": i,
                    "conversations": [
                        {"from": "human", "value": f"{domain} question {i}"},
                        {"from": "gpt", "value": "x" * i},
                    ],
                }
            )
    return rows


RATIOS = {"math": 0.708, "code": 0.208, "science": 0.083}


@pytest.mark.parametrize("total", [0, 1, 7, 100, 2500, 9999])
def test_quotas_sum_to_total(total: int) -> None:
    quotas = compute_quotas(total, RATIOS)
    assert sum(quotas.values()) == total
    assert all(q >= 0 for q in quotas.values())


def test_quotas_largest_remainder() -> None:
    # Exact shares 3.33.., 3.33.., 3.33..: floors give 9, the first domain gets the extra row.
    assert compute_quotas(10, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == {"a": 4, "b": 3, "c": 3}
    # Shares 5.5 / 3.0 / 1.5 -> the 0.5 remainders tie; listed order breaks the tie.
    assert compute_quotas(10, {"a": 0.55, "b": 0.30, "c": 0.15}) == {"a": 6, "b": 3, "c": 1}


def test_quotas_floor_matches_original_notebook() -> None:
    quotas = compute_quotas(2500, RATIOS, method="floor")
    assert quotas == {"math": 1770, "code": 520, "science": 207}
    assert sum(quotas.values()) == 2497


def test_sample_meets_quotas() -> None:
    quotas = {"math": 30, "code": 10, "science": 5}
    picked = stratified_sample(make_rows(), quotas, seed=1)
    assert domain_counts(picked) == quotas


def test_random_sample_is_not_first_n() -> None:
    quotas = {"math": 30, "code": 10, "science": 5}
    picked = stratified_sample(make_rows(), quotas, seed=1)
    math_idx = sorted(r["idx"] for r in picked if r["domain"] == "math")
    assert math_idx != list(range(30))
    assert max(math_idx) >= 30


def test_random_sample_depends_on_seed() -> None:
    quotas = {"math": 30, "code": 10, "science": 5}
    rows = make_rows()
    a = stratified_sample(rows, quotas, seed=1)
    b = stratified_sample(rows, quotas, seed=1)
    c = stratified_sample(rows, quotas, seed=2)
    ids = [[(r["domain"], r["idx"]) for r in s] for s in (a, b, c)]
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_random_sample_covers_the_pool() -> None:
    # Over many seeds every math row should be picked at least once.
    rows = make_rows(per_domain=50)
    hit: set[int] = set()
    for seed in range(200):
        picked = stratified_sample(rows, {"math": 5}, seed=seed)
        hit.update(r["idx"] for r in picked)
    assert hit == set(range(50))


def test_first_n_reproduces_original() -> None:
    quotas = {"math": 3, "code": 2, "science": 1}
    picked = stratified_sample(make_rows(), quotas, method="first_n")
    got = {(r["domain"], r["idx"]) for r in picked}
    assert got == {("math", 0), ("math", 1), ("math", 2), ("code", 0), ("code", 1), ("science", 0)}


def test_pool_factor_stops_early() -> None:
    read = 0

    def counting(rows):
        nonlocal read
        for row in rows:
            read += 1
            yield row

    # Interleave domains (code 0, math 0, science 0, code 1, ...) so all fill early.
    base = make_rows()
    rows = [r for i in range(200) for r in base[i::200]]
    stratified_sample(counting(rows), {"math": 2, "code": 2, "science": 2}, pool_factor=3)
    assert read < len(rows)


def test_keep_filter_and_shortfall_warning() -> None:
    keep = make_length_filter(10, lambda msgs: len(msgs[-1]["content"]))
    with pytest.warns(UserWarning, match="science"):
        picked = stratified_sample(make_rows(), {"science": 20}, keep=keep, seed=0)
    assert len(picked) == 11  # answers of length 0..10
    assert all(len(r["conversations"][1]["value"]) <= 10 for r in picked)


def test_other_domains_and_zero_quotas_ignored() -> None:
    picked = stratified_sample(make_rows(), {"math": 4, "code": 0}, seed=0)
    assert domain_counts(picked) == {"math": 4}


def test_to_gemma_chat() -> None:
    convo = [
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "<think>2+2=4</think>\n4"},
    ]
    assert to_gemma_chat(convo) == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>\n4"},
    ]


def test_to_gemma_chat_passes_role_content_through() -> None:
    convo = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert to_gemma_chat(convo) == convo


def test_to_gemma_chat_rejects_unknown_speaker() -> None:
    with pytest.raises(ValueError, match="unknown speaker"):
        to_gemma_chat([{"from": "bot", "value": "hi"}])


def test_save_rows_jsonl(tmp_path: Path) -> None:
    rows = make_rows(per_domain=2)
    path = save_rows(rows, tmp_path / "sub" / "rows.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows


def test_no_heavy_imports() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys, cogtools.config, cogtools.data, cogtools.train;"
        "heavy = ('torch', 'unsloth', 'datasets', 'transformers', 'trl');"
        "print(','.join(m for m in heavy if m in sys.modules))"
    )
    env = {**os.environ, "PYTHONPATH": str(src)}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
