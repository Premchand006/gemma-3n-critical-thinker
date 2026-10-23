"""Turn lighteval results into a markdown table with per-domain means.

Every number in the output comes from the results dict passed in. Tasks that
have no entry in the results are shown as "not run" and are left out of the
means.

Accepted input shapes:

* the dict from ``Pipeline.get_results()`` in lighteval 0.11.0, which has a
  ``"results"`` key mapping full task names (``"leaderboard|gsm8k|5"``) to
  ``{metric_name: value, metric_name + "_stderr": value}``, plus an ``"all"`` entry;
* the record written by ``cogtools.eval.lighteval_runner`` (that dict under
  ``"lighteval_results"``, plus run metadata).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

NOT_RUN = "not run"

# Domains with no lighteval task, and the reason shown in the summary.
UNAVAILABLE_DOMAINS: dict[str, str] = {
    "code": "not available in lighteval 0.11 (no HumanEval or MBPP task)",
}

# Main metric per task family, in order of preference. arc and hellaswag use
# acc_norm and truthfulqa uses mc2, following the Open LLM Leaderboard.
PREFERRED_METRICS: dict[str, tuple[str, ...]] = {
    "arc": ("acc_norm", "acc"),
    "hellaswag": ("acc_norm", "acc"),
    "truthfulqa": ("truthfulqa_mc2", "truthfulqa_mc1"),
    "gsm8k": ("qem", "em", "extractive_match", "exact_match"),
    "math": ("qem", "em", "extractive_match", "exact_match"),
    "mmlu": ("acc", "acc_norm", "em"),
    "winogrande": ("acc", "acc_norm"),
}
FALLBACK_METRICS: tuple[str, ...] = (
    "acc_norm",
    "acc",
    "qem",
    "em",
    "extractive_match",
    "exact_match",
    "truthfulqa_mc2",
    "pass@1",
)


@dataclass
class TaskScore:
    spec: str
    metric: str
    value: float
    stderr: float | None = None
    n_subtasks: int = 1


def _parse_spec(spec: str) -> tuple[str, str, str | None]:
    """Split ``suite|task|fewshot`` into parts. The task keeps any ``@k=1`` suffix."""
    parts = spec.split("|")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None
    return "", parts[0], None


def _strip_params(task: str) -> str:
    """``math:algebra@k=1`` -> ``math:algebra``."""
    return task.split("@", 1)[0]


def _family(task: str) -> str:
    """``mmlu:college_mathematics`` -> ``mmlu``; ``arc:challenge`` -> ``arc``."""
    return _strip_params(task).split(":", 1)[0]


def _same_task(spec: str, key: str) -> bool:
    s_suite, s_task, s_fs = _parse_spec(spec)
    k_suite, k_task, k_fs = _parse_spec(key)
    if s_suite != k_suite or _strip_params(s_task) != _strip_params(k_task):
        return False
    return s_fs is None or k_fs is None or s_fs == k_fs


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _main_metric(task: str, metrics: dict[str, Any]) -> tuple[str, float, float | None] | None:
    """Pick the main metric from one task's metric dict."""
    order = PREFERRED_METRICS.get(_family(task), ()) + FALLBACK_METRICS
    names = [n for n in order if _is_number(metrics.get(n))]
    if not names:
        names = sorted(n for n, v in metrics.items() if not n.endswith("_stderr") and _is_number(v))
    if not names:
        return None
    name = names[0]
    stderr = metrics.get(f"{name}_stderr")
    return name, float(metrics[name]), float(stderr) if _is_number(stderr) else None


def extract_task_results(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the ``{task_name: {metric: value}}`` mapping from either input shape."""
    if not isinstance(results, dict):
        return {}
    inner = results.get("lighteval_results")
    source = inner if isinstance(inner, dict) else results
    task_results = source.get("results")
    if not isinstance(task_results, dict):
        return {}
    return {k: v for k, v in task_results.items() if k != "all" and isinstance(v, dict)}


def find_task_score(spec: str, task_results: dict[str, dict[str, Any]]) -> TaskScore | None:
    """Look up one task spec in the results.

    Tries an exact key match, then a match that ignores ``@`` metric parameters
    and a missing few-shot field. For a superset spec such as
    ``leaderboard|mmlu|5`` it uses the ``mmlu:_average`` entry if lighteval wrote
    one, else the mean of the matching subtask scores.
    """
    candidates = [spec] if spec in task_results else []
    if not candidates:
        candidates = [k for k in task_results if _same_task(spec, k)]
    for key in candidates:
        picked = _main_metric(_parse_spec(key)[1], task_results[key])
        if picked is not None:
            name, value, stderr = picked
            return TaskScore(spec=spec, metric=name, value=value, stderr=stderr)

    suite, task, fewshot = _parse_spec(spec)
    base = _strip_params(task)
    if ":" in base:
        return None

    subtasks: list[TaskScore] = []
    for key, metrics in task_results.items():
        k_suite, k_task, k_fs = _parse_spec(key)
        k_base = _strip_params(k_task)
        if k_suite != suite or not k_base.startswith(base + ":"):
            continue
        if fewshot is not None and k_fs is not None and k_fs != fewshot:
            continue
        picked = _main_metric(k_task, metrics)
        if picked is None:
            continue
        name, value, stderr = picked
        if k_base == f"{base}:_average":
            return TaskScore(spec=spec, metric=name, value=value, stderr=stderr)
        subtasks.append(TaskScore(spec=key, metric=name, value=value, stderr=stderr))

    if not subtasks:
        return None
    metric_names = sorted({s.metric for s in subtasks})
    return TaskScore(
        spec=spec,
        metric="/".join(metric_names),
        value=sum(s.value for s in subtasks) / len(subtasks),
        stderr=None,
        n_subtasks=len(subtasks),
    )


def _code(spec: str) -> str:
    # Escape pipes so the spec does not split the markdown table cell.
    return "`" + spec.replace("|", "\\|") + "`"


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: dict, domains: dict[str, list[str]]) -> str:
    """Build a markdown summary of lighteval results grouped by domain.

    Args:
        results: lighteval results dict, or the record written by ``run_lighteval``.
        domains: domain name -> list of task specs, as in ``configs/eval_domains.yaml``.

    Returns:
        Markdown with a per-task table, a per-domain table of means, and the
        overall mean over all distinct tasks that have a score. Means are
        unweighted averages of task scores; a superset task such as MMLU counts
        as one task.
    """
    task_results = extract_task_results(results)
    lines: list[str] = ["## Lighteval results", ""]

    meta = results if isinstance(results, dict) else {}
    if "model" in meta:
        lines.append(f"Model: `{meta.get('model')}`. Adapter: `{meta.get('adapter') or 'none'}`.")
    if meta.get("max_samples") is not None:
        lines.append(
            f"max_samples={meta['max_samples']} per task: these are partial scores "
            "and are not comparable to full-test-set numbers."
        )
    if len(lines) > 2:
        lines.append("")

    lines += ["| Domain | Task | Metric | Score | Stderr |", "|---|---|---|---|---|"]
    scored: dict[str, TaskScore] = {}
    domain_rows: list[str] = []
    for domain, specs in domains.items():
        specs = list(specs or [])
        if not specs:
            reason = UNAVAILABLE_DOMAINS.get(domain, "no tasks configured")
            lines.append(f"| {domain} | none | | {reason} | |")
            domain_rows.append(f"| {domain} | 0/0 | {reason} |")
            continue
        values: list[float] = []
        for spec in specs:
            score = find_task_score(spec, task_results)
            if score is None:
                lines.append(f"| {domain} | {_code(spec)} | | {NOT_RUN} | |")
                continue
            metric = score.metric
            if score.n_subtasks > 1:
                metric += f" (mean of {score.n_subtasks} subtasks)"
            value, stderr = _fmt(score.value), _fmt(score.stderr)
            lines.append(f"| {domain} | {_code(spec)} | {metric} | {value} | {stderr} |")
            values.append(score.value)
            scored[spec] = score
        mean = _mean(values)
        mean_text = NOT_RUN if mean is None else _fmt(mean)
        domain_rows.append(f"| {domain} | {len(values)}/{len(specs)} | {mean_text} |")

    lines += ["", "| Domain | Tasks scored | Mean score |", "|---|---|---|", *domain_rows, ""]

    overall = _mean([s.value for s in scored.values()])
    if overall is None:
        lines.append("Overall: no task scores found in the results.")
    else:
        lines.append(f"Overall mean over {len(scored)} tasks: {_fmt(overall)}")
    lines.append("")
    return "\n".join(lines)
