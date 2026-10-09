"""Run the cognitive-tools agent on a math problem set and score exact match.

Input is a JSONL file with one object per line: ``{"id", "question",
"answer"}``. For each problem the agent's final answer (the ``ANSWER:`` line,
or the last ``\\boxed{}`` when there is none) is normalised and compared with
the normalised reference answer. Normalisation is deliberately small: it strips
``$``, ``\\boxed{}``, ``\\text{}``, spacing commands, a leading ``x =``, a
trailing period, thousands separators and case, and compares numbers as exact
fractions (so ``0.5``, ``1/2`` and ``\\frac{1}{2}`` match). It does not check
symbolic equivalence, so answers such as ``2\\sqrt{2}`` and ``\\sqrt{8}`` count
as different.

Outputs, in ``output_dir``:

* ``traces.jsonl``: one line per problem with the reference, the prediction,
  the score and the full trace.
* ``summary.json``: counts and accuracy computed from those lines.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from cogtools.tools.orchestrator import extract_boxed, extract_final_answer
from cogtools.tools.trace import Trace


class Solver(Protocol):
    def solve(self, question: str) -> Trace: ...


def load_problems(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read ``{"id", "question", "answer"}`` records from a JSONL file."""
    problems: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [k for k in ("question", "answer") if k not in record]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing field(s) {missing}")
            record.setdefault("id", str(lineno))
            problems.append(record)
            if limit is not None and len(problems) >= limit:
                break
    return problems


_TEXT_CMD_RE = re.compile(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\{([^{}]*)\}")
_FRAC_RE = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
_FRAC_SHORT_RE = re.compile(r"\\[dt]?frac(\d)(\d)")
_THOUSANDS_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")


def _as_fraction(s: str) -> Fraction | None:
    if _NUMBER_RE.fullmatch(s):
        return Fraction(s)
    if "/" in s:
        num, _, den = s.partition("/")
        num, den = num.strip("()"), den.strip("()")
        if _NUMBER_RE.fullmatch(num) and _NUMBER_RE.fullmatch(den) and Fraction(den) != 0:
            return Fraction(num) / Fraction(den)
    return None


def normalize_answer(answer: object) -> str:
    """Normalise an answer string for exact-match comparison."""
    s = str(answer).strip()
    boxed = extract_boxed(s)
    if boxed is not None:
        s = boxed
    s = s.strip().strip("$").strip()
    for cmd in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\displaystyle"):
        s = s.replace(cmd, "")
    s = _TEXT_CMD_RE.sub(r"\1", s)
    s = _FRAC_RE.sub(r"(\1)/(\2)", s)
    s = _FRAC_SHORT_RE.sub(r"\1/\2", s)
    s = re.sub(r"\((-?\w+)\)", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\u00b0", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.rstrip(".").strip()
    s = re.sub(r"^[A-Za-z]\s*=\s*", "", s)
    if _THOUSANDS_RE.fullmatch(s):
        s = s.replace(",", "")
    s = re.sub(r"\s+", "", s).lower()
    value = _as_fraction(s)
    if value is not None:
        return str(value)
    return s


def is_correct(predicted: str | None, reference: object) -> bool:
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(reference)


def predicted_answer(trace: Trace) -> str | None:
    """The trace's final answer, or an answer found in its last text step."""
    if trace.final_answer is not None:
        return trace.final_answer
    for step in reversed(trace.steps):
        if isinstance(step, str):
            return extract_final_answer(step)
    return None


def score_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the summary dict from per-problem records (as written to JSONL)."""
    records = list(records)
    n = len(records)
    n_correct = sum(1 for r in records if r["correct"])
    stopped = Counter(r["stopped_reason"] for r in records)
    calls: Counter[str] = Counter()
    errors = 0
    for r in records:
        for step in r["trace"]["steps"]:
            if step.get("type") == "tool_call":
                calls[step["name"]] += 1
                if step.get("error"):
                    errors += 1
    return {
        "n_problems": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else None,
        "n_without_answer": sum(1 for r in records if r["predicted"] is None),
        "stopped_reasons": dict(stopped),
        "tool_calls": dict(calls),
        "tool_call_errors": errors,
        "mean_tool_calls_per_problem": (sum(calls.values()) / n) if n else None,
        "problems_with_a_tool_call": sum(
            1 for r in records if any(s.get("type") == "tool_call" for s in r["trace"]["steps"])
        ),
    }


def run_agent_eval(
    agent: Solver,
    problems: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve every problem, write ``traces.jsonl`` and ``summary.json``.

    ``run_config`` (model id, adapter, tools, max_turns, data path, ...) is
    copied into the summary so a result file says how it was produced.
    Returns the summary dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces_path = out / "traces.jsonl"
    records: list[dict[str, Any]] = []
    started = time.time()
    with open(traces_path, "w", encoding="utf-8") as f:
        for problem in problems:
            t0 = time.time()
            trace = agent.solve(str(problem["question"]))
            predicted = predicted_answer(trace)
            record = {
                "id": problem.get("id"),
                "question": problem["question"],
                "reference": problem["answer"],
                "predicted": predicted,
                "correct": is_correct(predicted, problem["answer"]),
                "stopped_reason": trace.stopped_reason,
                "n_tool_calls": len(trace.tool_calls),
                "seconds": round(time.time() - t0, 3),
                "trace": trace.to_dict(),
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    summary = score_records(records)
    summary["seconds_total"] = round(time.time() - started, 3)
    summary["traces_file"] = traces_path.name
    summary["config"] = dict(run_config or {})
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "is_correct",
    "load_problems",
    "normalize_answer",
    "predicted_answer",
    "run_agent_eval",
    "score_records",
]
