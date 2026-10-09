import json
from pathlib import Path

import pytest

from cogtools.eval.agent_eval import (
    is_correct,
    load_problems,
    normalize_answer,
    run_agent_eval,
)
from cogtools.tools import CognitiveToolsAgent, ScriptedBackend

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_math.jsonl"


@pytest.mark.parametrize(
    ("pred", "ref"),
    [
        ("14", "14"),
        ("$14$", "14"),
        ("\\boxed{14}", "14"),
        ("x = 6", "6"),
        ("1,000", "1000"),
        ("0.5", "\\frac{1}{2}"),
        ("3/8", "\\dfrac{3}{8}"),
        ("42.", "42"),
        ("\\text{(B)}", "(b)"),
    ],
)
def test_normalize_matches(pred, ref):
    assert is_correct(pred, ref)


def test_normalize_mismatch():
    assert not is_correct("15", "14")
    assert not is_correct(None, "14")
    assert normalize_answer("2\\sqrt{2}") != normalize_answer("\\sqrt{8}")


def test_sample_file_loads():
    problems = load_problems(SAMPLE)
    assert 5 <= len(problems) <= 10
    assert all({"id", "question", "answer"} <= set(p) for p in problems)
    assert len(load_problems(SAMPLE, limit=2)) == 2


def test_run_agent_eval_writes_traces_and_summary(tmp_path):
    problems = [
        {"id": "a", "question": "1+1?", "answer": "2"},
        {"id": "b", "question": "2+2?", "answer": "4"},
        {"id": "c", "question": "3+3?", "answer": "6"},
    ]
    backend = ScriptedBackend(
        [
            # a: one tool call, then correct
            '```python\nunderstand_question("1+1?")\n```',
            "Add the two numbers.",
            "ANSWER: 2",
            # b: wrong answer
            "ANSWER: 5",
            # c: boxed answer, no ANSWER line
            "It is \\boxed{6}.",
        ]
    )
    agent = CognitiveToolsAgent(backend, max_turns=3)
    summary = run_agent_eval(agent, problems, tmp_path, run_config={"model": "scripted"})

    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["id"] for r in records] == ["a", "b", "c"]
    assert [r["correct"] for r in records] == [True, False, True]
    assert records[0]["n_tool_calls"] == 1
    assert records[0]["trace"]["steps"][1]["name"] == "understand_question"

    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk == summary
    assert summary["n_problems"] == 3
    assert summary["n_correct"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["tool_calls"] == {"understand_question": 1}
    assert summary["stopped_reasons"] == {"answer": 2, "no_tool_call": 1}
    assert summary["config"] == {"model": "scripted"}


def test_empty_problem_set(tmp_path):
    summary = run_agent_eval(CognitiveToolsAgent(ScriptedBackend([])), [], tmp_path)
    assert summary["n_problems"] == 0
    assert summary["accuracy"] is None
