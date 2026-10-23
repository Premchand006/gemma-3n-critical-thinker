"""Tests for cogtools.eval.summarize. All scores below are made-up fixture values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cogtools.eval.summarize import NOT_RUN, extract_task_results, find_task_score, summarize

DOMAINS = {
    "math": ["leaderboard|gsm8k|5", "lighteval|math:algebra@k=1|4"],
    "science": ["leaderboard|arc:challenge|25", "leaderboard|mmlu:high_school_physics|5"],
    "general": ["leaderboard|mmlu|5", "leaderboard|truthfulqa:mc|0"],
    "code": [],
}

# Shape of Pipeline.get_results() in lighteval 0.11.0 (only the parts we read).
FAKE_RESULTS = {
    "config_general": {"model_name": "fake"},
    "results": {
        "leaderboard|gsm8k|5": {"qem": 0.4, "qem_stderr": 0.05},
        "lighteval|math:algebra|4": {"em": 0.2, "em_stderr": 0.04, "maj@k": 0.2},
        "leaderboard|arc:challenge|25": {
            "acc": 0.5,
            "acc_stderr": 0.05,
            "acc_norm": 0.6,
            "acc_norm_stderr": 0.05,
        },
        "leaderboard|mmlu:abstract_algebra|5": {"acc": 0.3, "acc_stderr": 0.04},
        "leaderboard|mmlu:anatomy|5": {"acc": 0.5, "acc_stderr": 0.04},
        "leaderboard|truthfulqa:mc|0": {"truthfulqa_mc1": 0.3, "truthfulqa_mc2": 0.45},
        "all": {"acc": 0.9},
    },
    "versions": {},
}


def test_main_metric_choice():
    tr = extract_task_results(FAKE_RESULTS)
    assert find_task_score("leaderboard|arc:challenge|25", tr).metric == "acc_norm"
    assert find_task_score("leaderboard|truthfulqa:mc|0", tr).value == pytest.approx(0.45)
    # @k=1 parameter is ignored when matching the result key
    assert find_task_score("lighteval|math:algebra@k=1|4", tr).metric == "em"


def test_superset_uses_mean_of_subtasks():
    score = find_task_score("leaderboard|mmlu|5", extract_task_results(FAKE_RESULTS))
    assert score.value == pytest.approx(0.4)
    assert score.n_subtasks == 2


def test_superset_prefers_average_entry():
    results = {"results": {**FAKE_RESULTS["results"], "leaderboard|mmlu:_average|5": {"acc": 0.77}}}
    score = find_task_score("leaderboard|mmlu|5", extract_task_results(results))
    assert score.value == pytest.approx(0.77)


def test_domain_and_overall_means():
    text = summarize(FAKE_RESULTS, DOMAINS)
    # math: (0.4 + 0.2) / 2
    assert "| math | 2/2 | 0.3000 |" in text
    # science: arc 0.6 scored, physics missing
    assert "| science | 1/2 | 0.6000 |" in text
    # general: (0.4 + 0.45) / 2
    assert "| general | 2/2 | 0.4250 |" in text
    # overall over 5 distinct tasks: (0.4 + 0.2 + 0.6 + 0.4 + 0.45) / 5 = 0.41
    assert "Overall mean over 5 tasks: 0.4100" in text
    # the "all" aggregate from lighteval is not used
    assert "0.9000" not in text


def test_missing_task_shown_as_not_run():
    text = summarize(FAKE_RESULTS, DOMAINS)
    physics_rows = [line for line in text.splitlines() if "high_school_physics" in line]
    assert physics_rows and NOT_RUN in physics_rows[0]


def test_code_domain_marked_unavailable():
    text = summarize(FAKE_RESULTS, DOMAINS)
    assert "not available in lighteval 0.11" in text


def test_accepts_runner_record():
    record = {"model": "m", "adapter": None, "max_samples": 10, "lighteval_results": FAKE_RESULTS}
    text = summarize(record, DOMAINS)
    assert "Overall mean over 5 tasks: 0.4100" in text
    assert "max_samples=10" in text


@pytest.mark.parametrize("results", [{}, {"results": {}}, {"lighteval_results": None}])
def test_empty_results_do_not_crash(results):
    text = summarize(results, DOMAINS)
    assert "no task scores found" in text
    assert "| math | 0/2 | not run |" in text


def test_empty_domains():
    assert "no task scores found" in summarize(FAKE_RESULTS, {})


def test_domains_yaml_matches_tasks_yaml():
    root = Path(__file__).resolve().parents[1] / "configs"
    tasks = yaml.safe_load((root / "eval_tasks.yaml").read_text(encoding="utf-8"))["tasks"]
    domains = yaml.safe_load((root / "eval_domains.yaml").read_text(encoding="utf-8"))
    in_domains = {t for specs in domains.values() for t in (specs or [])}
    assert in_domains == set(tasks)
