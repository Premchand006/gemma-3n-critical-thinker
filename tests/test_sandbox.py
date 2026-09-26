import time

from cogtools.tools.sandbox import run_python


def test_prints_output():
    result = run_python("print(sum(range(10)))", timeout=30)
    assert result.stdout.strip() == "45"
    assert result.returncode == 0
    assert not result.timed_out
    assert not result.truncated


def test_error_goes_to_stderr():
    result = run_python("raise ValueError('boom')", timeout=30)
    assert result.returncode != 0
    assert "ValueError: boom" in result.stderr
    assert "ValueError: boom" in result.as_observation()


def test_timeout_kills_process():
    start = time.monotonic()
    result = run_python("while True:\n    pass", timeout=1.0)
    assert result.timed_out
    assert result.returncode is None
    assert time.monotonic() - start < 15
    assert "time limit" in result.as_observation()


def test_output_is_capped():
    result = run_python("while True:\n    print('x' * 1000)", timeout=30, max_output_chars=5000)
    assert result.truncated
    assert not result.timed_out
    assert len(result.stdout) <= 5000


def test_unicode_output():
    result = run_python("print('\\u03c0 \\u2248 3.14')", timeout=30)
    assert result.stdout.strip() == "π ≈ 3.14"


def test_isolated_mode():
    result = run_python("import sys; print(sys.flags.isolated)", timeout=30)
    assert result.stdout.strip() == "1"
