import json

import pytest

from cogtools.tools import (
    TOOL_CALL_STOP,
    CognitiveToolsAgent,
    ScriptedBackend,
    ToolCall,
    Trace,
    extract_final_answer,
    find_tool_call,
)
from cogtools.tools.prompts import UNDERSTAND_QUESTION_PROMPT, USE_CODE_PROMPT

QUESTION = "What is 2 + 3 * 4?"

CALL_REPLY = (
    "I will break the question down first.\n"
    "```python\n"
    'understand_question(question="What is 2 + 3 * 4?")\n'
    "```\n"
    "Observation: the answer is 20\n"  # invented tool output, must be dropped
    "ANSWER: 20"
)


def test_tool_call_parsed_executed_and_fed_back():
    backend = ScriptedBackend(
        [
            CALL_REPLY,
            "Multiplication first: 3 * 4 = 12.",  # the tool's reply
            "So 2 + 12 = 14.\nANSWER: 14",
        ]
    )
    trace = CognitiveToolsAgent(backend).solve(QUESTION)

    assert trace.stopped_reason == "answer"
    assert trace.final_answer == "14"
    calls = trace.tool_calls
    assert len(calls) == 1
    assert calls[0].name == "understand_question"
    assert calls[0].arguments == {"question": QUESTION}
    assert calls[0].output == "Multiplication first: 3 * 4 = 12."
    assert calls[0].error is None

    # Main model call asked to stop at the end of a tool call.
    assert backend.calls[0]["stop"] == list(TOOL_CALL_STOP)

    # The tool ran in a fresh context: its prompt plus its inputs only.
    tool_msgs = backend.calls[1]["messages"]
    assert [m["role"] for m in tool_msgs] == ["system", "user"]
    assert tool_msgs[0]["content"] == UNDERSTAND_QUESTION_PROMPT
    assert QUESTION in tool_msgs[1]["content"]

    # The observation was fed back, and the invented continuation was dropped.
    final_msgs = backend.calls[2]["messages"]
    assert final_msgs[-1] == {
        "role": "user",
        "content": "Observation: Multiplication first: 3 * 4 = 12.",
    }
    assistant = final_msgs[-2]
    assert assistant["role"] == "assistant"
    assert assistant["content"].endswith("```")
    assert "the answer is 20" not in assistant["content"]
    assert "ANSWER: 20" not in assistant["content"]


def test_answer_without_tools():
    backend = ScriptedBackend(["3 * 4 = 12, plus 2 is 14.\nANSWER: 14"])
    trace = CognitiveToolsAgent(backend).solve(QUESTION)
    assert trace.final_answer == "14"
    assert trace.stopped_reason == "answer"
    assert trace.tool_calls == []


def test_no_tool_call_and_no_answer_line_uses_boxed():
    backend = ScriptedBackend(["The result is $\\boxed{14}$."])
    trace = CognitiveToolsAgent(backend).solve(QUESTION)
    assert trace.stopped_reason == "no_tool_call"
    assert trace.final_answer == "14"


def test_max_turns_stops_loop():
    reply = '```python\nrecall_related("q")\n```'
    backend = ScriptedBackend([reply, "example", reply, "example", reply, "example"])
    trace = CognitiveToolsAgent(backend, max_turns=2).solve(QUESTION)
    assert trace.stopped_reason == "max_turns"
    assert trace.final_answer is None
    assert len(trace.tool_calls) == 2
    assert backend.remaining == 2


def test_disabled_tool_rejected():
    reply = '```python\nbacktracking(question="q", current_reasoning="r")\n```'
    backend = ScriptedBackend([reply, "ANSWER: 14"])
    agent = CognitiveToolsAgent(backend, enabled_tools=["understand_question"])
    trace = agent.solve(QUESTION)
    call = trace.tool_calls[0]
    assert call.name == "backtracking"
    assert call.error is not None
    assert "not available" in call.output
    # No tool model call was made; the error went straight back.
    assert len(backend.calls) == 2
    assert backend.calls[1]["messages"][-1]["content"].startswith("Observation: Error:")
    assert trace.final_answer == "14"


def test_use_code_disabled_by_default():
    reply = '```python\nuse_code(question="q")\n```'
    backend = ScriptedBackend([reply, "ANSWER: 1"])
    trace = CognitiveToolsAgent(backend).solve(QUESTION)
    assert trace.tool_calls[0].error is not None
    assert "def use_code(" not in backend.calls[0]["messages"][0]["content"]


def test_malformed_call_reported_to_model():
    reply = '```python\nexamine_answer(question="q", current_proposed_answer=\n```'
    backend = ScriptedBackend([reply, "ANSWER: 14"])
    trace = CognitiveToolsAgent(backend).solve(QUESTION)
    call = trace.tool_calls[0]
    assert call.name == "examine_answer"
    assert call.error
    assert backend.calls[1]["messages"][-1]["content"].startswith("Observation: Error:")
    assert trace.final_answer == "14"


def test_non_literal_argument_reported():
    reply = "```python\nexamine_answer(question=q)\n```"
    backend = ScriptedBackend([reply, "ANSWER: 14"])
    trace = CognitiveToolsAgent(backend).solve(QUESTION)
    assert "literal" in trace.tool_calls[0].error


def test_tool_name_in_prose_is_not_a_call():
    reply = "I solved it without backtracking (it was easy). 2 + 12 = 14.\nANSWER: 14"
    trace = CognitiveToolsAgent(ScriptedBackend([reply])).solve(QUESTION)
    assert trace.tool_calls == []
    assert trace.final_answer == "14"


def test_paper_style_print_dict_call():
    text = (
        "print(examine_answer({'question': 'Find the GCD', "
        "'current_proposed_answer': 'The GCD is 21', 'model': 'math_problem'}))"
    )
    parsed = find_tool_call(text)
    assert parsed is not None and parsed.error is None
    assert parsed.name == "examine_answer"
    assert parsed.arguments["current_proposed_answer"] == "The GCD is 21"
    assert parsed.end == len(text)


def test_string_with_parentheses_parsed():
    text = 'recall_related(question="f(x) = (x+1)^2, find f(2)") and more text'
    parsed = find_tool_call(text)
    assert parsed.arguments == {"question": "f(x) = (x+1)^2, find f(2)"}


def test_answer_before_tool_call_ends_loop():
    reply = 'ANSWER: 14\n```python\nunderstand_question("q")\n```'
    backend = ScriptedBackend([reply])
    trace = CognitiveToolsAgent(backend).solve(QUESTION)
    assert trace.final_answer == "14"
    assert trace.tool_calls == []


def test_missing_reasoning_argument_filled_from_trace():
    reply = 'Proposed answer 14.\n```python\nexamine_answer(question="q")\n```'
    backend = ScriptedBackend([reply, "correct", "ANSWER: 14"])
    CognitiveToolsAgent(backend).solve(QUESTION)
    tool_user = backend.calls[1]["messages"][1]["content"]
    assert "current_proposed_answer:\nProposed answer 14." in tool_user


def test_use_code_runs_generated_code():
    code_reply = "Thought: compute it.\nCode:\n```python\nprint(2 + 3 * 4)\n```"
    backend = ScriptedBackend(
        ['```python\nuse_code(question="What is 2 + 3 * 4?")\n```', code_reply, "ANSWER: 14"]
    )
    agent = CognitiveToolsAgent(
        backend, enabled_tools=["understand_question", "use_code"], code_timeout=30
    )
    trace = agent.solve(QUESTION)
    call = trace.tool_calls[0]
    assert call.error is None
    assert backend.calls[1]["messages"][0]["content"] == USE_CODE_PROMPT
    assert "Generated code block:" in call.output
    assert "Execution Output: 14" in call.output


def test_unknown_enabled_tool_raises():
    with pytest.raises(ValueError):
        CognitiveToolsAgent(ScriptedBackend([]), enabled_tools=["guess"])


def test_trace_round_trip_and_markdown():
    trace = Trace(
        question="q",
        steps=["thinking", ToolCall("recall_related", {"question": "q"}, "examples")],
        final_answer="3",
        stopped_reason="answer",
    )
    data = json.loads(trace.to_json())
    assert data["steps"][1]["type"] == "tool_call"
    assert Trace.from_dict(data) == trace
    md = trace.to_markdown()
    assert "tool call `recall_related`" in md
    assert "Final answer: 3" in md


def test_extract_final_answer():
    assert extract_final_answer("so ANSWER: 7\n") == "7"
    assert extract_final_answer("x = \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"
    assert extract_final_answer("no answer here") is None


def test_no_tools_baseline_answers_directly():
    agent = CognitiveToolsAgent(ScriptedBackend(["The sum is 4. ANSWER: 4"]), enabled_tools=[])
    trace = agent.solve("What is 2 + 2?")
    assert trace.final_answer == "4"
    assert trace.stopped_reason == "answer"
