from cogtools.tools import prompts as P


def test_five_tool_prompts_present():
    assert set(P.TOOL_PROMPTS) == set(P.AVAILABLE_TOOLS)
    assert P.USE_CODE not in P.ALL_TOOLS
    assert len(P.ALL_TOOLS) == 4


def test_prompts_keep_paper_wording():
    assert P.UNDERSTAND_QUESTION_PROMPT.startswith(
        "You are a mathematical reasoning assistant designed to analyze and break down"
    )
    assert "5. DO NOT provide any answer to the question" in P.UNDERSTAND_QUESTION_PROMPT
    assert "Identify 2 or 3 **similar problems**" in P.RECALL_RELATED_PROMPT
    assert "- DO NOT provide the actual answer to the problem." in P.EXAMINE_ANSWER_PROMPT
    assert "**Backtrack Point:**" in P.BACKTRACKING_PROMPT
    assert "ALWAYS explicitly PRINT the final result" in P.USE_CODE_PROMPT
    assert "Never hallucinate results." in P.COGNITIVE_TOOLS_SYSTEM_PROMPT


def test_python_string_leftovers_removed():
    for text in P.TOOL_PROMPTS.values():
        assert not text.rstrip().endswith('"')
        assert '"""' not in text


def test_reconstructed_math_in_recall_prompt():
    assert P.RECALL_RELATED_PROMPT.count("Final Answer: $\\boxed{...}$") == 3


def test_no_curly_quotes():
    texts = [P.COGNITIVE_TOOLS_SYSTEM_PROMPT, *P.TOOL_PROMPTS.values()]
    for text in texts:
        for ch in "‘’“”":
            assert ch not in text


def test_system_prompt_lists_only_enabled_tools():
    prompt = P.build_system_prompt(["understand_question", "examine_answer"])
    assert P.SIGNATURE_PLACEHOLDER not in prompt
    assert "def understand_question(" in prompt
    assert "def examine_answer(" in prompt
    assert "def recall_related(" not in prompt
    assert "def use_code(" not in prompt
    assert "```python" in prompt


def test_unknown_tool_rejected():
    import pytest

    with pytest.raises(ValueError):
        P.build_system_prompt(["solve_it"])


def test_tool_messages_hold_only_prompt_and_inputs():
    msgs = P.build_tool_messages(
        "examine_answer", {"question": "Q?", "current_proposed_answer": "4"}
    )
    assert msgs[0] == {"role": "system", "content": P.EXAMINE_ANSWER_PROMPT}
    assert msgs[1]["role"] == "user"
    assert "question:\nQ?" in msgs[1]["content"]
    assert "current_proposed_answer:\n4" in msgs[1]["content"]
    assert len(msgs) == 2
