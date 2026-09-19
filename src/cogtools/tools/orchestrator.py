"""Tool-calling loop for the cognitive tools of Ebouky et al. (arXiv:2506.12115).

The loop follows the paper (section 3 and Algorithm 1 in v2):

1. The main model gets the cognitive-tools system prompt and the question,
   and generates until it writes a tool call. ``HFBackend`` stops on the
   string ``")\\n```"``, the end of a fenced call. The reply is also cut right
   after the first complete tool call, so text the model writes past the call
   (for example an invented tool result) is dropped.
2. The call is parsed with the ``ast`` module. Calls look like the paper's
   traces: ``understand_question(question="...")``, ``print(examine_answer({...}))``,
   keyword, positional or single-dict arguments, inside or outside a
   ```` ```python ```` block.
3. The tool runs as a fresh backend call that sees only the tool prompt and
   the tool arguments.
4. The tool output is sent back as a user turn starting with ``Observation:``
   and the main model continues.
5. The loop ends when the model writes ``ANSWER:`` or replies with no tool
   call.

``max_turns`` limits the number of main-model generations. The paper sets no
such limit; it is added here so a model that keeps calling tools cannot loop
forever.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .backends import Backend, Message
from .prompts import (
    ALL_TOOLS,
    AVAILABLE_TOOLS,
    TOOL_PARAMETERS,
    USE_CODE,
    build_system_prompt,
    build_tool_messages,
)
from .sandbox import DEFAULT_MAX_OUTPUT_CHARS, DEFAULT_TIMEOUT_S, run_python
from .trace import ToolCall, Trace

#: Stop strings for the main model: a closing parenthesis then a closing fence.
TOOL_CALL_STOP: tuple[str, ...] = (")\n```",)

_ANSWER_RE = re.compile(r"ANSWER:[ \t]*(.*)")
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_TOOL_NAME_RE = re.compile(
    r"(?:print\(\s*)?\b(" + "|".join(re.escape(n) for n in AVAILABLE_TOOLS) + r")\s*\("
)


class ToolCallError(ValueError):
    """A tool call that cannot be run. The message is shown to the model."""


@dataclass
class ParsedCall:
    """A tool call found in the model's text.

    ``end`` is the index in the text just after the call. ``error`` is set when
    the call was found but could not be parsed; ``arguments`` is then empty.
    """

    name: str
    arguments: dict[str, Any]
    start: int
    end: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Parsing


def _call_from_expr(expr: ast.AST, name: str) -> ast.Call | None:
    for node in ast.walk(expr):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            return node
    return None


def _literal(node: ast.AST, what: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ToolCallError(f"{what} must be a literal value such as a string ({exc})") from exc


def call_arguments(name: str, call: ast.Call) -> dict[str, Any]:
    """Turn the arguments of a parsed call into a dict of parameter values.

    Accepted forms: ``tool("q")``, ``tool(question="q")`` and
    ``tool({"question": "q"})`` as in the paper's traces.
    """
    params = TOOL_PARAMETERS[name]
    args: dict[str, Any] = {}
    positional = list(call.args)
    if len(positional) == 1 and isinstance(positional[0], ast.Dict) and not call.keywords:
        value = _literal(positional[0], "the argument dict")
        if not all(isinstance(k, str) for k in value):
            raise ToolCallError("argument dict keys must be strings")
        return dict(value)
    if len(positional) > len(params):
        raise ToolCallError(
            f"{name} takes at most {len(params)} positional argument(s), got {len(positional)}"
        )
    for param, node in zip(params, positional, strict=False):
        args[param] = _literal(node, f"argument '{param}'")
    for kw in call.keywords:
        if kw.arg is None:
            raise ToolCallError("** arguments are not supported")
        args[kw.arg] = _literal(kw.value, f"argument '{kw.arg}'")
    return args


def _close_offset(text: str, start: int) -> list[int]:
    return [i + 1 for i in range(start, len(text)) if text[i] == ")"]


def find_tool_call(text: str) -> ParsedCall | None:
    """Find the first tool call in ``text``.

    A regex only locates candidate starts (a tool name followed by ``(``,
    optionally wrapped in ``print(``). The call itself is parsed with
    ``ast.parse``: the shortest span from the start that parses as a Python
    expression is taken. A candidate inside a code fence or a ``print(``
    wrapper that cannot be parsed, or whose arguments are not literals, is
    returned with ``error`` set. Such a candidate in plain prose is skipped.
    """
    for match in _TOOL_NAME_RE.finditer(text):
        name = match.group(1)
        start = match.start()
        # Inside a code fence or a print( wrapper the model clearly meant a
        # call, so a broken call is reported back. Elsewhere it may be prose
        # ("without backtracking (see above)") and is skipped.
        meant_as_call = text[:start].count("```") % 2 == 1 or text.startswith("print(", start)
        result: ParsedCall | None = None
        for end in _close_offset(text, match.end() - 1):
            try:
                expr = ast.parse(text[start:end].strip(), mode="eval")
            except SyntaxError:
                continue
            call = _call_from_expr(expr, name)
            if call is None:
                continue
            try:
                result = ParsedCall(name, call_arguments(name, call), start, end)
            except ToolCallError as exc:
                result = ParsedCall(name, {}, start, end, error=str(exc))
            break
        if result is None:
            result = ParsedCall(
                name,
                {},
                start,
                len(text),
                error="could not parse the tool call as a Python expression",
            )
        if result.error is None or meant_as_call:
            return result
    return None


def extract_answer(text: str) -> str | None:
    """Return the text after the last ``ANSWER:`` in ``text``, or ``None``."""
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer or None


def extract_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` in ``text``, or ``None``."""
    idx = text.rfind("\\boxed")
    while idx != -1:
        brace = text.find("{", idx)
        if brace != -1 and text[idx + len("\\boxed") : brace].strip() == "":
            depth = 0
            for j in range(brace, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace + 1 : j].strip()
        idx = text.rfind("\\boxed", 0, idx)
    return None


def extract_final_answer(text: str) -> str | None:
    """``ANSWER:`` line if present, else the last ``\\boxed{}``, else ``None``."""
    answer = extract_answer(text)
    if answer is not None:
        return answer
    return extract_boxed(text)


# ---------------------------------------------------------------------------
# Agent


def _extract_code(text: str) -> str | None:
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    if "```" in text:
        tail = text.rsplit("```", 1)[1]
        tail = re.sub(r"^(?:python|py)?[ \t]*\n", "", tail)
        return tail.strip() or None
    return None


class CognitiveToolsAgent:
    """Solve a question with the paper's cognitive tools.

    Args:
        backend: Model used for the main reasoning and for every tool call
            (the paper uses the same model for both).
        enabled_tools: Tool names offered to the model. Defaults to the four
            cognitive tools. Add ``"use_code"`` to let the model run Python;
            see ``cogtools.tools.sandbox`` for what that means.
        max_turns: Maximum number of main-model generations. Not in the paper.
        max_new_tokens: Token budget for each main-model generation.
        tool_max_new_tokens: Token budget for each tool call.
        code_timeout: Seconds before ``use_code`` execution is killed.
        code_max_output_chars: Output cap for ``use_code`` execution.
        generation_kwargs: Extra keyword arguments passed to every
            ``backend.generate`` call (for example ``temperature``).

    Missing tool arguments are filled in: ``question`` with the question being
    solved, and the reasoning arguments (``current_proposed_answer``,
    ``current_reasoning``, ``reasoning``) with the model's text so far.
    """

    def __init__(
        self,
        backend: Backend,
        enabled_tools: Sequence[str] = ALL_TOOLS,
        max_turns: int = 8,
        max_new_tokens: int = 1024,
        tool_max_new_tokens: int = 1024,
        code_timeout: float = DEFAULT_TIMEOUT_S,
        code_max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        unknown = [t for t in enabled_tools if t not in AVAILABLE_TOOLS]
        if unknown:
            raise ValueError(f"unknown tool(s): {', '.join(unknown)}")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.backend = backend
        self.enabled_tools: tuple[str, ...] = tuple(dict.fromkeys(enabled_tools))
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.tool_max_new_tokens = tool_max_new_tokens
        self.code_timeout = code_timeout
        self.code_max_output_chars = code_max_output_chars
        self.generation_kwargs = dict(generation_kwargs or {})
        self.system_prompt = build_system_prompt(self.enabled_tools)

    def initial_messages(self, question: str) -> list[Message]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

    def solve(self, question: str) -> Trace:
        trace = Trace(question=question)
        messages = self.initial_messages(question)
        reasoning_so_far: list[str] = []

        for _turn in range(self.max_turns):
            reply = self.backend.generate(
                messages,
                max_new_tokens=self.max_new_tokens,
                stop=list(TOOL_CALL_STOP),
                **self.generation_kwargs,
            )
            parsed = find_tool_call(reply)
            answer_pos = reply.find("ANSWER:")

            if parsed is None or (answer_pos != -1 and answer_pos < parsed.start):
                # No tool call before the answer: this turn is final.
                text = reply.strip()
                trace.steps.append(text)
                answer = extract_answer(text)
                if answer is not None:
                    trace.final_answer = answer
                    trace.stopped_reason = "answer"
                else:
                    trace.final_answer = extract_boxed(text)
                    trace.stopped_reason = "no_tool_call"
                return trace

            # Keep the text up to the end of the call and drop the rest.
            kept = reply[: parsed.end]
            if kept.count("```") % 2 == 1:
                kept += "\n```"
            kept = kept.strip()
            trace.steps.append(kept)
            reasoning_so_far.append(kept)

            call = self._run_tool(parsed, question, "\n\n".join(reasoning_so_far))
            trace.steps.append(call)
            messages.append({"role": "assistant", "content": kept})
            messages.append({"role": "user", "content": "Observation: " + call.output})

        trace.stopped_reason = "max_turns"
        return trace

    # -- tool execution -----------------------------------------------------

    def _run_tool(self, parsed: ParsedCall, question: str, reasoning: str) -> ToolCall:
        name = parsed.name
        if parsed.error:
            msg = f"Error: could not run {name}: {parsed.error}."
            return ToolCall(name, parsed.arguments, msg, error=parsed.error)
        if name not in self.enabled_tools:
            err = f"tool '{name}' is not available"
            msg = f"Error: {err}. Available tools: {', '.join(self.enabled_tools)}."
            return ToolCall(name, parsed.arguments, msg, error=err)

        # Backend errors (out of memory, a missing model) are not caught here:
        # they are failures of the run, not something the model can fix.
        arguments = dict(parsed.arguments)
        inputs = self._tool_inputs(name, arguments, question, reasoning)
        if name == USE_CODE:
            output = self._run_code_tool(inputs)
        else:
            output = self.backend.generate(
                build_tool_messages(name, inputs),
                max_new_tokens=self.tool_max_new_tokens,
                **self.generation_kwargs,
            ).strip()
        return ToolCall(name, arguments, output)

    @staticmethod
    def _tool_inputs(
        name: str, arguments: dict[str, Any], question: str, reasoning: str
    ) -> dict[str, str]:
        inputs: dict[str, str] = {}
        for param in TOOL_PARAMETERS[name]:
            value = arguments.get(param)
            if value is None or value == "":
                value = question if param == "question" else reasoning
            inputs[param] = str(value)
        return inputs

    def _run_code_tool(self, inputs: dict[str, str]) -> str:
        reply = self.backend.generate(
            build_tool_messages(USE_CODE, inputs),
            max_new_tokens=self.tool_max_new_tokens,
            **self.generation_kwargs,
        )
        code = _extract_code(reply)
        if not code:
            return "No code block found in the generated reply:\n" + reply.strip()
        result = run_python(
            code, timeout=self.code_timeout, max_output_chars=self.code_max_output_chars
        )
        # Same layout as the observation in the paper's example trace.
        return (
            "Generated code block:\n```python\n"
            + code
            + "\n```\nExecution Output: "
            + result.as_observation()
        )


__all__ = [
    "CognitiveToolsAgent",
    "ParsedCall",
    "TOOL_CALL_STOP",
    "ToolCallError",
    "call_arguments",
    "extract_answer",
    "extract_boxed",
    "extract_final_answer",
    "find_tool_call",
]
