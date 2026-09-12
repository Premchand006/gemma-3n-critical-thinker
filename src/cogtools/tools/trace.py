"""Records of one agent run: the model's text, the tool calls and the answer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool call made by the main model.

    ``output`` is the text returned to the model as the ``Observation:`` turn.
    ``error`` is set when the call could not be run (unparseable call, disabled
    tool, bad arguments, backend failure); ``output`` then holds the error
    message that was shown to the model.
    """

    name: str
    arguments: dict[str, Any]
    output: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "name": self.name,
            "arguments": self.arguments,
            "output": self.output,
            "error": self.error,
        }


Step = str | ToolCall


@dataclass
class Trace:
    """Full record of ``CognitiveToolsAgent.solve``.

    ``steps`` alternates between model text (``str``) and ``ToolCall`` records,
    in the order they happened. ``stopped_reason`` is one of ``"answer"``
    (the model wrote ``ANSWER:``), ``"no_tool_call"`` (the model stopped
    without a tool call or an ``ANSWER:`` line) or ``"max_turns"``.
    """

    question: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    stopped_reason: str = ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s for s in self.steps if isinstance(s, ToolCall)]

    def to_dict(self) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        for step in self.steps:
            if isinstance(step, ToolCall):
                steps.append(step.to_dict())
            else:
                steps.append({"type": "text", "content": step})
        return {
            "question": self.question,
            "steps": steps,
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        steps: list[Step] = []
        for item in data.get("steps", []):
            if item.get("type") == "tool_call":
                steps.append(
                    ToolCall(
                        name=item["name"],
                        arguments=dict(item.get("arguments", {})),
                        output=item.get("output", ""),
                        error=item.get("error"),
                    )
                )
            else:
                steps.append(item.get("content", ""))
        return cls(
            question=data["question"],
            steps=steps,
            final_answer=data.get("final_answer"),
            stopped_reason=data.get("stopped_reason", ""),
        )

    def to_markdown(self) -> str:
        lines = ["## Question", "", self.question, ""]
        for i, step in enumerate(self.steps, start=1):
            if isinstance(step, ToolCall):
                lines.append(f"### Step {i}: tool call `{step.name}`")
                lines.append("")
                lines.append("Arguments:")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(step.arguments, ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")
                if step.error:
                    lines.append(f"Error: {step.error}")
                    lines.append("")
                lines.append("Observation:")
                lines.append("")
                lines.append(_quote(step.output))
            else:
                lines.append(f"### Step {i}: model")
                lines.append("")
                lines.append(_quote(step))
            lines.append("")
        lines.append("## Result")
        lines.append("")
        answer = self.final_answer if self.final_answer is not None else "(none)"
        lines.append(f"Final answer: {answer}")
        lines.append("")
        lines.append(f"Stopped because: {self.stopped_reason}")
        lines.append("")
        return "\n".join(lines)


def _quote(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in text.splitlines()) or ">"


__all__ = ["Step", "ToolCall", "Trace"]
