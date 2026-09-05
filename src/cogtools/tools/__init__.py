"""Cognitive tools agent (after Ebouky et al., arXiv:2506.12115).

Importing this package does not import torch or transformers.
"""

from .backends import Backend, HFBackend, ScriptedBackend
from .orchestrator import (
    TOOL_CALL_STOP,
    CognitiveToolsAgent,
    extract_answer,
    extract_boxed,
    extract_final_answer,
    find_tool_call,
)
from .prompts import (
    ALL_TOOLS,
    AVAILABLE_TOOLS,
    TOOL_PROMPTS,
    USE_CODE,
    build_system_prompt,
    build_tool_messages,
)
from .sandbox import SandboxResult, run_python
from .trace import ToolCall, Trace

__all__ = [
    "ALL_TOOLS",
    "AVAILABLE_TOOLS",
    "TOOL_CALL_STOP",
    "TOOL_PROMPTS",
    "USE_CODE",
    "Backend",
    "CognitiveToolsAgent",
    "HFBackend",
    "SandboxResult",
    "ScriptedBackend",
    "ToolCall",
    "Trace",
    "build_system_prompt",
    "build_tool_messages",
    "extract_answer",
    "extract_boxed",
    "extract_final_answer",
    "find_tool_call",
    "run_python",
]
