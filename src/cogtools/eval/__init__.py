"""Evaluation code.

Submodules:
    lighteval_runner: scores the model (base plus optional LoRA) with lighteval.
    summarize: turns a lighteval results dict into a markdown table.
    agent_eval: runs the cognitive-tools agent loop on a math set.

Nothing heavy is imported here. Import the submodule you need directly, for
example ``from cogtools.eval.summarize import summarize``.
"""
