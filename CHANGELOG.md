# Changelog

## 0.1.0 (2026-11-17)

First release as a Python package. It replaces the original Colab notebooks.

- `cogtools.data`: random sampling per domain from OpenThoughts3-1.2M with exact quotas, plus an optional token-length filter. The original took the first rows of each domain.
- `cogtools.train`: the Unsloth LoRA recipe from the original notebook as a function with a YAML config. Each run writes `run_record.json` with the config, package versions and git commit.
- `cogtools.tools`: the cognitive-tools loop from Ebouky et al. (2025). It stops at the tool call, parses the call with `ast`, gives each tool a fresh context and feeds results back as `Observation:` turns.
- `cogtools.eval.lighteval_runner`: Lighteval 0.11 scoring of the base model plus LoRA adapter, with task names that exist in that version.
- `cogtools.eval.agent_eval`: answer checking for the agent loop on JSONL math problems.
- Removed benchmark numbers that no run had produced. See `docs/history.md`.
- Removed a Hugging Face token that was hardcoded in the original training notebook.
