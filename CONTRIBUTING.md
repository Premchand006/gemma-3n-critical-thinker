# Contributing

## Set up

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

The tests run on CPU and need only `pytest` and `pyyaml`. Code that loads a model imports torch inside functions, so importing the package does not need a GPU.

## Before you open a pull request

- `ruff check .` and `ruff format --check .` pass.
- `pytest` passes.
- Notebooks are committed with outputs cleared. The `nbstripout` pre-commit hook does this.
- No tokens or keys in code, configs or notebooks. Read the Hugging Face token from the `HF_TOKEN` environment variable or a Colab secret.

## Reporting results

Only report scores that a run in this repository produced. Commit the `results/<timestamp>/results.json` file with the summary table, and state the config, the package versions (both are recorded in the JSON) and the GPU. If a run was cut short or used `--max-samples`, say so next to the numbers.

## Writing style

Docs, docstrings and comments should be plain and specific. Say what the code does and what was measured. Leave out claims that a run did not test, and wording that sells rather than describes.
