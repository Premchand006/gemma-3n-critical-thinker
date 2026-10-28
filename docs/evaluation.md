# Evaluation

There are two kinds of evaluation. Neither has been run on a GPU yet.

## 1. Lighteval tasks (model alone)

`scripts/evaluate.py` scores the base model plus the LoRA adapter on 13 Lighteval tasks. The cognitive tools are not used here; this measures the fine-tuned model by itself.

| Domain | Tasks (Lighteval 0.11 spec) |
| --- | --- |
| Math | `leaderboard\|gsm8k\|5`, `lighteval\|math:algebra@k=1\|4`, `lighteval\|math:geometry@k=1\|4`, `lighteval\|math:number_theory@k=1\|4`, `lighteval\|math:prealgebra@k=1\|4` |
| Science | `leaderboard\|arc:challenge\|25`, `leaderboard\|mmlu:high_school_chemistry\|5`, `leaderboard\|mmlu:high_school_physics\|5`, `leaderboard\|mmlu:college_mathematics\|5` |
| General | `leaderboard\|mmlu\|5`, `leaderboard\|hellaswag\|10`, `leaderboard\|truthfulqa:mc\|0`, `leaderboard\|winogrande\|5` |
| Code | none: Lighteval 0.11 has no HumanEval or MBPP task |

The full list is in `configs/eval_tasks.yaml`, and the domain grouping is in `configs/eval_domains.yaml`.

```bash
pip install -r requirements/eval.txt && pip install -e . --no-deps
python scripts/evaluate.py --config configs/eval_tasks.yaml --max-samples 100
python scripts/evaluate.py --adapter none --max-samples 100        # base model, for comparison
python scripts/summarize_results.py results/<timestamp>/results.json --out results/<timestamp>/summary.md
```

How the runner works:

- It loads the base model in 4-bit (NF4) and attaches the adapter with `PeftModel.from_pretrained`, without merging. It then wraps the model with `TransformersModel.from_model`. Lighteval's own adapter loader is not used because in 0.11 it reads a config field that does not exist and pins `peft==0.3.0`.
- `max_length` defaults to 4,096 tokens. Lighteval 0.11 cannot find Gemma 3n's text config and would otherwise fall back to 2,048, which truncates 25-shot ARC prompts.
- The compute dtype defaults to float16 because the T4 has no bfloat16. Gemma models can overflow in float16. If scores look broken, try `--compute-dtype float32`.
- The MATH tasks use `@k=1`, which gives one greedy sample per problem, in place of the default 4-sample majority vote. That default caused the original notebook's `temperature=0` error. The `@k=1` syntax was taken from the Lighteval source and has not been run. If 0.11 rejects it, remove those four lines from the config.
- Each run writes Lighteval's own files plus `results/<timestamp>/results.json`. The JSON holds the model, adapter, task list, `max_samples`, package versions and the raw results.

`summarize_results.py` picks one main metric per task (for example `acc_norm` for ARC and HellaSwag, `truthfulqa_mc2` for TruthfulQA). It computes domain means and an overall mean from the scores in the file, and marks missing tasks as "not run". It never fills in numbers.

With `--max-samples 100`, each score comes from 100 questions. For an accuracy near 50%, the 95% confidence interval is roughly ±10 points, so small differences between runs mean little.

## 2. Agent loop (cognitive tools)

`scripts/agent_eval.py` runs the cognitive-tools agent over a JSONL file of problems in the form `{"id", "question", "answer"}`. It extracts the final answer from `ANSWER:` or `\boxed{}`, normalizes it (numbers, fractions, spacing) and checks it against the reference.

```bash
python scripts/agent_eval.py --data data/sample_math.jsonl                 # all four tools
python scripts/agent_eval.py --data data/sample_math.jsonl --tools ""      # no tools, as a baseline
```

`data/sample_math.jsonl` has a few easy problems for checking that the pipeline works. It is not a benchmark. For a real measurement, use problems like the ones the paper used (AIME 2024, MATH500, AMC) in the same JSONL format. Run with all tools and again with tools turned off, and compare. The script writes `traces.jsonl` (one line per problem with the full trace) and `summary.json` (counts and accuracy) to the output folder.

## Results

| Run | Model | Status |
| --- | --- | --- |
| Lighteval, 13 tasks | `gemma-3n-E4B-it` + `anhsirk/gemma-3n-critical-thinker` | not yet measured |
| Lighteval, 13 tasks | `gemma-3n-E4B-it` (base) | not yet measured |
| Agent loop, math set | adapter, with tools / without tools | not yet measured |

Add a row here, with a link to its results JSON, after each real run.
