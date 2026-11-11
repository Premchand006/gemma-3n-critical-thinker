# What changed from the original project files

The project started as four Colab notebooks and several report drafts (a proposal, an IEEE-style paper in Word and PDF, and a results table). This repository replaces them. This page records what those files contained and why parts were dropped, so that nobody re-uses the old numbers by mistake.

## The notebooks

| Notebook | What it did | Outcome |
| --- | --- | --- |
| `Gemma3N_(4B)_finetuning` | LoRA fine-tune of `gemma-3n-E4B-it` on 2,497 OpenThoughts3 rows | Ran. Trained on 240 examples in 632 s and pushed the adapter. See `training.md` |
| `openthoughts3_finetuning` | A longer config: Gemma 3 4B (not 3n), r = 16, 1,000 steps, GGUF export | Never ran past model loading. It would have failed: it read `question` and `instruction` columns that OpenThoughts3 does not have |
| `Cognitive_tools` | A first version of the tool loop | Ran on Qwen3-4B, not Gemma 3n. In the recorded run the model never actually triggered a tool. It wrote its own tool line, made up the result and answered in one step |
| `lighteval_gemma_colab` | Benchmark evaluation | Scored `open-thoughts/OpenThinker3-1.5B`, not Gemma 3n. Every task failed (wrong task names, a `temperature=0` error on MATH) and no scores were produced. The cognitive tools were defined but never called |

The training notebook also had a Hugging Face write token in plain text. It has been left out of this repository, and the token should be revoked on Hugging Face.

## The report drafts

The drafts reported an overall score of 59.2% across 15 tasks, a score for each task, a 5 to 8 point gain from the cognitive tools "estimated via ablation", and how often each tool was called. None of the notebooks produced any of these numbers, and the only evaluation run failed. The 59.2% does not match the drafts' own tables either: those tables average to 57.9% by task and 55.2% by domain.

The drafts also described a training run that did not happen: 5,000 examples, r = 16, 1,000 steps, about 12 hours, and a validation loss of 1.02. Those values came from the notebook that never ran.

Other errors in the drafts:

- Gemma 3n E4B was called a 1.5B model. It has about 8B raw parameters, with a memory footprint close to a 4B model.
- A comparison table credited GPT-4.1 with cognitive tools with GSM8K, HumanEval and MMLU scores and cited the cognitive-tools paper. That paper reports math benchmarks only. The other models' numbers in the table did not match their primary sources either.
- HumanEval and MBPP were listed as evaluated. Lighteval 0.11 has neither task.
- One reference appeared twice. The PDF ended with a pasted list of web sources, and the Word files had broken hyperlink field text.

## What this repository keeps

- The training recipe that ran, as `configs/train_gemma3n_e4b_original.yaml` with the exact package versions.
- The idea and the prompts of the cognitive tools, rebuilt to follow the paper's loop.
- The Lighteval task list, corrected to task names that exist and with the code tasks removed.
- The published adapter, documented with its limits in `model_card.md`.

No benchmark result from the old files is kept. The results tables say "not yet measured" until a run in this repository fills them.
