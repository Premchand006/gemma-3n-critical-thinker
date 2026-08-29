# Training

## The October 2025 run

This run produced the adapter [`anhsirk/gemma-3n-critical-thinker`](https://huggingface.co/anhsirk/gemma-3n-critical-thinker). The config is `configs/train_gemma3n_e4b_original.yaml` and the package set is `requirements/train-original.txt`.

| Setting | Value |
| --- | --- |
| Base model | `unsloth/gemma-3n-E4B-it`, 4-bit (Unsloth swapped in `unsloth/gemma-3n-e4b-it-unsloth-bnb-4bit`) |
| Packages | unsloth 2025.10.1, transformers 4.55.4, trl 0.22.2, torch 2.8.0+cu126, xformers 0.0.32.post2 |
| LoRA | r = 8, alpha = 8, dropout 0, bias none, language, attention and MLP modules, vision layers frozen |
| Trainable parameters | 19,210,240 of 7,869,188,432 |
| Chat format | Unsloth `gemma-3` template, leading `<bos>` removed, loss on model turns only (`train_on_responses_only`) |
| Steps | 60, batch 1, gradient accumulation 4, so 240 examples |
| Optimizer | `adamw_8bit`, learning rate 2e-4, linear schedule, 5 warmup steps, weight decay 0.01 |
| Sequence length | 1,024 tokens |
| Precision | fp32 compute (Unsloth does not use fp16 for Gemma 3n) |
| Seed | 3407 |
| Hardware | One Tesla T4 (14.7 GB) on Colab |
| Train time | 632 seconds |
| Peak reserved GPU memory | 13.6 GB (already reserved before training began, so the share used by training alone is unknown) |
| Saved | LoRA adapter only, 76.9 MB |

The per-step loss was not saved, so there is no loss curve for this run.

### Known problems with this run

1. **Few examples.** The notebook collected 2,497 rows, but 60 steps at an effective batch of 4 trained on only 240 of them, about 10%.
2. **Truncated examples.** OpenThoughts3 answers are long; sampled ones ran from about 33,000 to 72,000 characters. At 1,024 tokens nearly every example was cut off early in its `<think>` section. The model never saw the end of a reasoning trace or a final answer during training.
3. **Not a random sample.** The dataset is sorted by domain (code, then math, then science). The notebook took the first rows of each domain, which also meant streaming about 1.1 million rows before the science quota filled.
4. **Quota rounding.** `int(2500 * ratio)` per domain gave 1,770 + 520 + 207 = 2,497 rows rather than 2,500.
5. **Wrong license tag.** The pushed adapter is tagged `apache-2.0`. A Gemma derivative falls under the Gemma Terms of Use. The model card in `docs/model_card.md` has the corrected tag.

## What the code does now

- `sampling: random` streams the dataset through a seeded shuffle buffer and reservoir-samples each domain, so the rows are not the first ones in the file. `sampling: first_n` is kept only to reproduce the old run.
- `quota_rounding: largest_remainder` makes the quotas add up to `total_samples` exactly.
- `max_token_length` drops rows longer than the limit, so the rows that remain are trained on in full. Most OpenThoughts3 rows are long, so expect this filter to drop a large share. `scripts/prepare_data.py` prints how many rows it kept per domain.
- `run_training` writes `run_record.json` to the output folder with the config, package versions, git commit and trainer metrics.
- `push_to_hub_repo` uploads only when the `HF_TOKEN` environment variable is set. The token is never written to disk or into a notebook.

## Recommended next run (untested)

`configs/train_gemma3n_e4b.yaml` with `requirements/train.txt` uses random sampling, `max_seq_length` and `max_token_length` of 8,192, and one full epoch over 2,500 rows (625 optimizer steps). Nobody has checked yet whether 8,192 tokens fits in T4 memory. If it does not, lower both values together, for example to 4,096.

```bash
pip install -r requirements/train.txt && pip install -e . --no-deps
python scripts/prepare_data.py --config configs/train_gemma3n_e4b.yaml --out data/subset.jsonl
python scripts/train.py --config configs/train_gemma3n_e4b.yaml
```

## Model facts

Gemma 3n E4B has about 8B raw parameters. Per-Layer Embeddings and the MatFormer design let it run with a memory footprint close to a 4B model (about 3 GB in 4-bit). The context window is 32K tokens. Unsloth's GGUF export for Gemma 3n supports Q8_0, BF16 and F16 only, for text.
