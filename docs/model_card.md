---
base_model: unsloth/gemma-3n-E4B-it
library_name: peft
license: gemma
datasets:
  - open-thoughts/OpenThoughts3-1.2M
language:
  - en
pipeline_tag: text-generation
tags:
  - gemma3n
  - lora
  - unsloth
  - trl
  - sft
  - reasoning
---

# gemma-3n-critical-thinker

A LoRA adapter for `gemma-3n-E4B-it`, trained briefly on a sample of OpenThoughts3-1.2M. Code and training record: https://github.com/Premchand006/gemma-3n-critical-thinker

## Training

- Base: `unsloth/gemma-3n-E4B-it`, loaded in 4-bit with Unsloth 2025.10.1
- LoRA r = 8, alpha = 8, language layers only
- 60 steps at an effective batch size of 4, so 240 examples, at 1,024 tokens per example
- Learning rate 2e-4, AdamW 8-bit, one Tesla T4, 632 seconds

## Limits

- This is a short run. Most training examples were cut off at 1,024 tokens, before the end of their reasoning, so the adapter has seen few complete solutions.
- The 240 examples came from the first rows of each domain in the dataset, not a random sample.
- No benchmark scores have been measured for this adapter.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3n-E4B-it",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "anhsirk/gemma-3n-critical-thinker")
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E4B-it")
```

## License

Gemma 3n and models derived from it are under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). The training data, OpenThoughts3-1.2M, is Apache 2.0.
