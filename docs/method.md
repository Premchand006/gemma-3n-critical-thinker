# How the cognitive tools work

The agent loop follows Ebouky, Bartezzaghi and Rigotti, "Eliciting Reasoning in Language Models with Cognitive Tools" (arXiv:2506.12115). The code is in `src/cogtools/tools/`.

## The tools

Each tool is a prompt. When the main model calls a tool, the same model answers that prompt in a fresh conversation. It sees only the tool prompt and the tool's arguments, not the main reasoning so far.

| Tool | Input | What the prompt asks for |
| --- | --- | --- |
| `understand_question` | the question | Break the problem into its parts and name the concepts and methods that apply, without solving it |
| `recall_related` | the question | Recall two or three similar solved problems with their solutions |
| `examine_answer` | the question and the current answer | Check the current reasoning for errors and missed cases, without giving the answer |
| `backtracking` | the question and the reasoning so far | Find where the reasoning went wrong and suggest another approach |
| `use_code` | the question | Write Python that solves or checks the problem; the code is run and its printed output returned |

The system prompt and the five tool prompts are the paper's text. The changes are listed at the top of `prompts.py` and in `NOTICE`. `use_code` is off by default because it runs model-written code on your machine (see below).

## The loop

1. The main model gets the system prompt, the tool signatures and the question.
2. It generates until it writes a tool call. `HFBackend` stops on the end of a fenced call (`)` then a closing fence), and the reply is cut after the first complete call. If a model writes a made-up tool result after the call, that text is dropped.
3. The call is parsed with Python's `ast` module. The accepted forms are the ones in the paper's traces: `understand_question(question="...")`, `print(examine_answer({...}))`, and keyword, positional or single-dict arguments.
4. The tool runs and its output goes back to the main model as a user turn that starts with `Observation:`.
5. The loop ends when the model writes `ANSWER:` or replies without a tool call.

`max_turns` (default 8) caps the number of main-model generations. The paper sets no cap. A malformed call, or a call to a tool that is not enabled, is not fatal: the error goes back to the model as the observation, so it can try again.

Each run returns a `Trace` with the question, every model turn, every tool call with its arguments and output, the final answer and the reason the loop stopped. `Trace.to_json()` and `Trace.to_markdown()` write it out.

## use_code

`sandbox.run_python` writes the code to a temporary file and runs it with `python -I` in a child process. It has a timeout (10 seconds by default) and caps the output size. This only stops runaway loops and huge outputs. It does not stop file or network access, so enable `use_code` only on a machine you are willing to let the model run code on.

## What the paper tested, and what it did not

The paper tested Qwen2.5 7B and 32B, Llama 3.1 8B, Llama 3.3 70B and GPT-4.1, all on math benchmarks only: AIME 2024, MATH500, AMC and the math part of the smolagents benchmark. The gains it reports were large for the 32B and 70B models and small for the 7B and 8B ones. The headline GPT-4.1 numbers differ between versions of the preprint, so cite the version you read. The paper reports no GSM8K, HumanEval or MMLU numbers.

Nobody has measured whether the tools help Gemma 3n E4B. `scripts/agent_eval.py` is there to find out. Run it on the same problem file twice, once with the tools and once with `--tools ""` (no tools offered, so the model answers directly), and compare the two. `--tools understand_question,examine_answer` and similar runs show which tools matter.
