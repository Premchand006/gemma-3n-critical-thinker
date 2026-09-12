"""Text-generation backends used by the agent.

A backend turns a list of chat messages (``{"role": ..., "content": ...}``)
into one reply string. ``HFBackend`` runs a Hugging Face model locally.
``ScriptedBackend`` returns preset replies and is used by the tests.

torch, transformers and peft are imported inside ``HFBackend.__init__`` so
this module can be imported on a machine without them.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]


@runtime_checkable
class Backend(Protocol):
    def generate(
        self,
        messages: list[Message],
        max_new_tokens: int = 1024,
        stop: Sequence[str] | None = None,
        **kw: Any,
    ) -> str:
        """Return the model's reply to ``messages``.

        ``stop`` lists strings at which generation should end. Backends that
        cannot stop on strings may ignore it; the orchestrator also cuts the
        reply after the first tool call.
        """
        ...


Reply = str | Callable[[list[Message]], str]


class ScriptedBackend:
    """Return preset replies in order. For tests and dry runs.

    Each reply is a string or a function of the messages. ``calls`` records
    every request so tests can check what was sent. ``stop`` is ignored.
    Raises ``IndexError`` when the script runs out.
    """

    def __init__(self, replies: Iterable[Reply]):
        self._replies: list[Reply] = list(replies)
        self._next = 0
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[Message],
        max_new_tokens: int = 1024,
        stop: Sequence[str] | None = None,
        **kw: Any,
    ) -> str:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_new_tokens": max_new_tokens,
                "stop": list(stop) if stop else None,
                **kw,
            }
        )
        if self._next >= len(self._replies):
            raise IndexError("ScriptedBackend ran out of replies")
        reply = self._replies[self._next]
        self._next += 1
        return reply(messages) if callable(reply) else reply

    @property
    def remaining(self) -> int:
        return len(self._replies) - self._next


class HFBackend:
    """Local Hugging Face model with an optional LoRA adapter.

    Args:
        model_id: Base model id or local path, for example
            ``"unsloth/gemma-3n-E4B-it"`` or ``"google/gemma-3n-E4B-it"``.
        adapter: Optional PEFT adapter id or path, for example
            ``"anhsirk/gemma-3n-critical-thinker"``.
        load_in_4bit: Load the base weights in 4-bit NF4 with bitsandbytes
            (needs a CUDA GPU).
        temperature: Sampling temperature. 0 means greedy decoding.
        top_p: Nucleus sampling cutoff, used when sampling.
        device_map: Passed to ``from_pretrained``.

    The Hugging Face token, if one is needed for gated weights, is read from
    the ``HF_TOKEN`` environment variable. It is never stored or written.
    """

    def __init__(
        self,
        model_id: str,
        adapter: str | None = None,
        load_in_4bit: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.95,
        device_map: str | dict[str, Any] = "auto",
    ):
        import torch
        import transformers
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.adapter = adapter
        self.temperature = temperature
        self.top_p = top_p
        token = os.environ.get("HF_TOKEN") or None

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        load_kwargs: dict[str, Any] = {"device_map": device_map, "token": token}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            load_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except ValueError:
            # Multimodal checkpoints such as Gemma 3n may only be registered
            # under the image-text-to-text auto class in some versions.
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                model_id, **load_kwargs
            )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter, token=token)
        model.eval()
        self.model = model
        self._torch = torch

    def generate(
        self,
        messages: list[Message],
        max_new_tokens: int = 1024,
        stop: Sequence[str] | None = None,
        **kw: Any,
    ) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        temperature = kw.pop("temperature", self.temperature)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=self.top_p)
        else:
            gen_kwargs["do_sample"] = False
        if stop:
            gen_kwargs.update(stop_strings=list(stop), tokenizer=self.tokenizer)
        gen_kwargs.update(kw)
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


__all__ = ["Backend", "HFBackend", "Message", "ScriptedBackend"]
