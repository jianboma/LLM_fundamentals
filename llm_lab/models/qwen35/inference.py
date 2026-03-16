from __future__ import annotations

import sys
from pathlib import Path
from threading import Thread
from typing import Any, Generator

import torch

from .hf_loader import load_hf_qwen3_5_for_inference


def _to_device_inputs(inputs: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


class Qwen3_5InferenceEngine:
    """
    Inference engine with two decoding backends:
    - native: uses local reimplementation only (no `transformers` generation code path)
    - transformers: uses HF `generate()` for full algorithm support
    """

    def __init__(
        self,
        weights_dir: str | Path,
        backend: str = "native",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        assistant_model_dir: str | Path | None = None,
    ):
        self.weights_dir = Path(weights_dir).resolve()
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.assistant_model_dir = Path(assistant_model_dir).resolve() if assistant_model_dir is not None else None
        self.report: dict[str, Any] = {}
        self.model = None
        self.config = None
        self.tokenizer = None
        self.assistant_model = None

        if backend == "native":
            self._init_native()
        elif backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError("backend must be one of: native, transformers")

    @staticmethod
    def _native_generate_kwargs(generation_kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "num_return_sequences",
            "seed",
            "eos_token_id",
            "use_cache",
        }
        return {k: v for k, v in generation_kwargs.items() if k in allowed}

    def _init_native(self):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("Native backend requires `tokenizers`. Install with `pip install tokenizers`.") from exc

        tokenizer_path = self.weights_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        model, config, report = load_hf_qwen3_5_for_inference(
            weights_dir=self.weights_dir,
            device=self.device,
            dtype=self.dtype,
            strict=False,
        )
        self.model = model
        self.config = config
        self.report = {
            "backend": "native",
            **report,
        }

    def _init_transformers(self):
        # llm_lab/models/qwen35/inference.py -> repo root is parents[3]
        project_root = Path(__file__).resolve().parents[3]
        local_transformers_src = project_root / "third_party" / "transformers" / "src"
        if str(local_transformers_src) not in sys.path:
            sys.path.insert(0, str(local_transformers_src))

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[reportMissingImports]
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "unknown")
            raise ImportError(
                "Transformers backend is unavailable because dependency "
                f"`{missing}` is missing. Install required deps in your env, e.g. "
                "`pip install regex huggingface_hub safetensors tokenizers`."
            ) from exc
        except ImportError as exc:
            raise ImportError(
                "Transformers backend requires local `third_party/transformers` importable dependencies."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.weights_dir,
            torch_dtype=self.dtype,
        )
        model = model.to(self.device)
        model.eval()
        self.model = model
        self.config = model.config

        if self.assistant_model_dir is not None:
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                self.assistant_model_dir,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.assistant_model.eval()

        self.report = {
            "backend": "transformers",
            "model_class": model.__class__.__name__,
            "assistant_model": str(self.assistant_model_dir) if self.assistant_model_dir is not None else None,
        }

    @staticmethod
    def normalize_generation_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(raw_kwargs)
        if kwargs.get("max_tokens") is not None and kwargs.get("max_new_tokens") is None:
            kwargs["max_new_tokens"] = int(kwargs.pop("max_tokens"))
        if kwargs.get("n") is not None and kwargs.get("num_return_sequences") is None:
            kwargs["num_return_sequences"] = int(kwargs.pop("n"))
        if kwargs.get("temperature") == 0:
            kwargs["do_sample"] = False

        clean = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            clean[k] = v
        return clean

    def generate_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> torch.LongTensor:
        generation_kwargs = generation_kwargs or {}
        generation_kwargs = self.normalize_generation_kwargs(generation_kwargs)
        seed = generation_kwargs.pop("seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))

        if self.backend == "native":
            generation_kwargs = self._native_generate_kwargs(generation_kwargs)
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask
            if generation_kwargs.get("eos_token_id") is None:
                generation_kwargs["eos_token_id"] = getattr(self.config, "eos_token_id", None)
            return self.model.generate(input_ids=input_ids, **generation_kwargs)

        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        if self.assistant_model is not None and generation_kwargs.get("assistant_model") is None:
            generation_kwargs["assistant_model"] = self.assistant_model
        return self.model.generate(input_ids=input_ids, **generation_kwargs)

    def stream_text(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        generation_kwargs = generation_kwargs or {}
        generation_kwargs = self.normalize_generation_kwargs(generation_kwargs)
        seed = generation_kwargs.pop("seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))

        if self.backend == "native":
            yield from self._stream_text_native(prompt, generation_kwargs)
        else:
            yield from self._stream_text_transformers(prompt, generation_kwargs)

    def _stream_text_native(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any],
    ) -> Generator[str, None, None]:
        if generation_kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("native streaming only supports num_return_sequences=1.")
        if generation_kwargs.get("num_beams", 1) != 1:
            raise ValueError("native backend does not support beam search streaming.")

        generate_kwargs = self._native_generate_kwargs(generation_kwargs)
        max_new_tokens = int(generate_kwargs.get("max_new_tokens", 128))
        do_sample = bool(generate_kwargs.get("do_sample", False))
        temperature = float(generate_kwargs.get("temperature", 1.0))
        top_k = generate_kwargs.get("top_k")
        top_p = generate_kwargs.get("top_p")
        repetition_penalty = float(generate_kwargs.get("repetition_penalty", 1.0))
        presence_penalty = float(generate_kwargs.get("presence_penalty", 0.0))
        frequency_penalty = float(generate_kwargs.get("frequency_penalty", 0.0))
        use_cache = bool(generate_kwargs.get("use_cache", True))
        eos_token_id = generate_kwargs.get("eos_token_id")
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        generated = input_ids
        generated_attention_mask = attention_mask
        cache = None

        for _ in range(max_new_tokens):
            if cache is None:
                model_input_ids = generated
                model_attention_mask = generated_attention_mask
            else:
                model_input_ids = generated[:, -1:]
                model_attention_mask = torch.ones_like(model_input_ids)

            outputs = self.model.forward(
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,
                past_key_values=cache,
                use_cache=use_cache,
            )
            cache = outputs.past_key_values
            next_token = self.model._sample_next_token(
                outputs.logits[:, -1, :],
                generated=generated,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            generated = torch.cat([generated, next_token], dim=-1)
            generated_attention_mask = torch.cat(
                [generated_attention_mask, torch.ones_like(next_token, dtype=generated_attention_mask.dtype)],
                dim=-1,
            )
            next_token_id = int(next_token[0, 0].item())
            token_text = self.tokenizer.decode([next_token_id])
            if token_text:
                yield token_text
            if eos_token_id is not None and next_token_id == int(eos_token_id):
                break

    def _stream_text_transformers(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any],
    ) -> Generator[str, None, None]:
        if generation_kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("streaming only supports num_return_sequences=1.")
        if generation_kwargs.get("num_beams", 1) > 1:
            raise ValueError("streaming with transformers backend currently requires num_beams=1.")

        project_root = Path(__file__).resolve().parents[3]
        local_transformers_src = project_root / "third_party" / "transformers" / "src"
        if str(local_transformers_src) not in sys.path:
            sys.path.insert(0, str(local_transformers_src))

        try:
            from transformers import TextIteratorStreamer  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError("Transformers backend streaming requires TextIteratorStreamer.") from exc

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = _to_device_inputs(encoded, self.device)
        kwargs = dict(generation_kwargs)
        kwargs["streamer"] = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        if self.assistant_model is not None and kwargs.get("assistant_model") is None:
            kwargs["assistant_model"] = self.assistant_model

        thread = Thread(target=self.model.generate, kwargs={**encoded, **kwargs}, daemon=True)
        thread.start()
        streamer = kwargs["streamer"]
        for text in streamer:
            if text:
                yield text
        thread.join()

    def generate_text(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any] | None = None,
        return_full_text: bool = False,
    ) -> dict[str, Any]:
        generation_kwargs = generation_kwargs or {}

        if self.backend == "native":
            encoded = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            generated_ids = self.generate_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_kwargs=generation_kwargs,
            )
            prompt_len = input_ids.shape[1]
            sequences = generated_ids.tolist()
            if return_full_text:
                texts = [self.tokenizer.decode(seq) for seq in sequences]
            else:
                texts = [self.tokenizer.decode(seq[prompt_len:]) for seq in sequences]
            return {
                "generated_ids": generated_ids,
                "texts": texts,
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(generated_ids.shape[1] - prompt_len),
            }

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = _to_device_inputs(encoded, self.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        generated_ids = self.generate_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_kwargs=generation_kwargs,
        )
        prompt_len = int(input_ids.shape[1])
        sequences = generated_ids.tolist()
        if return_full_text:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        else:
            texts = self.tokenizer.batch_decode([seq[prompt_len:] for seq in sequences], skip_special_tokens=True)
        return {
            "generated_ids": generated_ids,
            "texts": texts,
            "prompt_tokens": prompt_len,
            "completion_tokens": int(generated_ids.shape[1] - prompt_len),
        }
