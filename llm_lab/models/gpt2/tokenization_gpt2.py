from __future__ import annotations

import os
from pathlib import Path

import torch

try:
    from tokenizers import Tokenizer as HFTokenizer
except ImportError as exc:
    raise ImportError(
        "tokenization_gpt2 requires the `tokenizers` package. Install with `pip install tokenizers`."
    ) from exc


class GPT2Tokenizer:
    """
    Lightweight tokenizer wrapper with the same interface used by nanochat loaders.
    """

    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path: str) -> "GPT2Tokenizer":
        return cls(HFTokenizer.from_pretrained(hf_path))

    @classmethod
    def from_file(cls, tokenizer_path: str | Path) -> "GPT2Tokenizer":
        return cls(HFTokenizer.from_file(str(tokenizer_path)))

    @classmethod
    def from_directory(cls, tokenizer_dir: str | Path) -> "GPT2Tokenizer":
        tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
        return cls.from_file(tokenizer_path)

    def get_vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode_special(self, token_text: str) -> int | None:
        return self.tokenizer.token_to_id(token_text)

    def get_bos_token_id(self) -> int:
        bos = self.encode_special("<|bos|>")
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        if bos is None:
            raise ValueError("Could not resolve BOS token id from tokenizer vocabulary.")
        return int(bos)

    def _encode_one(
        self,
        text: str,
        prepend: int | str | None = None,
        append: int | str | None = None,
    ) -> list[int]:
        token_ids: list[int] = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            if prepend_id is None:
                raise ValueError(f"Unknown prepend token: {prepend}")
            token_ids.append(int(prepend_id))
        token_ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            if append_id is None:
                raise ValueError(f"Unknown append token: {append}")
            token_ids.append(int(append_id))
        return token_ids

    def encode(
        self,
        text: str | list[str],
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int | None = None,
    ) -> list[int] | list[list[int]]:
        del num_threads
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        if isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        raise TypeError(f"Unsupported input type for encode: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def get_tokenizer(base_dir: str | Path | None = None) -> GPT2Tokenizer:
    if base_dir is None:
        from nanochat.common import get_base_dir

        base_dir = get_base_dir()
    tokenizer_path = Path(base_dir) / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    return GPT2Tokenizer.from_file(tokenizer_path)


def get_token_bytes(device: str | torch.device = "cpu", base_dir: str | Path | None = None) -> torch.Tensor:
    if base_dir is None:
        from nanochat.common import get_base_dir

        base_dir = get_base_dir()
    token_bytes_path = Path(base_dir) / "tokenizer" / "token_bytes.pt"
    if not token_bytes_path.exists():
        raise FileNotFoundError(f"token_bytes.pt not found: {token_bytes_path}")
    with open(os.fspath(token_bytes_path), "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
