from __future__ import annotations

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


try:
    from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
except ImportError as exc:
    raise ImportError(
        "tokenization_qwen3_5 requires the `tokenizers` package. "
        "Install it with `pip install tokenizers`."
    ) from exc


class Qwen3_5Tokenizer:
    """
    Lightweight tokenizer backend matching Qwen3.5 BPE pre-tokenization behavior.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        unk_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        add_prefix_space: bool = False,
    ):
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.add_prefix_space = add_prefix_space
        self._vocab = vocab if vocab is not None else {"<|endoftext|>": 0}
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.normalizer = normalizers.NFC()
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(PRETOKENIZE_REGEX),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space,
                    use_regex=False,
                ),
            ]
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoding = self._tokenizer.encode(text)
        ids = list(encoding.ids)
        if add_special_tokens:
            ids.append(self._vocab.get(self.eos_token, 0))
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def __call__(self, text: str) -> dict[str, list[int]]:
        input_ids = self.encode(text, add_special_tokens=False)
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
