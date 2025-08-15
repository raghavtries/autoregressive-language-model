"""Byte-level BPE tokenizer implementation."""

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


class Tokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(self):
        """Initializes tokenizer with special tokens."""
        self.tokenizer = None
        self.special_tokens = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def train(self, corpus_paths: list[str], vocab_size: int) -> None:
        """Trains byte-level BPE tokenizer on corpus with special tokens."""
        tokenizer = HFTokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            show_progress=True,
        )

        tokenizer.train(corpus_paths, trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A:0 $B:1 <eos>",
            special_tokens=[
                ("<bos>", self.special_tokens["<bos>"]),
                ("<eos>", self.special_tokens["<eos>"]),
            ],
        )

        self.tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encodes text to token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs to text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        return self.tokenizer.decode(ids)

    def save(self, path: str) -> None:
        """Saves tokenizer to file."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        """Loads tokenizer from file."""
        self.tokenizer = HFTokenizer.from_file(path)

    @property
    def vocab_size(self) -> int:
        """Returns vocabulary size."""
        if self.tokenizer is None:
            return len(self.special_tokens)
        return self.tokenizer.get_vocab_size()

    def get_special_token_id(self, token: str) -> int | None:
        """Returns ID for special token."""
        if self.tokenizer is None:
            return self.special_tokens.get(token)

        return self.tokenizer.token_to_id(token)
