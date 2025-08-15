"""Configuration management for language model."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    block_size: int

    def __post_init__(self):
        """Validates model configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int
    micro_batch_size: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    amp: bool
    grad_accumulation_steps: int = 1

    def __post_init__(self):
        """Validates training configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive")


@dataclass
class DataConfig:
    """Data configuration."""

    train_path: str
    val_path: str
    block_size: int

    def __post_init__(self):
        """Validates data configuration parameters."""
        if not Path(self.train_path).exists():
            raise ValueError(f"train_path does not exist: {self.train_path}")
        if not Path(self.val_path).exists():
            raise ValueError(f"val_path does not exist: {self.val_path}")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""

    vocab_size: int
    special_tokens: dict[str, str]

    def __post_init__(self):
        """Validates tokenizer configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        required_tokens = {"pad", "bos", "eos", "unk"}
        if not all(token in self.special_tokens for token in required_tokens):
            raise ValueError(f"special_tokens must contain: {required_tokens}")


def from_yaml(
    path: str,
) -> tuple[ModelConfig, TrainConfig, DataConfig, TokenizerConfig]:
    """Loads configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        config_dict = yaml.safe_load(f)

    model_config = config_dict.get("model", {})
    train_config = config_dict.get("train", {})
    data_config = config_dict.get("data", {})
    tokenizer_config = config_dict.get("tokenizer", {})

    if "learning_rate" in train_config:
        train_config["learning_rate"] = float(train_config["learning_rate"])

    model = ModelConfig(**model_config)
    train = TrainConfig(**train_config)
    data = DataConfig(**data_config)
    tokenizer = TokenizerConfig(**tokenizer_config)

    return model, train, data, tokenizer
