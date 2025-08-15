"""Dataset and data loading utilities."""

import os

import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Streaming text dataset producing contiguous token blocks."""

    def __init__(self, data_path: str, tokenizer, block_size: int):
        """Initializes dataset with tokenized text blocks."""
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(data_path, encoding="utf-8") as f:
            text = f.read()

        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.num_blocks = max(0, len(self.tokens) - block_size + 1)

    def __len__(self) -> int:
        """Returns dataset length."""
        return self.num_blocks

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns token block at index."""
        if idx >= self.num_blocks:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.num_blocks}"
            )

        start_idx = idx
        end_idx = start_idx + self.block_size
        block_tokens = self.tokens[start_idx:end_idx]

        return torch.tensor(block_tokens, dtype=torch.long)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Creates dataloader with specified parameters."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Creates causal mask for attention (upper triangle)."""
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    return mask


def create_train_val_datasets(
    train_path: str,
    val_path: str,
    tokenizer,
    block_size: int,
    val_split: float = 0.1,
) -> tuple[TextDataset, TextDataset]:
    """Creates train and validation datasets."""
    if val_path and os.path.exists(val_path):
        train_dataset = TextDataset(train_path, tokenizer, block_size)
        val_dataset = TextDataset(val_path, tokenizer, block_size)
    else:
        train_dataset = TextDataset(train_path, tokenizer, block_size)
        val_dataset = TextDataset(train_path, tokenizer, block_size)

    return train_dataset, val_dataset
