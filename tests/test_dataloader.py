"""Tests for dataloader functionality."""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, "src")

from llm_ar.data import TextDataset, make_dataloader
from llm_ar.tokenizer import Tokenizer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    text = """
    This is a sample text for testing the dataloader.
    It contains multiple sentences and various words.
    The dataloader should be able to process this text.
    """

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def trained_tokenizer(sample_data):
    """Create a trained tokenizer for testing."""
    tokenizer = Tokenizer()
    tokenizer.train([sample_data], vocab_size=100)
    return tokenizer


def test_dataloader_batch_shape(trained_tokenizer, sample_data):
    """Test that dataloader produces batches with correct shape [B, T]."""
    block_size = 8
    batch_size = 2

    # Create dataset and dataloader
    dataset = TextDataset(sample_data, trained_tokenizer, block_size)
    dataloader = make_dataloader(dataset, batch_size, shuffle=False)

    # Get a batch
    batch = next(iter(dataloader))

    # Check shape
    assert batch.shape == (
        batch_size,
        block_size,
    ), f"Expected shape ({batch_size}, {block_size}), got {batch.shape}"


def test_dataloader_batch_dtype(trained_tokenizer, sample_data):
    """Test that dataloader produces batches with dtype long."""
    block_size = 8
    batch_size = 2

    # Create dataset and dataloader
    dataset = TextDataset(sample_data, trained_tokenizer, block_size)
    dataloader = make_dataloader(dataset, batch_size, shuffle=False)

    # Get a batch
    batch = next(iter(dataloader))

    # Check dtype
    assert batch.dtype == torch.long, f"Expected dtype torch.long, got {batch.dtype}"


def test_dataloader_different_batch_sizes(trained_tokenizer, sample_data):
    """Test dataloader with different batch sizes."""
    block_size = 8

    for batch_size in [1, 2, 4]:
        dataset = TextDataset(sample_data, trained_tokenizer, block_size)
        dataloader = make_dataloader(dataset, batch_size, shuffle=False)

        batch = next(iter(dataloader))
        assert (
            batch.shape[0] == batch_size
        ), f"Expected batch size {batch_size}, got {batch.shape[0]}"


def test_dataloader_different_block_sizes(trained_tokenizer, sample_data):
    """Test dataloader with different block sizes."""
    batch_size = 2

    for block_size in [4, 8, 16]:
        dataset = TextDataset(sample_data, trained_tokenizer, block_size)
        dataloader = make_dataloader(dataset, batch_size, shuffle=False)

        batch = next(iter(dataloader))
        assert (
            batch.shape[1] == block_size
        ), f"Expected block size {block_size}, got {batch.shape[1]}"


def test_dataloader_shuffle(trained_tokenizer, sample_data):
    """Test that dataloader shuffle works correctly."""
    block_size = 8
    batch_size = 2

    dataset = TextDataset(sample_data, trained_tokenizer, block_size)

    # Test without shuffle
    dataloader_no_shuffle = make_dataloader(dataset, batch_size, shuffle=False)
    batch_no_shuffle = next(iter(dataloader_no_shuffle))

    # Test with shuffle
    dataloader_shuffle = make_dataloader(dataset, batch_size, shuffle=True)
    batch_shuffle = next(iter(dataloader_shuffle))

    # Both should have correct shape and dtype
    assert batch_no_shuffle.shape == (batch_size, block_size)
    assert batch_shuffle.shape == (batch_size, block_size)
    assert batch_no_shuffle.dtype == torch.long
    assert batch_shuffle.dtype == torch.long


def test_dataloader_drop_last(trained_tokenizer, sample_data):
    """Test that dataloader drops incomplete batches."""
    block_size = 8
    batch_size = 10  # Large batch size to ensure incomplete batches

    dataset = TextDataset(sample_data, trained_tokenizer, block_size)
    dataloader = make_dataloader(dataset, batch_size, shuffle=False)

    # Count batches
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        # All batches should have full batch_size
        assert (
            batch.shape[0] == batch_size
        ), f"Expected batch size {batch_size}, got {batch.shape[0]}"

    # Should have at least one batch
    assert batch_count > 0, "Dataloader should produce at least one batch"


def test_dataset_length(trained_tokenizer, sample_data):
    """Test that dataset length is calculated correctly."""
    block_size = 8

    dataset = TextDataset(sample_data, trained_tokenizer, block_size)

    # Length should be positive
    assert len(dataset) > 0, "Dataset length should be positive"

    # Length should be reasonable (depends on text length and block_size)
    assert len(dataset) <= 100, "Dataset length should be reasonable"


def test_dataset_item_access(trained_tokenizer, sample_data):
    """Test that dataset items can be accessed correctly."""
    block_size = 8

    dataset = TextDataset(sample_data, trained_tokenizer, block_size)

    # Test first item
    first_item = dataset[0]
    assert first_item.shape == (
        block_size,
    ), f"Expected shape ({block_size},), got {first_item.shape}"
    assert (
        first_item.dtype == torch.long
    ), f"Expected dtype torch.long, got {first_item.dtype}"

    # Test last item
    last_item = dataset[len(dataset) - 1]
    assert last_item.shape == (
        block_size,
    ), f"Expected shape ({block_size},), got {last_item.shape}"
    assert (
        last_item.dtype == torch.long
    ), f"Expected dtype torch.long, got {last_item.dtype}"
