"""Tests for causal mask functionality."""

import sys

import torch

sys.path.insert(0, "src")

from llm_ar.data import causal_mask


def test_causal_mask_shape():
    """Test that causal mask has correct shape."""
    # Test with different sequence lengths
    for T in [4, 8, 16, 32]:
        mask = causal_mask(T, torch.device("cpu"))
        assert mask.shape == (
            T,
            T,
        ), f"Mask shape should be ({T}, {T}), got {mask.shape}"


def test_causal_mask_strictly_upper_triangular():
    """Test that causal mask is strictly upper triangular."""
    T = 5
    mask = causal_mask(T, torch.device("cpu"))

    # Test that mask is True only above the main diagonal
    upper_triangle = torch.triu(mask, diagonal=1)
    assert torch.all(
        upper_triangle == mask
    ), "Mask should be True only above main diagonal"

    # Test that diagonal and below are all False
    lower_triangle_and_diagonal = torch.tril(mask, diagonal=0)
    assert torch.all(~lower_triangle_and_diagonal), "Diagonal and below should be False"


def test_causal_mask_device():
    """Test that causal mask works on different devices."""
    T = 4

    # Test mask creation on CPU
    mask_cpu = causal_mask(T, torch.device("cpu"))
    assert mask_cpu.device == torch.device("cpu")

    # Test mask creation on GPU if available
    if torch.cuda.is_available():
        mask_gpu = causal_mask(T, torch.device("cuda"))
        assert mask_gpu.device == torch.device("cuda")
        # Verify masks are identical on different devices
        assert torch.all(mask_cpu == mask_gpu.cpu())


def test_causal_mask_attention_blocking():
    """Test that causal mask properly blocks future attention."""
    T = 4
    mask = causal_mask(T, torch.device("cpu"))

    # Create a simple attention matrix
    attention_scores = torch.randn(T, T)

    # Apply causal mask to attention scores
    masked_scores = attention_scores.masked_fill(mask, float("-inf"))

    # Verify that masked positions have -inf
    assert torch.all(
        torch.isinf(masked_scores[mask])
    ), "Masked positions should have -inf"

    # Verify that unmasked positions retain original values
    unmasked_positions = ~mask
    assert torch.allclose(
        masked_scores[unmasked_positions], attention_scores[unmasked_positions]
    ), "Unmasked positions should retain original values"


def test_causal_mask_edge_cases():
    """Test causal mask with edge cases."""
    # Test with T=1
    mask_1 = causal_mask(1, torch.device("cpu"))
    assert mask_1.shape == (1, 1)
    assert not mask_1[0, 0]  # Single position should not be masked

    # Test with T=2
    mask_2 = causal_mask(2, torch.device("cpu"))
    assert mask_2.shape == (2, 2)
    assert not mask_2[0, 0]  # Diagonal should be False
    assert mask_2[0, 1]  # Upper triangle should be True
    assert not mask_2[1, 0]  # Lower triangle should be False
    assert not mask_2[1, 1]  # Diagonal should be False
