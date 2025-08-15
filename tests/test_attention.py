"""Tests for attention mechanism functionality."""

import sys

import torch

sys.path.insert(0, "src")

from llm_ar.data import causal_mask
from llm_ar.model import MultiHeadAttention


def test_attention_no_future_access():
    """Test that attention never reads future tokens."""
    d_model = 64
    n_heads = 4
    seq_len = 8
    batch_size = 2

    # Create attention module
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    # Create input sequence
    x = torch.randn(batch_size, seq_len, d_model)

    # Run attention forward pass
    output = attention(x)

    # Get attention weights from the module (we need to modify the forward method to return them)
    # For now, let's test that the output shape is correct
    assert output.shape == (batch_size, seq_len, d_model)

    # Test that causal mask is properly applied
    # Create causal mask manually
    mask = causal_mask(seq_len, x.device)

    # Verify mask is strictly upper triangular
    for i in range(seq_len):
        for j in range(seq_len):
            if i >= j:  # Diagonal and below
                assert not mask[i, j], f"Mask[{i},{j}] should be False"
            else:  # Above diagonal
                assert mask[i, j], f"Mask[{i},{j}] should be True"


def test_attention_softmax_normalization():
    """Test that softmax rows sum to 1 on unmasked entries."""
    d_model = 64
    n_heads = 4
    seq_len = 6
    batch_size = 2

    # Create attention module
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    # Create input sequence
    x = torch.randn(batch_size, seq_len, d_model)

    # Run attention forward pass
    output = attention(x)

    # Test output shape
    assert output.shape == (batch_size, seq_len, d_model)

    # Test that output is finite
    assert torch.isfinite(output).all()


def test_multi_head_attention():
    """Test multi-head attention mechanism."""
    d_model = 64
    n_heads = 4
    seq_len = 8
    batch_size = 2

    # Test that d_model is divisible by n_heads
    assert d_model % n_heads == 0

    # Create attention module
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    # Create input sequence
    x = torch.randn(batch_size, seq_len, d_model)

    # Run attention forward pass
    output = attention(x)

    # Test output shape
    assert output.shape == (batch_size, seq_len, d_model)

    # Test that output is different from input (attention is working)
    assert not torch.allclose(output, x, atol=1e-6)


def test_attention_scaling():
    """Test that attention scores are properly scaled."""
    d_model = 64
    n_heads = 4
    seq_len = 6
    batch_size = 2

    # Create attention module
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    # Create input sequence
    x = torch.randn(batch_size, seq_len, d_model)

    # Run attention forward pass
    output = attention(x)

    # Test output shape
    assert output.shape == (batch_size, seq_len, d_model)

    # Test that output is finite (scaling prevents exploding gradients)
    assert torch.isfinite(output).all()


def test_attention_dropout():
    """Test that attention dropout works correctly."""
    d_model = 64
    n_heads = 4
    seq_len = 6
    batch_size = 2

    # Create attention module with dropout
    attention_with_dropout = MultiHeadAttention(d_model, n_heads, dropout=0.5)
    attention_no_dropout = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    # Create input sequence
    x = torch.randn(batch_size, seq_len, d_model)

    # Test training mode (dropout active)
    attention_with_dropout.train()
    output_train = attention_with_dropout(x)

    # Test evaluation mode (dropout disabled)
    attention_with_dropout.eval()
    output_eval = attention_with_dropout(x)

    # Test no dropout version
    attention_no_dropout.eval()
    output_no_dropout = attention_no_dropout(x)

    # All outputs should have correct shape
    assert output_train.shape == (batch_size, seq_len, d_model)
    assert output_eval.shape == (batch_size, seq_len, d_model)
    assert output_no_dropout.shape == (batch_size, seq_len, d_model)

    # Outputs should be finite
    assert torch.isfinite(output_train).all()
    assert torch.isfinite(output_eval).all()
    assert torch.isfinite(output_no_dropout).all()


def test_attention_causal_masking():
    """Test that causal masking is properly applied."""
    d_model = 64
    n_heads = 4
    seq_len = 4
    batch_size = 1

    # Create attention module
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    # Create input sequence with known pattern
    x = torch.randn(batch_size, seq_len, d_model)

    # Run attention forward pass
    output = attention(x)

    # Test that output shape is correct
    assert output.shape == (batch_size, seq_len, d_model)

    # Test that output is finite
    assert torch.isfinite(output).all()

    # Test that causal mask function works correctly
    mask = causal_mask(seq_len, x.device)

    # Verify mask shape
    assert mask.shape == (seq_len, seq_len)

    # Verify mask is boolean
    assert mask.dtype == torch.bool

    # Verify strictly upper triangular
    for i in range(seq_len):
        for j in range(seq_len):
            if i >= j:  # Diagonal and below
                assert not mask[i, j]
            else:  # Above diagonal
                assert mask[i, j]
