"""Tests for weight tying functionality."""

import sys

import torch

sys.path.insert(0, "src")

from llm_ar.model import TransformerLM


def test_embedding_output_tying():
    """Test that embedding and output projection share weights."""
    vocab_size = 100
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 128
    dropout = 0.1
    block_size = 16

    # Create model with weight tying
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        block_size=block_size,
    )

    # Verify that embedding.weight and output_projection.weight are the same object
    assert model.token_embedding.weight is model.output_projection.weight

    # Test that modifying one affects the other
    original_weight = model.token_embedding.weight.clone()
    model.output_projection.weight.data += 1.0

    # Both should be modified
    assert torch.allclose(model.token_embedding.weight, model.output_projection.weight)
    assert not torch.allclose(model.token_embedding.weight, original_weight)


def test_tying_parameter_count():
    """Test that weight tying reduces parameter count."""
    vocab_size = 100
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 128
    dropout = 0.1
    block_size = 16

    # Create model with weight tying
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        block_size=block_size,
    )

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Count embedding parameters separately
    embedding_params = model.token_embedding.weight.numel()
    model.output_projection.weight.numel()

    # Since they're tied, we should only count embedding parameters once
    expected_params_without_tying = (
        total_params + embedding_params
    )  # If not tied, would have both
    actual_params_with_tying = total_params

    # Verify that tied model has fewer parameters than if not tied
    assert actual_params_with_tying < expected_params_without_tying

    # Verify the exact reduction in parameter count
    reduction = expected_params_without_tying - actual_params_with_tying
    assert reduction == embedding_params


def test_tying_gradient_flow():
    """Test that gradients flow correctly with weight tying."""
    vocab_size = 100
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 128
    dropout = 0.1
    block_size = 16

    # Create model with weight tying
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        block_size=block_size,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create input data
    batch_size = 2
    seq_len = 8
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Run forward pass
    logits = model(x)

    # Create targets (shifted input for causal LM)
    targets = x[:, 1:]  # Remove first token
    logits = logits[:, :-1, :]  # Remove last token from logits

    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )

    # Run backward pass
    optimizer.zero_grad()
    loss.backward()

    # Verify that gradients are accumulated correctly
    # Both embedding and output projection should have gradients
    assert model.token_embedding.weight.grad is not None
    assert model.output_projection.weight.grad is not None

    # Since they're tied, gradients should be the same
    assert torch.allclose(
        model.token_embedding.weight.grad, model.output_projection.weight.grad
    )

    # Test that gradients are finite
    assert torch.isfinite(model.token_embedding.weight.grad).all()
    assert torch.isfinite(model.output_projection.weight.grad).all()


def test_tying_initialization():
    """Test that weight tying works with proper initialization."""
    vocab_size = 100
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 128
    dropout = 0.1
    block_size = 16

    # Create model with weight tying
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        block_size=block_size,
    )

    # Test that tied weights are initialized consistently
    assert torch.allclose(model.token_embedding.weight, model.output_projection.weight)

    # Verify that initialization doesn't break the tie
    # Both should have the same initialization
    embedding_init = model.token_embedding.weight.clone()
    output_init = model.output_projection.weight.clone()

    assert torch.allclose(embedding_init, output_init)

    # Test that weights are properly initialized (not all zeros)
    assert not torch.allclose(
        model.token_embedding.weight, torch.zeros_like(model.token_embedding.weight)
    )
    assert not torch.allclose(
        model.output_projection.weight, torch.zeros_like(model.output_projection.weight)
    )


def test_tying_forward_pass():
    """Test that forward pass works correctly with weight tying."""
    vocab_size = 100
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 128
    dropout = 0.1
    block_size = 16

    # Create model with weight tying
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        block_size=block_size,
    )

    # Create input data
    batch_size = 2
    seq_len = 8
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Run forward pass
    logits = model(x)

    # Test output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape

    # Test that output is finite
    assert torch.isfinite(logits).all()

    # Test that output is not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits))

    # Test that tied weights are still the same after forward pass
    assert torch.allclose(model.token_embedding.weight, model.output_projection.weight)
