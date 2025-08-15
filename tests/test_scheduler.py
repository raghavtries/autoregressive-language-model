"""Tests for learning rate scheduler functionality."""

import math
import sys

import torch

sys.path.insert(0, "src")

from llm_ar.train import CosineWarmupScheduler


def test_warmup_then_cosine():
    """Test that warmup then cosine schedule values match formula."""
    target_lr = 0.001
    warmup_steps = 100
    max_steps = 1000

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=target_lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Test learning rate at different steps
    for step in [0, 50, 100, 200, 500, 1000]:
        scheduler.step(step)
        lr = optimizer.param_groups[0]["lr"]

        if step <= warmup_steps:
            # During warmup: lr = target_lr * (step / warmup_steps)
            expected_lr = target_lr * (step / warmup_steps)
        else:
            # During cosine decay: lr = target_lr * 0.5 * (1 + cos(pi * (step - warmup_steps) / (max_steps - warmup_steps)))
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            expected_lr = target_lr * 0.5 * (1 + math.cos(math.pi * progress))

        assert (
            abs(lr - expected_lr) < 1e-6
        ), f"Step {step}: expected {expected_lr}, got {lr}"


def test_warmup_phase():
    """Test learning rate during warmup phase."""
    target_lr = 0.001
    warmup_steps = 100
    max_steps = 1000

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=target_lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Test warmup phase
    for step in range(0, warmup_steps + 1, 10):
        scheduler.step(step)
        lr = optimizer.param_groups[0]["lr"]
        expected_lr = target_lr * (step / warmup_steps)

        assert (
            abs(lr - expected_lr) < 1e-6
        ), f"Step {step}: expected {expected_lr}, got {lr}"

    # Test that lr increases linearly
    scheduler.step(0)
    lr_0 = optimizer.param_groups[0]["lr"]
    scheduler.step(50)
    lr_50 = optimizer.param_groups[0]["lr"]
    scheduler.step(100)
    lr_100 = optimizer.param_groups[0]["lr"]

    assert lr_0 == 0.0
    assert abs(lr_50 - target_lr * 0.5) < 1e-6
    assert abs(lr_100 - target_lr) < 1e-6


def test_cosine_decay_phase():
    """Test learning rate during cosine decay phase."""
    target_lr = 0.001
    warmup_steps = 100
    max_steps = 1000

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=target_lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Test cosine decay phase
    for step in range(warmup_steps, max_steps + 1, 100):
        scheduler.step(step)
        lr = optimizer.param_groups[0]["lr"]
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        expected_lr = target_lr * 0.5 * (1 + math.cos(math.pi * progress))

        assert (
            abs(lr - expected_lr) < 1e-6
        ), f"Step {step}: expected {expected_lr}, got {lr}"

    # Test that lr decreases monotonically
    scheduler.step(100)
    lr_100 = optimizer.param_groups[0]["lr"]  # End of warmup
    scheduler.step(200)
    lr_200 = optimizer.param_groups[0]["lr"]
    scheduler.step(500)
    lr_500 = optimizer.param_groups[0]["lr"]
    scheduler.step(1000)
    lr_1000 = optimizer.param_groups[0]["lr"]  # End of training

    assert lr_100 == target_lr
    assert lr_200 < lr_100
    assert lr_500 < lr_200
    assert lr_1000 < lr_500

    # Test that final lr is close to minimum
    assert abs(lr_1000 - target_lr * 0.5 * (1 + math.cos(math.pi))) < 1e-6


def test_scheduler_step():
    """Test scheduler step functionality."""
    target_lr = 0.001
    warmup_steps = 100
    max_steps = 1000

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=target_lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Test step() method
    for step in range(10):
        # Step the scheduler
        scheduler.step(step)

        # Check that optimizer lr was updated
        lr = optimizer.param_groups[0]["lr"]
        assert lr >= 0.0  # Should be non-negative
        assert lr <= target_lr  # Should not exceed target lr during warmup

    # Test edge case when step exceeds max_steps
    scheduler.step(max_steps + 10)
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr >= 0.0  # Should be non-negative


def test_scheduler_state():
    """Test scheduler state saving and loading."""
    target_lr = 0.001
    warmup_steps = 100
    max_steps = 1000

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=target_lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Test state_dict
    state = scheduler.state_dict()
    assert "warmup_steps" in state
    assert "max_steps" in state
    assert "min_lr" in state
    assert "base_lr" in state
    assert state["warmup_steps"] == warmup_steps
    assert state["max_steps"] == max_steps
    assert state["base_lr"] == target_lr

    # Test load_state_dict
    new_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=0.0)
    new_scheduler = CosineWarmupScheduler(new_optimizer, 0, 0)
    new_scheduler.load_state_dict(state)

    # Verify that loaded scheduler produces same lr values
    for step in [0, 50, 100, 200, 500, 1000]:
        scheduler.step(step)
        lr_original = optimizer.param_groups[0]["lr"]

        new_scheduler.step(step)
        lr_loaded = new_optimizer.param_groups[0]["lr"]
        assert abs(lr_original - lr_loaded) < 1e-6

    # Test state consistency across save/load cycles
    state2 = new_scheduler.state_dict()
    assert state2["warmup_steps"] == warmup_steps
    assert state2["max_steps"] == max_steps
    assert state2["base_lr"] == target_lr


def test_scheduler_edge_cases():
    """Test scheduler edge cases."""
    target_lr = 0.001
    warmup_steps = 100
    max_steps = 1000

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=target_lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Test negative step
    scheduler.step(-10)
    lr_negative = optimizer.param_groups[0]["lr"]
    assert lr_negative >= 0.0  # Should be non-negative

    # Test step beyond max_steps
    scheduler.step(max_steps + 100)
    lr_beyond = optimizer.param_groups[0]["lr"]
    # Should be very close to min_lr (0.0) due to clamping
    assert lr_beyond >= 0.0 and lr_beyond < 1e-4

    # Test warmup_steps = 0
    optimizer_no_warmup = torch.optim.AdamW(
        [torch.nn.Parameter(torch.randn(10))], lr=target_lr
    )
    scheduler_no_warmup = CosineWarmupScheduler(optimizer_no_warmup, 0, max_steps)
    scheduler_no_warmup.step(0)
    lr_no_warmup = optimizer_no_warmup.param_groups[0]["lr"]
    expected_no_warmup = target_lr * 0.5 * (1 + math.cos(0))
    assert abs(lr_no_warmup - expected_no_warmup) < 1e-6

    # Test warmup_steps = max_steps
    optimizer_all_warmup = torch.optim.AdamW(
        [torch.nn.Parameter(torch.randn(10))], lr=target_lr
    )
    scheduler_all_warmup = CosineWarmupScheduler(
        optimizer_all_warmup, max_steps, max_steps
    )
    scheduler_all_warmup.step(max_steps)
    lr_all_warmup = optimizer_all_warmup.param_groups[0]["lr"]
    assert abs(lr_all_warmup - target_lr) < 1e-6


def test_scheduler_monotonicity():
    """Test that learning rate changes monotonically."""
    target_lr = 0.001
    warmup_steps = 100
    max_steps = 1000

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=target_lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Test monotonicity during warmup
    prev_lr = -1
    for step in range(warmup_steps + 1):
        scheduler.step(step)
        lr = optimizer.param_groups[0]["lr"]
        assert (
            lr >= prev_lr
        ), f"LR should be non-decreasing during warmup at step {step}"
        prev_lr = lr

    # Test monotonicity during decay
    prev_lr = float("inf")
    for step in range(warmup_steps, max_steps + 1):
        scheduler.step(step)
        lr = optimizer.param_groups[0]["lr"]
        assert lr <= prev_lr, f"LR should be non-increasing during decay at step {step}"
        prev_lr = lr
