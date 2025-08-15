#!/usr/bin/env python3
"""Performance test script to compare different optimization configurations."""

import copy
import json
import sys
import time

import torch
import yaml

sys.path.insert(0, "src")

from llm_ar.data import TextDataset, make_dataloader
from llm_ar.model import TransformerLM
from llm_ar.tokenizer import Tokenizer
from llm_ar.utils import set_seed


def test_performance_config(
    config_name: str, config: dict, num_steps: int = 20
) -> dict:
    """Test performance with a specific configuration."""
    print(f"\n=== Testing {config_name} ===")

    # Set seed for reproducibility
    set_seed(42, deterministic=True)

    # Enable high precision matmul
    torch.set_float32_matmul_precision("high")

    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.train([config["data"]["train_path"]], config["model"]["vocab_size"])

    # Create dataset and dataloader
    dataset = TextDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        block_size=config["model"]["block_size"],
    )
    dataloader = make_dataloader(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )

    # Create model
    model = TransformerLM(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        block_size=config["model"]["block_size"],
    )

    # Apply optimizations
    use_compile = config.get("train", {}).get("use_compile", False)
    use_gradient_checkpointing = config.get("train", {}).get(
        "use_gradient_checkpointing", False
    )
    use_amp = config.get("train", {}).get("amp", False)

    print(
        f"Config: compile={use_compile}, gc={use_gradient_checkpointing}, amp={use_amp}"
    )

    if use_compile:
        model = torch.compile(model)
        print("Applied torch.compile")

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["learning_rate"]
    )

    # Initialize AMP scaler if needed
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Performance measurement
    total_tokens_processed = 0
    start_time = time.time()

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Torch compile: {use_compile}")
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")

    # Training loop
    dataloader_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = batch.to(device)
        batch_size, seq_len = batch.shape
        total_tokens_processed += batch_size * seq_len

        # Forward pass
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
        else:
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
            )

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        # Print progress
        if step % 5 == 0:
            elapsed_time = time.time() - start_time
            tokens_per_sec = (
                total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
            )
            print(
                f"Step {step}: loss={loss.item():.4f}, tokens/sec={tokens_per_sec:.1f}"
            )

    # Calculate final metrics
    total_time = time.time() - start_time
    final_tokens_per_sec = total_tokens_processed / total_time if total_time > 0 else 0

    results = {
        "config_name": config_name,
        "device": str(device),
        "amp": use_amp,
        "torch_compile": use_compile,
        "gradient_checkpointing": use_gradient_checkpointing,
        "total_time": total_time,
        "total_tokens": total_tokens_processed,
        "tokens_per_sec": final_tokens_per_sec,
        "final_loss": loss.item(),
    }

    print(f"Final tokens/sec: {final_tokens_per_sec:.1f}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens_processed:,}")

    return results


def main():
    """Main performance test function."""
    # Load base configuration
    with open("configs/tiny_test_performance.yaml") as f:
        base_config = yaml.safe_load(f)

    # Convert scientific notation strings to floats
    if "train" in base_config and "learning_rate" in base_config["train"]:
        base_config["train"]["learning_rate"] = float(
            base_config["train"]["learning_rate"]
        )

    # Test configurations
    configs = {}

    # Baseline (no optimizations)
    configs["baseline"] = copy.deepcopy(base_config)

    # With AMP only
    configs["with_amp"] = copy.deepcopy(base_config)
    configs["with_amp"]["train"]["amp"] = True

    # With compile only
    configs["with_compile"] = copy.deepcopy(base_config)
    configs["with_compile"]["train"]["use_compile"] = True

    # With gradient checkpointing only
    configs["with_gradient_checkpointing"] = copy.deepcopy(base_config)
    configs["with_gradient_checkpointing"]["train"]["use_gradient_checkpointing"] = True

    # All optimizations
    configs["all_optimizations"] = copy.deepcopy(base_config)
    configs["all_optimizations"]["train"]["amp"] = True
    configs["all_optimizations"]["train"]["use_compile"] = True
    configs["all_optimizations"]["train"]["use_gradient_checkpointing"] = True

    # Run performance tests
    results = []

    for config_name, config in configs.items():
        try:
            result = test_performance_config(config_name, config, num_steps=20)
            results.append(result)
        except Exception as e:
            print(f"Error testing {config_name}: {e}")
            results.append(
                {
                    "config_name": config_name,
                    "error": str(e),
                    "tokens_per_sec": 0,
                }
            )

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 80)

    for result in results:
        if "error" in result:
            print(f"{result['config_name']:30} | ERROR: {result['error']}")
        else:
            print(
                f"{result['config_name']:30} | {result['tokens_per_sec']:8.1f} tokens/sec | "
                f"AMP: {result['amp']} | Compile: {result['torch_compile']} | "
                f"GC: {result['gradient_checkpointing']}"
            )

    # Save results
    with open("performance_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to performance_results.json")


if __name__ == "__main__":
    main()
