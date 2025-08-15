"""Training script for language model."""

import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from llm_ar.data import create_train_val_datasets, make_dataloader
from llm_ar.model import TransformerLM
from llm_ar.tokenizer import Tokenizer
from llm_ar.utils import CSVLogger, load_checkpoint, save_checkpoint, set_seed


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, step: int):
        """Updates learning rate for current step."""
        if step < 0:
            lr = self.min_lr
        elif step < self.warmup_steps:
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            if self.max_steps == self.warmup_steps:
                lr = self.base_lr
            else:
                progress = (step - self.warmup_steps) / (
                    self.max_steps - self.warmup_steps
                )
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

        lr = max(lr, self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        """Returns scheduler state."""
        return {
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state_dict):
        """Loads scheduler state."""
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_steps = state_dict["max_steps"]
        self.min_lr = state_dict["min_lr"]
        self.base_lr = state_dict["base_lr"]


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss with shifted targets."""
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss


def train(
    model: TransformerLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    checkpoint_path: str | None = None,
    resume: bool = False,
) -> None:
    """Training loop with AMP, gradient accumulation, and checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    torch.set_float32_matmul_precision("high")

    set_seed(config.get("seed", 42), deterministic=True)

    use_compile = config.get("train", {}).get("use_compile", False)
    if use_compile:
        model = torch.compile(model)
        print("Applied torch.compile to model")

    use_gradient_checkpointing = config.get("train", {}).get(
        "use_gradient_checkpointing", False
    )
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config["train"]["warmup_steps"],
        max_steps=config["train"]["max_steps"],
        min_lr=config.get("min_lr", 0.0),
    )

    scaler = GradScaler() if config.get("amp", False) else None

    log_dir = Path(config.get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)
    logger = CSVLogger(log_dir / "training_log.csv")

    start_step = 0
    best_val_loss = float("inf")

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_step, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path
        )
        print(f"Resumed from step {start_step} with best val loss {best_val_loss}")

    model.train()
    total_loss = 0.0
    num_batches = 0

    total_tokens_processed = 0
    training_start_time = time.time()

    print(f"Starting training from step {start_step}")
    print(f"Device: {device}")
    print(f"AMP enabled: {scaler is not None}")
    print(f"Torch compile: {use_compile}")
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")

    for step in range(start_step, config["train"]["max_steps"]):
        try:
            batch = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device)

        batch_size, seq_len = batch.shape
        total_tokens_processed += batch_size * seq_len

        if scaler is not None:
            with autocast():
                logits = model(batch)
                loss = compute_loss(logits, batch)
                loss = loss / config.get("grad_accumulation_steps", 1)
        else:
            logits = model(batch)
            loss = compute_loss(logits, batch)
            loss = loss / config.get("grad_accumulation_steps", 1)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % config.get("grad_accumulation_steps", 1) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.get("grad_clip", 1.0)
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        scheduler.step(step)

        total_loss += loss.item() * config.get("grad_accumulation_steps", 1)
        num_batches += 1

        if step % config.get("log_interval", 10) == 0:
            avg_loss = total_loss / num_batches
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed_time = time.time() - training_start_time
            tokens_per_sec = (
                total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
            )

            metrics = {
                "step": step,
                "train_loss": avg_loss,
                "learning_rate": current_lr,
                "grad_norm": grad_norm.item() if "grad_norm" in locals() else 0.0,
                "val_loss": None,
                "tokens_per_sec": tokens_per_sec,
                "total_tokens": total_tokens_processed,
            }

            logger.log(metrics)

            print(
                f"Step {step}: loss={avg_loss:.4f}, lr={current_lr:.6f}, grad_norm={metrics['grad_norm']:.4f}, tokens/sec={tokens_per_sec:.1f}"
            )

            total_loss = 0.0
            num_batches = 0

        if step % config.get("eval_interval", 100) == 0 and step > 0:
            val_loss = evaluate(model, val_loader, device, config.get("amp", False))

            metrics = {
                "step": step,
                "val_loss": val_loss,
            }
            logger.log(metrics)

            print(f"Step {step}: val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = log_dir / f"best_model_step_{step}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, step, val_loss, str(checkpoint_path)
                )
                print(f"Saved best model checkpoint: {checkpoint_path}")

        if step % config.get("save_interval", 500) == 0 and step > 0:
            checkpoint_path = log_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(
                model, optimizer, scheduler, step, loss.item(), str(checkpoint_path)
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    final_val_loss = evaluate(model, val_loader, device, config.get("amp", False))
    print(f"Final validation loss: {final_val_loss:.4f}")

    total_training_time = time.time() - training_start_time
    final_tokens_per_sec = (
        total_tokens_processed / total_training_time if total_training_time > 0 else 0
    )

    print("\n=== Training Performance Summary ===")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Total tokens processed: {total_tokens_processed:,}")
    print(f"Average tokens per second: {final_tokens_per_sec:.1f}")
    print(f"Device: {device}")
    print(f"AMP enabled: {scaler is not None}")
    print(f"Torch compile: {use_compile}")
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print("=" * 40)

    final_checkpoint_path = log_dir / "final_model.pt"
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        config["train"]["max_steps"],
        final_val_loss,
        str(final_checkpoint_path),
    )
    print(f"Saved final model: {final_checkpoint_path}")

    logger.close()


def evaluate(
    model: TransformerLM,
    val_loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> float:
    """Evaluates model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            if amp:
                with autocast():
                    logits = model(batch)
                    loss = compute_loss(logits, batch)
            else:
                logits = model(batch)
                loss = compute_loss(logits, batch)

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches


def main():
    """Main training function."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Train language model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if "train" in config and "learning_rate" in config["train"]:
        config["train"]["learning_rate"] = float(config["train"]["learning_rate"])

    set_seed(config.get("seed", 42), deterministic=True)

    tokenizer = Tokenizer()
    if config.get("tokenizer_path"):
        tokenizer.load(config["tokenizer_path"])
    else:
        tokenizer.train([config["data"]["train_path"]], config["model"]["vocab_size"])

    train_dataset, val_dataset = create_train_val_datasets(
        config["data"]["train_path"],
        config["data"].get("val_path"),
        tokenizer,
        config["data"]["block_size"],
    )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )

    val_loader = make_dataloader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )

    model = TransformerLM(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        block_size=config["model"]["block_size"],
    )

    print(f"Model parameters: {model.get_num_params():,}")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
