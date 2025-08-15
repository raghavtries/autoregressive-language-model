"""Evaluation script for computing perplexity."""

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from llm_ar.data import create_train_val_datasets, make_dataloader
from llm_ar.model import TransformerLM
from llm_ar.tokenizer import Tokenizer
from llm_ar.utils import set_seed


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss with shifted targets."""
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    return loss


def evaluate(
    model: TransformerLM,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluates model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            logits = model(batch)
            loss_per_token = compute_loss(logits, batch)
            num_tokens = loss_per_token.numel()

            total_loss += loss_per_token.sum().item()
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"val_loss": avg_loss, "perplexity": perplexity}


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: dict[str, Any],
    device: torch.device,
) -> TransformerLM:
    """Loads model from checkpoint."""
    model = TransformerLM(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        block_size=config["model"]["block_size"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def main():
    """CLI for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate language model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if "train" in config and "learning_rate" in config["train"]:
        config["train"]["learning_rate"] = float(config["train"]["learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer()
    if config.get("tokenizer_path"):
        tokenizer.load(config["tokenizer_path"])
    else:
        tokenizer.train([config["data"]["train_path"]], config["model"]["vocab_size"])

    _, val_dataset = create_train_val_datasets(
        config["data"]["train_path"],
        config["data"].get("val_path"),
        tokenizer,
        config["data"]["block_size"],
    )

    val_loader = make_dataloader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
        shuffle=False,
    )

    model = load_model_from_checkpoint(args.checkpoint, config, device)
    model = model.to(device)

    metrics = evaluate(model, val_loader, device)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
