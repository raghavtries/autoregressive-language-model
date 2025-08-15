#!/usr/bin/env python3
"""Baseline comparison script: evaluate our model vs HF gpt2."""

import argparse
import json

import torch
import yaml
from llm_ar.data import TextDataset, make_dataloader
from llm_ar.model import TransformerLM
from llm_ar.tokenizer import Tokenizer
from llm_ar.utils import set_seed
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_our_model(
    checkpoint_path: str, config: dict, device: torch.device
) -> TransformerLM:
    """Load our trained model from checkpoint."""
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


def load_gpt2_model(device: torch.device) -> GPT2LMHeadModel:
    """Load HF GPT-2 model."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model.to(device)


def compute_loss(model, dataloader, device, is_gpt2=False):
    """Compute average loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # For causal LM, input is batch[:, :-1], targets are batch[:, 1:]
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            if is_gpt2:
                # GPT-2 expects input_ids and labels
                outputs = model(input_ids=input_ids, labels=targets)
                loss = outputs.loss
            else:
                # Our model expects input_ids and returns logits
                logits = model(input_ids)
                # For causal LM, logits predict the next token
                # So logits[i] should predict targets[i]
                # No need to shift logits since our model already handles this

                # Compute cross-entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                loss = loss_fct(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )

            # Count tokens (excluding padding if any)
            num_tokens = (targets != 0).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return avg_loss


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return torch.exp(torch.tensor(loss)).item()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare our model with HF GPT-2 baseline"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to our model checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed, deterministic=True)

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Convert scientific notation strings to floats
    if "train" in config and "learning_rate" in config["train"]:
        config["train"]["learning_rate"] = float(config["train"]["learning_rate"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize our tokenizer
    our_tokenizer = Tokenizer()
    if config.get("tokenizer_path"):
        our_tokenizer.load(config["tokenizer_path"])
    else:
        # Train tokenizer if not provided
        our_tokenizer.train(
            [config["data"]["train_path"]], config["model"]["vocab_size"]
        )

    # Create validation dataset and dataloader
    val_dataset = TextDataset(
        data_path=args.val_data,
        tokenizer=our_tokenizer,
        block_size=config["model"]["block_size"],
    )
    val_dataloader = make_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for simplicity
        pin_memory=False,
        shuffle=False,
    )

    print(f"Validation dataset size: {len(val_dataset)} blocks")
    print(f"Validation dataloader batches: {len(val_dataloader)}")

    # Load our model
    print("Loading our model...")
    our_model = load_our_model(args.checkpoint, config, device)
    our_model = our_model.to(device)
    our_model.eval()

    # Load GPT-2 model
    print("Loading GPT-2 model...")
    gpt2_model = load_gpt2_model(device)
    gpt2_model.eval()

    # Note: We can't directly compare perplexities because:
    # 1. Different tokenizers (our BPE vs GPT-2's tokenizer)
    # 2. Different vocabularies
    # 3. Different training data

    # For a fair comparison, we need to use the same tokenizer
    # Let's use GPT-2's tokenizer for both models
    print("Loading GPT-2 tokenizer...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    # Create a new dataset with GPT-2 tokenizer
    def create_gpt2_dataset(data_path: str, tokenizer, block_size: int):
        """Create dataset using GPT-2 tokenizer."""
        with open(data_path, encoding="utf-8") as f:
            text = f.read()

        # Tokenize the entire text
        tokens = tokenizer.encode(text)

        # Create blocks
        num_blocks = len(tokens) // block_size
        return num_blocks, tokens[: num_blocks * block_size]

    # Create GPT-2 tokenized dataset
    gpt2_num_blocks, gpt2_tokens = create_gpt2_dataset(
        args.val_data, gpt2_tokenizer, config["model"]["block_size"]
    )

    print(f"GPT-2 tokenized blocks: {gpt2_num_blocks}")

    # For a fair comparison, let's create a simple test that uses the same tokenization approach
    # Since we can't easily adapt our model to use GPT-2's tokenizer, let's just report both results
    # and note that this is not a direct comparison due to different tokenizers

    print("Evaluating our model...")
    our_loss = compute_loss(our_model, val_dataloader, device, is_gpt2=False)
    our_perplexity = compute_perplexity(our_loss)

    # For GPT-2, we need to create a custom evaluation loop
    # since we can't use our dataloader (different tokenization)
    print("Evaluating GPT-2 model...")
    gpt2_model.eval()
    gpt2_total_loss = 0.0
    gpt2_total_tokens = 0

    with torch.no_grad():
        for i in range(gpt2_num_blocks):
            start_idx = i * config["model"]["block_size"]
            end_idx = start_idx + config["model"]["block_size"]
            block_tokens = gpt2_tokens[start_idx:end_idx]

            # Convert to tensor
            input_ids = torch.tensor([block_tokens], dtype=torch.long, device=device)

            # For causal LM, input is input_ids[:, :-1], targets are input_ids[:, 1:]
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Forward pass
            outputs = gpt2_model(input_ids=inputs, labels=targets)
            loss = outputs.loss

            # Count tokens
            num_tokens = (targets != gpt2_tokenizer.eos_token_id).sum().item()
            gpt2_total_loss += loss.item() * num_tokens
            gpt2_total_tokens += num_tokens

    gpt2_avg_loss = (
        gpt2_total_loss / gpt2_total_tokens if gpt2_total_tokens > 0 else float("inf")
    )
    gpt2_perplexity = compute_perplexity(gpt2_avg_loss)

    # Calculate relative difference
    if gpt2_perplexity > 0:
        relative_diff = ((our_perplexity - gpt2_perplexity) / gpt2_perplexity) * 100
    else:
        relative_diff = float("inf")

    # Prepare results
    results = {
        "our_model": {"loss": our_loss, "perplexity": our_perplexity},
        "gpt2_baseline": {"loss": gpt2_avg_loss, "perplexity": gpt2_perplexity},
        "comparison": {
            "relative_difference_percent": relative_diff,
            "within_target_range": abs(relative_diff) <= 15.0,
        },
        "config": {
            "checkpoint": args.checkpoint,
            "config_file": args.config,
            "val_data": args.val_data,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }

    # Print results
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 60)
    print("Our Model:")
    print(f"  Loss: {our_loss:.6f}")
    print(f"  Perplexity: {our_perplexity:.2f}")
    print("\nGPT-2 Baseline:")
    print(f"  Loss: {gpt2_avg_loss:.6f}")
    print(f"  Perplexity: {gpt2_perplexity:.2f}")
    print("\nComparison:")
    print(f"  Relative Difference: {relative_diff:+.2f}%")
    print(f"  Within Target Range (≤15%): {'✅' if abs(relative_diff) <= 15.0 else '❌'}")
    print("\nNote: This comparison uses different tokenizers and vocabularies.")
    print("      For a fair comparison, both models should use the same tokenizer.")
    print("=" * 60)

    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
