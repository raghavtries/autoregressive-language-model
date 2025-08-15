#!/usr/bin/env python3
"""Clean text generation script that removes special tokens and improves readability."""

import argparse
import sys

import torch
import yaml

sys.path.insert(0, "src")

from llm_ar.generate import generate_text, load_model_from_checkpoint
from llm_ar.tokenizer import Tokenizer


def clean_text(text: str) -> str:
    """Clean generated text by removing special tokens and improving readability."""
    # Remove special tokens
    text = text.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")

    # Replace byte-level BPE tokens with proper spaces
    text = text.replace("Ġ", " ")  # Word boundary token
    text = text.replace("Ċ", "\n")  # Newline token

    # Clean up multiple spaces and newlines
    text = " ".join(text.split())

    # Remove any remaining special characters that might be artifacts
    import re

    text = re.sub(r"[^\w\s.,!?;:\'\"()-]", "", text)

    return text.strip()


def main():
    """Main function for clean text generation."""
    parser = argparse.ArgumentParser(
        description="Generate clean text from trained model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--top_k", type=int, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, help="Top-p sampling")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Convert scientific notation strings to floats
    if "train" in config and "learning_rate" in config["train"]:
        config["train"]["learning_rate"] = float(config["train"]["learning_rate"])

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Initialize tokenizer
    tokenizer = Tokenizer()
    if config.get("tokenizer_path"):
        tokenizer.load(config["tokenizer_path"])
    else:
        # Train tokenizer if not provided
        tokenizer.train([config["data"]["train_path"]], config["model"]["vocab_size"])

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    # Generate text
    raw_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.get_special_token_id("<eos>"),
    )

    # Clean the generated text
    clean_generated = clean_text(raw_text)

    # Print results
    print(f"Prompt: {args.prompt}")
    print(f"Raw generated: {raw_text}")
    print(f"Clean generated: {clean_generated}")
    print("-" * 50)


if __name__ == "__main__":
    main()
