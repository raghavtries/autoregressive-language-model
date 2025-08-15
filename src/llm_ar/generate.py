"""Text generation CLI."""

import argparse
import json

import torch
import yaml

from llm_ar.model import TransformerLM
from llm_ar.tokenizer import Tokenizer
from llm_ar.utils import set_seed


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: dict,
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


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> str:
    """Generates text from prompt."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    block_size = model.block_size
    if len(input_ids) > block_size:
        input_ids = input_ids[-block_size:]

    input_tensor = torch.tensor(
        [input_ids], dtype=torch.long, device=next(model.parameters()).device
    )

    with torch.no_grad():
        available_space = block_size - len(input_ids)
        actual_max_tokens = min(max_new_tokens, available_space)

        generated_ids = model.generate(
            input_tensor,
            max_new_tokens=actual_max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids[0].tolist())

    return generated_text


def main():
    """CLI for text generation."""
    parser = argparse.ArgumentParser(description="Generate text with language model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")
    parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p sampling parameter"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_prompts",
        action="store_true",
        help="Treat prompt as newline-separated batch",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file for results"
    )
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

    model = load_model_from_checkpoint(args.checkpoint, config, device)
    model = model.to(device)
    model.eval()

    eos_token_id = tokenizer.get_special_token_id("<eos>")

    if args.batch_prompts:
        prompts = args.prompt.strip().split("\n")
        prompts = [p.strip() for p in prompts if p.strip()]
    else:
        prompts = [args.prompt]

    results = []
    for i, prompt in enumerate(prompts):
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=eos_token_id,
            )

            result = {
                "prompt": prompt,
                "generated_text": generated_text,
                "generation_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "seed": args.seed,
                },
            }
            results.append(result)

            if len(prompts) > 1:
                print(f"=== Prompt {i + 1} ===")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print()

        except Exception as e:
            print(f"Error generating text for prompt {i + 1}: {e}")
            results.append(
                {
                    "prompt": prompt,
                    "error": str(e),
                    "generation_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "seed": args.seed,
                    },
                }
            )

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")

    if len(prompts) > 1:
        successful = sum(1 for r in results if "error" not in r)
        print(f"Generated text for {successful}/{len(prompts)} prompts")


if __name__ == "__main__":
    main()
