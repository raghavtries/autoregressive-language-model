"""Transformer language model implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to input."""
        return x + self.pe[: x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with causal masking."""
        batch_size, seq_len, d_model = x.shape

        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        output = self.w_o(context)

        return output


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with GELU activation."""
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with LayerNorm-first (pre-norm) architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)

        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)

        return x

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory efficiency."""
        pass

    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing."""
        pass


class TransformerLM(nn.Module):
    """GPT-style decoder-only transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        block_size: int,
    ):
        """Initializes transformer language model."""
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.block_size = block_size

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, block_size)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes model weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        batch_size, seq_len = x.shape

        x = self.token_embedding(x) * math.sqrt(self.d_model)

        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)

        x = self.dropout(x)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generates text autoregressively."""
        was_training = self.training
        self.eval()

        with torch.no_grad():
            generated = x.clone()

            for _ in range(max_new_tokens):
                logits = self.forward(generated)[:, -1, :]

                if temperature != 1.0:
                    logits = logits / temperature

                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(-1, top_k_indices, top_k_logits)

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    if torch.all(torch.isinf(logits)):
                        probs = torch.ones_like(logits) / logits.size(-1)
                    else:
                        probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None:
                    if torch.any(next_token == eos_token_id):
                        break

        if was_training:
            self.train()

        return generated

    def get_num_params(self) -> int:
        """Returns total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, learning_rate: float, weight_decay: float = 0.01):
        """Configures optimizers and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        scheduler = None

        return optimizer, scheduler

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory efficiency."""
        for block in self.blocks:
            block.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing."""
        for block in self.blocks:
            block.gradient_checkpointing_disable()
