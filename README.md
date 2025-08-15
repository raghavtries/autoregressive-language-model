## Autoregressive Language Model

A fully functional autoregressive decoder-only transformer language model with multi-head self-attention, causal masking, LayerNorm-first architecture, sinusoidal positional encodings, weight tying, and a training pipeline implementing AdamW optimization with weight decay, cosine learning rate scheduling with warmup, gradient accumulation, gradient clipping, AMP, and full backpropagation.

It can:
- Train on text data (I used Shakespeare)
- Generate text continuations
- Evaluate perplexity
- Run on CPU or GPU
- Scale from tiny (659K params) to larger models


---

Parts:

- **Config:**  
  YAML → dataclasses, validation, experiment control  

- **Tokenizer:**  
  Byte-level BPE (HuggingFace), special tokens, trained from scratch  

- **Data Pipeline:**  
  Streaming text, contiguous token blocks, causal masking  

- **Transformer Architecture:**  
  Multi-head self-attention with causal masking  
  LayerNorm-first (pre-norm) for stability  
  GELU activation in feed-forward layers  
  Residual connections throughout  
  Weight tying between embeddings and output projection  

- **Key Components:**  
  `SinusoidalPositionalEncoding`: Non-learned position embeddings  
  `MultiHeadAttention`: Scaled dot-product attention with causal masking  
  `TransformerBlock`: Transformer layer with attention + MLP  
  `TransformerLM`: Full model with embeddings, transformer blocks, projection  

- **Generation:**  
  Temperature/top-k/top-p sampling  
  Early stopping on EOS tokens  
  Handles edge cases (e.g., all-negative logits)  

- **Training Loop:**  
  Cross-entropy loss with shifted targets  
  AdamW optimizer with weight decay  
  Cosine LR schedule with warmup  
  Gradient accumulation for larger effective batch sizes  
  Gradient clipping to prevent exploding gradients  
  Automatic Mixed Precision (AMP) for ~2× speedup  

- **Optimizations:**  
  Gradient checkpointing, torch.compile, reproducible seeds  

- **Evaluation:**  
  Perplexity computation, GPT-2 baseline comparison  

- **Testing:**  
  Unit tests for attention, generation, weight tying, edge cases  

- **CLI Tools:**  
  `train.py`, `eval.py`, `generate.py`  

- **Performance:**  
  Tiny model (659K params) → 2 min CPU train, 148.49 perplexity  

- **Limitations:**  
  Small size, limited data, CPU-bound training  

- **Extensibility:**  
  Larger configs, better tokenization, GPU/distributed training  
