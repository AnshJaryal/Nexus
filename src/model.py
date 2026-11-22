import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig


class CausalSelfAttention(nn.Module):

  def __init__(self, cfg: ModelConfig):
    super().__init__()
    assert (
        cfg.d_model % cfg.n_heads == 0
    ), "d_model must be divisible by n_heads"
    self.n_heads = cfg.n_heads
    self.head_dim = cfg.d_model // cfg.n_heads

    self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    self.attn_drop = nn.Dropout(cfg.dropout)
    self.resid_drop = nn.Dropout(cfg.dropout)

    # Lower-triangular causal mask: (1, 1, block_size, block_size)
    mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
        1, 1, cfg.block_size, cfg.block_size
    )
    self.register_buffer("causal_mask", mask)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape

    q = self.q_proj(x)  # (B, T, C)
    k = self.k_proj(x)  # (B, T, C)
    v = self.v_proj(x)  # (B, T, C)

    # Reshape to (B, n_heads, T, head_dim)
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(
        self.head_dim
    )  # (B, nh, T, T)
    scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    weights = self.attn_drop(weights)

    out = weights @ v  # (B, nh, T, head_dim)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):

  def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(cfg.d_model, cfg.d_ff),
        nn.GELU(),
        nn.Linear(cfg.d_ff, cfg.d_model),
        nn.Dropout(cfg.dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class DecoderBlock(nn.Module):

  def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.ln1 = nn.LayerNorm(cfg.d_model)
    self.attn = CausalSelfAttention(cfg)
    self.ln2 = nn.LayerNorm(cfg.d_model)
    self.mlp = FeedForward(cfg)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class DecoderOnlyLM(nn.Module):

  def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.cfg = cfg
    self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
    self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
    self.drop = nn.Dropout(cfg.dropout)

    self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
    self.ln_f = nn.LayerNorm(cfg.d_model)
    self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)


    self.lm_head.weight = self.token_emb.weight

  def forward(
      self, idx: torch.Tensor, targets: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T = idx.shape
    assert (
        T <= self.cfg.block_size
    ), f"Sequence length {T} exceeds block_size {self.cfg.block_size}"

    positions = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
    x = self.drop(self.token_emb(idx) + self.pos_emb(positions))

    for block in self.blocks:
      x = block(x)

    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    loss = None
    if targets is not None:
      # ignore_index=-100 lets you mask prompt tokens during fine-tuning if desired
      loss = F.cross_entropy(
          logits.view(-1, logits.size(-1)),
          targets.view(-1),
          ignore_index=-100,
      )

    return logits, loss

  @torch.no_grad()
  def generate(
      self,
      idx: torch.Tensor,
      max_new_tokens: int,
      temperature: float = 0.8,
      top_k: int | None = 50,
  ) -> torch.Tensor:
    self.eval()
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.cfg.block_size :]
      logits, _ = self(idx_cond)
      next_logits = logits[:, -1, :] / max(temperature, 1e-6)

      if top_k is not None:
        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits[next_logits < v[:, [-1]]] = float("-inf")

      probs = F.softmax(next_logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, next_token), dim=1)
    return idx