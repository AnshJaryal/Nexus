# src/train.py
from pathlib import Path
import torch
from src.config import ModelConfig, TrainConfig
from src.dataset import TokenStreamDataset
from src.model import DecoderOnlyLM


@torch.no_grad()
def estimate_loss(
    model: DecoderOnlyLM,
    train_ds: TokenStreamDataset,
    val_ds: TokenStreamDataset,
    train_cfg: TrainConfig,
) -> dict[str, float]:
  model.eval()
  out = {}
  for split, ds in [("train", train_ds), ("val", val_ds)]:
    losses = []
    for _ in range(train_cfg.eval_batches):
      xb, yb = ds.get_batch(train_cfg.batch_size, train_cfg.device)
      _, loss = model(xb, yb)
      losses.append(loss.item())
    out[split] = sum(losses) / len(losses)
  model.train()
  return out


def train(model_cfg: ModelConfig, train_cfg: TrainConfig):
  Path(train_cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

  train_ds = TokenStreamDataset(train_cfg.train_bin, model_cfg.block_size)
  val_ds = TokenStreamDataset(train_cfg.val_bin, model_cfg.block_size)

  model = DecoderOnlyLM(model_cfg).to(train_cfg.device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

  print(
      f"Training on {train_cfg.device} | batch_size={train_cfg.batch_size} | "
      f"block_size={model_cfg.block_size} | Params:"
      f" {sum(p.numel() for p in model.parameters()):,}"
  )

  for step in range(1, train_cfg.max_steps + 1):
    if step % train_cfg.eval_interval == 0 or step == 1:
      metrics = estimate_loss(model, train_ds, val_ds, train_cfg)
      print(
          f"Step {step:4d}/{train_cfg.max_steps} | train loss:"
          f" {metrics['train']:.4f} | val loss: {metrics['val']:.4f}"
      )

    xb, yb = train_ds.get_batch(train_cfg.batch_size, train_cfg.device)
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
    optimizer.step()

  torch.save(
      {
          "model_state": model.state_dict(),
          "model_cfg": model_cfg,
      },
      train_cfg.ckpt_path,
  )
  print(f"Checkpoint saved to {train_cfg.ckpt_path}")

if __name__ == "__main__":
  train(ModelConfig(), TrainConfig())
