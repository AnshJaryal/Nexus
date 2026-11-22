from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    vocab_size: int  = 50257
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    block_size: int = 256
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_steps: int = 2000
    eval_interval: int = 200
    eval_batches: int = 20
    grad_clip: float  = 1.0
    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"
    ckpt_path: str = "checkpoints/decoder_latest.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"