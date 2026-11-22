from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def build_token_bins(
    texts: list[str],
    tokenizer_name: str = "gpt2",
    val_ratio: float = 0.05,
    out_dir: str = "data/processed",
):
  Path(out_dir).mkdir(parents=True, exist_ok=True)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  eos_id = tokenizer.eos_token_id

  token_chunks = []
  for doc in texts:
    if not doc or not doc.strip():
      continue
    ids = tokenizer.encode(doc, add_special_tokens=False)
    ids.append(eos_id)
    token_chunks.append(np.array(ids, dtype=np.uint16))

  full_stream = np.concatenate(token_chunks)
  split_idx = int(len(full_stream) * (1.0 - val_ratio))

  train_arr = full_stream[:split_idx]
  val_arr = full_stream[split_idx:]

  train_arr.tofile(f"{out_dir}/train.bin")
  val_arr.tofile(f"{out_dir}/val.bin")

  print(f"Saved {len(train_arr):,} train tokens and {len(val_arr):,} val tokens.")


if __name__ == "__main__":
  # Example: extract 'content' (or 'text') column from a Parquet file
  df = pd.read_parquet("data/raw/sample.parquet", columns=["content"]).dropna()
  build_token_bins(df["content"].tolist())