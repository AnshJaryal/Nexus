from pathlib import Path
import pandas as pd
from data import build_token_bins
from src.config import ModelConfig, TrainConfig
from src.train import train


def load_texts_from_sources(raw_dir: str = "data/raw") -> list[str]:
  """Collect text/HTML/CSS/JS strings from raw files in data/raw."""
  texts: list[str] = []
  raw_path = Path(raw_dir)

  # 1. Any .parquet files (extract 'content' if present, else 'text')
  for pq_file in sorted(raw_path.glob("*.parquet")):
    df = pd.read_parquet(pq_file)
    col = "content" if "content" in df.columns else "text"
    if col in df.columns:
      texts.extend(df[col].dropna().astype(str).tolist())

  # 2. Any .jsonl files with a 'text' or 'content' column
  for jsonl_file in sorted(raw_path.glob("*.jsonl")):
    df = pd.read_json(jsonl_file, lines=True)
    col = "content" if "content" in df.columns else "text"
    if col in df.columns:
      texts.extend(df[col].dropna().astype(str).tolist())

  # 3. Any .txt, .html, .css, or .js files
  for ext in ("*.txt", "*.html", "*.css", "*.js"):
    for file_path in sorted(raw_path.rglob(ext)):
      content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
      if content:
        texts.append(content)

  return texts


def main():
  raw_dir = "data/raw"
  processed_dir = "data/processed"
  train_bin = f"{processed_dir}/train.bin"
  val_bin = f"{processed_dir}/val.bin"

  # Set skip_if_exists=True if you don't want to re-tokenize every run
  skip_if_exists = False

  if not (skip_if_exists and Path(train_bin).exists() and Path(val_bin).exists()):
    print("Step 1/2: Extracting and tokenizing dataset...")
    texts = load_texts_from_sources(raw_dir)
    assert len(texts) > 0, f"No training texts found in {raw_dir}/"
    build_token_bins(
        texts=texts,
        tokenizer_name="gpt2",
        val_ratio=0.05,
        out_dir=processed_dir,
    )
  else:
    print("Step 1/2: Reusing existing tokenized .bin files.")

  print("Step 2/2: Starting decoder training...")
  # Use the shared defaults in src/config.py so edits there apply to pipeline runs.
  model_cfg = ModelConfig()
  train_cfg = TrainConfig(
      train_bin=train_bin,
      val_bin=val_bin,
  )
  train(model_cfg, train_cfg)


if __name__ == "__main__":
  main()
