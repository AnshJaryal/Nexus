"""Stream a small WebSight sample from Hugging Face into a local Parquet file."""

import argparse
from pathlib import Path

from datasets import Dataset, Features, Image, Value, load_dataset


DATASET_NAME = "HuggingFaceM4/WebSight"
DEFAULT_OUTPUT = Path("data/processed/sample.parquet")


def fetch_sample(num_rows: int, output_path: Path) -> None:
    """Fetch at most ``num_rows`` examples without downloading the full dataset."""
    if num_rows < 1:
        raise ValueError("num_rows must be at least 1")

    stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    rows = list(stream.take(num_rows))
    if not rows:
        raise RuntimeError("The dataset stream returned no rows.")

    # Declare the known WebSight schema so image values are encoded as Parquet
    # image structs (bytes + path), while the text fields remain strings.
    features = Features(
        {
            "image": Image(),
            "text": Value("string"),
            "llm_generated_idea": Value("string"),
        }
    )
    dataset = Dataset.from_list(rows, features=features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    print(f"Saved {len(dataset)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--num-rows", type=int, default=30,
        help="number of rows to fetch (default: 30)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"output Parquet path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    fetch_sample(args.num_rows, args.output)


if __name__ == "__main__":
    main()
