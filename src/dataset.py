import numpy as np
import torch


class TokenStreamDataset:

  def __init__(self, bin_path: str, block_size: int, dtype=np.uint16):
    self.data = np.memmap(bin_path, dtype=dtype, mode="r")
    self.block_size = block_size
    assert (
        len(self.data) > block_size + 1
    ), f"Dataset too small ({len(self.data)} tokens) for block_size={block_size}"

  def get_batch(
      self, batch_size: int, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(self.data) - self.block_size - 1
    ix = np.random.randint(0, max_start, size=(batch_size,))

    x = np.stack(
        [self.data[i : i + self.block_size] for i in ix]
    ).astype(np.int64)
    y = np.stack(
        [self.data[i + 1 : i + 1 + self.block_size] for i in ix]
    ).astype(np.int64)

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )