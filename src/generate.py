import torch
from transformers import AutoTokenizer
from src.config import TrainConfig
from src.model import DecoderOnlyLM

def load_model(ckpt_path: str, device: str) -> DecoderOnlyLM:
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
  model_cfg = checkpoint["model_cfg"]

  model = DecoderOnlyLM(model_cfg).to(device)
  model.load_state_dict(checkpoint["model_state"])
  model.eval()
  return model


def generate_text(
    prompt: str,
    ckpt_path: str = "checkpoints/decoder_latest.pt",
    tokenizer_name: str = "gpt2",
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int | None = 50,
    device: str | None = None,
) -> str:
  if device is None:
    device = TrainConfig().device

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  model = load_model(ckpt_path, device)

  input_ids = tokenizer.encode(prompt, add_special_tokens=False)
  idx = torch.tensor([input_ids], dtype=torch.long, device=device)

  out_ids = model.generate(
      idx=idx,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
      top_k=top_k,
  )[0].tolist()

  return tokenizer.decode(out_ids, skip_special_tokens=False)


if __name__ == "__main__":
  sample_prompts = [
      "<!DOCTYPE html>\n<html>\n<head>\n<style>",
      ".container {",
      "Name: ",
  ]

  for p in sample_prompts:
    print("=" * 60)
    print(f"PROMPT:\n{p}")
    print("-" * 60)
    print(
        generate_text(p, max_new_tokens=120, temperature=0.8, top_k=40)
    )