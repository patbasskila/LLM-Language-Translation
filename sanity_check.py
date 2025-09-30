"""
SANITY CHECK SCRIPT
-------------------
Quick test to verify:
  1. Dataset loads correctly (English ↔ Spanish)
  2. Tokenization + padding works
  3. Dataloader returns proper batch shapes
  4. Masks behave correctly (pad masking + no-peak masking)
"""

import torch
from data import get_dataloader, tokenizer
from transformer import make_src_mask, make_tgt_mask  # reuse your mask functions

# Load a small batch
loader = get_dataloader(split="train", batch_size=2, max_len=10)
batch = next(iter(loader))

print("== Sanity Check ==")
print("Source IDs (EN):", batch["src"].shape)
print("Target IDs (ES):", batch["tgt"].shape)

# Decode sample text
print("\nSample Pair:")
src_ids = batch["src"][0]
tgt_ids = batch["tgt"][0]

print("EN:", tokenizer.decode(src_ids, skip_special_tokens=True))
print("ES:", tokenizer.decode(tgt_ids, skip_special_tokens=True))

# Pad token index (needed for masks)
pad_idx = tokenizer.pad_token_id

# Create masks
src_mask = make_src_mask(batch["src"], pad_idx)
tgt_mask = make_tgt_mask(batch["tgt"], pad_idx)

print("\n== Mask Shapes ==")
print("src_mask:", src_mask.shape)  # (batch, 1, src_len)
print("tgt_mask:", tgt_mask.shape)  # (batch, tgt_len, tgt_len)

# Show actual mask values for the first example
print("\nSource Mask (EN):")
print(src_mask[0][0])  # shape: (src_len)

print("\nTarget Mask (ES):")
print(tgt_mask[0])     # shape: (tgt_len, tgt_len)
