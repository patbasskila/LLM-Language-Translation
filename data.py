"""
Purpose

Prepare bilingual dataset (English ↔ Spanish).
Handle tokenization (via Hugging Face tokenizers or custom).
Create Dataset and DataLoader classes for PyTorch.
Apply source/target padding and build masks (using functions from transformer.py).
"""

# data.py
"""
1. Imports (torch, datasets, tokenizer, etc.)
2. TranslationDataset class
   - Loads parallel sentences
   - Applies tokenizer
   - Pads sequences
   - Returns tensors
3. DataLoader utility function
   - Wraps dataset with batching
   - Returns train/val iterators
"""

"""
DATA PIPELINE FOR ENGLISH ↔ SPANISH TRANSLATION
-----------------------------------------------
This file handles:
    1. Loading a bilingual dataset (English ↔ Spanish)
    2. Tokenizing sentences using a shared subword tokenizer
    3. Padding sequences and creating batches
    4. Adding BOS (start-of-sequence) and EOS (end-of-sequence) tokens
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# ============================================================
# 1) LOAD TOKENIZER + DEFINE SPECIAL TOKENS
# ============================================================
# Using a pretrained SentencePiece tokenizer trained for EN–ES
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# Ensure BOS/EOS tokens exist in tokenizer
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({"bos_token": "<s>"})
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"eos_token": "</s>"})

PAD_IDX = tokenizer.pad_token_id
BOS_IDX = tokenizer.bos_token_id
EOS_IDX = tokenizer.eos_token_id


# ============================================================
# 2) TRANSLATION DATASET CLASS
# ============================================================
class TranslationDataset(Dataset):
    """
    A PyTorch Dataset for English↔Spanish translation.
    Returns tokenized + padded source (EN) and target (ES) sequences.
    """

    def __init__(self, dataset_split, max_len=128):
        # dataset_split will be a HuggingFace Dataset object (train or validation)
        self.dataset = dataset_split
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract English + Spanish text
        # NOTE: opus100 uses the same "translation" dict as opus_books
        src_text = self.dataset[idx]["translation"]["en"]
        tgt_text = self.dataset[idx]["translation"]["es"]

        # Add BOS/EOS to the target sentence
        tgt_text = f"{tokenizer.bos_token} {tgt_text} {tokenizer.eos_token}"

        # Tokenize with truncation + padding
        src = tokenizer(src_text, max_length=self.max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        tgt = tokenizer(tgt_text, max_length=self.max_len, padding="max_length",
                        truncation=True, return_tensors="pt")

        # Remove batch dimension (keep shape [seq_len])
        src_ids = src["input_ids"].squeeze(0)
        tgt_ids = tgt["input_ids"].squeeze(0)

        return {
            "src": src_ids,  # encoder input (EN)
            "tgt": tgt_ids,  # decoder target (ES with BOS/EOS)
        }


# ============================================================
# 3) DATALOADER FUNCTION WITH AUTOMATIC TRAIN/VALIDATION SPLIT
# ============================================================
# Load the opus100 dataset (EN–ES subset)
_raw_dataset = load_dataset("opus100", "en-es")

# Create a 90/10 train-validation split
# This ensures evaluate() won't break looking for a missing "validation" split.
_split_dataset = _raw_dataset["train"].train_test_split(test_size=0.1, seed=42)

# Now we have _split_dataset["train"] and _split_dataset["test"]
# We'll alias "test" as "validation" for clarity
_split_dataset["validation"] = _split_dataset.pop("test")


def get_dataloader(split="train", batch_size=32, max_len=128):
    """
    Create a PyTorch DataLoader for either the train or validation split.
    """
    dataset = TranslationDataset(_split_dataset[split], max_len=max_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  # shuffle only for training
        num_workers=2,
        pin_memory=True,
        drop_last= True
    )
