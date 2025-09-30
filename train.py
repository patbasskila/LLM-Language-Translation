"""
Training Loop Plan (with validation loop + BLEU)

src → English token IDs
tgt → Spanish token IDs

We’ll feed these into the model with masks:

src_mask = make_src_mask(src, pad_idx)
tgt_mask = make_tgt_mask(tgt[:, :-1], pad_idx)

(we shift the target sequence for teacher forcing: model predicts token t+1 given tokens ≤ t).
"""

"""
TRAINING & VALIDATION LOOP FOR EN↔ES TRANSLATION
------------------------------------------------
Handles:
    1. Data loading
    2. Model initialization
    3. Training loop (forward, loss, backward, update)
    4. Validation loop with BLEU score
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sacrebleu import corpus_bleu

from data import get_dataloader, tokenizer
from transformer import Transformer, make_src_mask, make_tgt_mask, LabelSmoothingLoss
from utils import translate_sentence  # for proper autoregressive inference

# ============================================================
# 1) HYPERPARAMETERS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_VOCAB_SIZE = len(tokenizer)
TGT_VOCAB_SIZE = len(tokenizer)
PAD_IDX = tokenizer.pad_token_id

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
MAX_LEN = 96
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Toggle for resuming training
RESUME_TRAINING = True  # set False to always restart from scratch

# ============================================================
# 2) MODEL + OPTIMIZER + LOSS
# ============================================================
model = Transformer(
    src_vocab=SRC_VOCAB_SIZE,
    tgt_vocab=TGT_VOCAB_SIZE,
    d_model=384,
    N=6,
    heads=6,
    d_ff=1536,
    dropout=0.1,
    max_len=MAX_LEN
).to(DEVICE)

# Use label smoothing instead of raw CE loss for better generalization
# (can swap back to CrossEntropyLoss if desired)
criterion = LabelSmoothingLoss(TGT_VOCAB_SIZE, PAD_IDX, smoothing=0.1)

optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 3) CHECKPOINT FUNCTIONS
# ============================================================
def save_checkpoint(epoch, model, optimizer, val_loss, bleu, best=False):
    filename = "best.pt" if best else "last.pt"
    path = os.path.join(CHECKPOINT_DIR, filename)

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "bleu": bleu,
    }, path)
    print(f"[Checkpoint] Saved {filename} at epoch {epoch+1}")


def load_checkpoint(model, optimizer, best=False):
    filename = "best.pt" if best else "last.pt"
    path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"[Checkpoint] Loaded {filename} from epoch {checkpoint['epoch']+1}, "
              f"Val Loss: {checkpoint['val_loss']:.4f}, BLEU: {checkpoint['bleu']:.2f}")
        return checkpoint["epoch"], checkpoint["val_loss"], checkpoint["bleu"]
    else:
        print(f"[Checkpoint] No checkpoint found at {path}")
        return 0, float("inf"), 0.0


# ============================================================
# 4) TRAIN & VALIDATION LOOPS
# ============================================================
def train_epoch(model, optimizer, batch_size=32, max_len=64):
    """
    One training epoch:
      - Teacher forcing: model predicts token t+1 given tokens ≤ t
      - Loss averaged across all batches
    """
    model.train()
    dataloader = get_dataloader(split="train", batch_size=batch_size, max_len=max_len)
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        tgt_input = tgt[:, :-1]   # everything except last token
        tgt_labels = tgt[:, 1:]   # everything except first token

        src_mask = make_src_mask(src, PAD_IDX)
        tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Flatten for loss: (batch*seq_len, vocab_size)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_labels = tgt_labels.reshape(-1)

        loss = criterion(logits, tgt_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, batch_size=16, max_len=64):
    """
    Validation loop:
      - Computes loss on val set
      - Runs autoregressive decoding via translate_sentence()
      - Computes BLEU against ground-truth references
    """
    model.eval()
    dataloader = get_dataloader(split="validation", batch_size=batch_size, max_len=max_len)
    total_loss = 0
    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt"].to(DEVICE)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask = make_src_mask(src, PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Loss as before
            logits_reshaped = logits.reshape(-1, logits.size(-1))
            tgt_labels_reshaped = tgt_labels.reshape(-1)
            loss = criterion(logits_reshaped, tgt_labels_reshaped)
            total_loss += loss.item()

            # NEW: Use autoregressive inference instead of greedy argmax across logits
            for s, r in zip(src, tgt):
                input_text = tokenizer.decode(s, skip_special_tokens=True)
                ref_text = tokenizer.decode(r, skip_special_tokens=True)

                pred_text = translate_sentence(
                    input_text,
                    model=model,
                    tokenizer=tokenizer,
                    device=DEVICE,
                    max_len=max_len
                )

                all_preds.append(pred_text)
                all_refs.append([ref_text])

    bleu = corpus_bleu(all_preds, list(zip(*all_refs))).score
    return total_loss / len(dataloader), bleu


# ============================================================
# 5) MAIN TRAINING WITH CHECKPOINTING
# ============================================================
if __name__ == "__main__":
    if RESUME_TRAINING:
        start_epoch, best_val_loss, best_bleu = load_checkpoint(model, optimizer, best=False)
    else:
        start_epoch, best_val_loss, best_bleu = 0, float("inf"), 0.0

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_epoch(model, optimizer, batch_size=BATCH_SIZE, max_len=MAX_LEN)
        val_loss, bleu = evaluate(model, batch_size=BATCH_SIZE, max_len=MAX_LEN)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  BLEU:       {bleu:.2f}")

        # Always save last checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, bleu, best=False)

        # Save best checkpoint if BLEU improves
        if bleu > best_bleu:
            best_bleu = bleu
            save_checkpoint(epoch, model, optimizer, val_loss, bleu, best=True)


"""
How it works

At startup → calls load_checkpoint() if RESUME_TRAINING=True.
After each epoch:
    - Saves last.pt (latest model).
    - Saves best.pt if BLEU improves.

Stored data includes: epoch, model state, optimizer state, val_loss, and BLEU.

Resumption is seamless — if Colab disconnects, just rerun training, and it picks up where it left off.
"""
