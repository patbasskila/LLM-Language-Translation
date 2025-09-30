"""
MAIN ENTRY POINT
----------------
Command-line interface for:
    1. Training the Transformer model
    2. Evaluating on the validation set
    3. Translating custom sentences (EN -> ES)
"""

import argparse
import torch

from train import train_epoch, evaluate, save_checkpoint, load_checkpoint
from data import tokenizer
from transformer import Transformer, make_src_mask, make_tgt_mask
from utils import translate_sentence

# ============================================================
# HYPERPARAMETERS (default values; can be overridden by CLI args)
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = tokenizer.pad_token_id


def build_model(d_model=256, N=4, heads=8, d_ff=512, dropout=0.1, max_len=64):
    """Helper to construct Transformer with tokenizer vocab sizes"""
    return Transformer(
        src_vocab=len(tokenizer),
        tgt_vocab=len(tokenizer),
        d_model=d_model,
        N=N,
        heads=heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len
    ).to(DEVICE)


def main():
    parser = argparse.ArgumentParser(description="Transformer EN↔ES CLI")
    parser.add_argument("--mode", choices=["train", "eval", "translate"], required=True,
                        help="train: run training loop | eval: evaluate validation set | translate: translate text")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--best", action="store_true", help="Load best checkpoint instead of last")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--text", type=str, help="Sentence to translate if mode=translate")
    args = parser.parse_args()

    # Build model
    model = build_model()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Load checkpoint if resume flag is set
    start_epoch, best_val_loss, best_bleu = 0, float("inf"), 0.0
    if args.resume:
        start_epoch, best_val_loss, best_bleu = load_checkpoint(model, optimizer, best=args.best)

    # Mode handling
    if args.mode == "train":
        print(f"[Training] Starting at epoch {start_epoch+1} for {args.epochs} total epochs...")
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_epoch(model, optimizer, args.batch_size)
            val_loss, bleu = evaluate(model, args.batch_size)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu:.2f}")
            save_checkpoint(epoch, model, optimizer, val_loss, bleu, best=False)
            if bleu > best_bleu:
                best_bleu = bleu
                save_checkpoint(epoch, model, optimizer, val_loss, bleu, best=True)

    elif args.mode == "eval":
        val_loss, bleu = evaluate(model, args.batch_size)
        print(f"[Eval] Validation Loss: {val_loss:.4f}, BLEU: {bleu:.2f}")

    elif args.mode == "translate":
        if not args.text:
            raise ValueError("Please provide --text for translation mode")
        translation = translate_sentence(args.text, model, max_len=64)
        print(f"EN: {args.text}")
        print(f"ES: {translation}")


if __name__ == "__main__":
    main()
