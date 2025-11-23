English–Spanish Transformer (PyTorch, From Scratch)

This project implements a sequence-to-sequence Transformer model from scratch in PyTorch for English → Spanish translation.

Instead of using nn.Transformer or a ready-made translation model, the architecture, training loop, masking logic, and decoding are all implemented manually to deepen understanding of attention-based models and practical NLP training.

Features

From-scratch Transformer encoder–decoder

Token + sinusoidal positional embeddings

Multi-head self-attention and encoder–decoder cross-attention

Residual connections + LayerNorm + position-wise feed-forward layers

Real bilingual data

English–Spanish parallel corpus (via Hugging Face datasets)

Subword tokenization using a pre-trained Helsinki-NLP/opus-mt-en-es SentencePiece tokenizer

Training pipeline

Teacher forcing (predict next token given previous tokens)

Label smoothing loss

Noam-style warmup / inverse-sqrt learning rate schedule (optional)

Gradient clipping for stability

Training / validation split with BLEU evaluation

Decoding & evaluation

Autoregressive beam search decoding

BLEU score via sacrebleu

CLI + utilities

Train, evaluate, and translate from the command line

Sanity check script for dataset, dataloader, and masks

Jupyter notebook with the full pipeline and explanations

Project Structure
.
├── data.py                     # Dataset, tokenization, DataLoader helpers
├── transformer.py              # Transformer model + masks + label smoothing
├── train.py                    # Training & evaluation loop, BLEU computation
├── main.py                     # CLI entry point (train / eval / translate)
├── utils.py                    # Translation helper (beam search decoding)
├── sanity_check.py             # Quick checks for data + masks + shapes
└── Translation_Transformer.ipynb  # Notebook walkthrough of the full pipeline

Requirements

Typical dependencies (adjust to match your requirements.txt):
python >= 3.9
torch
transformers
datasets
sacrebleu
tqdm
sentencepiece

Install with:
pip install torch transformers datasets sacrebleu tqdm sentencepiece

Data & Tokenization
The project uses a parallel English–Spanish dataset (e.g., OPUS100 en–es) from Hugging Face:
data.py:
Downloads / loads the dataset via datasets.load_dataset(...).
Applies a shared subword tokenizer from Helsinki-NLP/opus-mt-en-es.
Adds BOS/EOS tokens to target sequences.
Pads / truncates sequences to max_len.
Exposes a get_dataloader(split="train" | "validation", ...) function.
You don’t need to manually download data; it is handled by the dataset + tokenizer code on first run.

Usage
1. Training

From the repo root:
python main.py --mode train \
  --epochs 5 \
  --batch_size 64 \
  --max_len 96

Typical behavior (depending on your implementation):
Automatically downloads data + tokenizer on first run.
Saves checkpoints (e.g., checkpoints/last.pt and checkpoints/best.pt).
Tracks validation loss and BLEU.
You can usually resume training with a flag like:

python main.py --mode train --resume

(or whatever main.py defines; run python main.py --help to see all options.)

2. Evaluation

Evaluate the model on the validation set using a saved checkpoint:

python main.py --mode eval \
  --checkpoint checkpoints/best.pt \
  --batch_size 64 \
  --max_len 96

Expected output: validation loss and BLEU score.

3. Translating a Sentence

Translate an English sentence to Spanish using the trained model:

python main.py --mode translate \
  --checkpoint checkpoints/best.pt \
  --text "This is a small example sentence."

Example output:

EN: This is a small example sentence.
ES: Esta es una oración de ejemplo pequeña.

Internally this calls utils.translate_sentence(...), which runs the encoder once, then uses beam search for autoregressive decoding until EOS or max_len.

4. Sanity Checks

Before or after training, you can run:

python sanity_check.py

This script typically checks:

Dataset & tokenizer wiring

Batch shapes from get_dataloader

Source / target masks (padding + no-peak)

A quick forward pass through the model

Notebook

Translation_Transformer.ipynb walks through:

Model architecture (with diagrams / comments)

Data preprocessing & tokenization

Training loop examples

Sample translations

It’s a good starting point if you want to understand or extend the implementation interactively (e.g., in Colab).

Extending the Project

Some ideas for future work:

Add bidirectional translation (ES → EN) by reusing the same architecture.

Experiment with different model sizes and max_len.

Add dropout scheduling or learning rate schedulers beyond Noam.

Implement length penalty and coverage penalty in beam search.

Log metrics to TensorBoard or Weights & Biases.
