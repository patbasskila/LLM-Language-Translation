# English–Spanish Transformer (PyTorch, From Scratch)

This repository implements a **Transformer encoder–decoder model from scratch** in PyTorch for **English → Spanish translation**.  
The goal is to recreate the architecture at a low level—attention, masking, decoding—while training on a real parallel corpus.

---

## 🚀 Features

- **Fully custom Transformer architecture**
  - Token + sinusoidal positional embeddings  
  - Multi-head self-attention  
  - Encoder–decoder cross-attention  
  - Feed-forward layers  
  - Residual connections + LayerNorm  

- **Real bilingual training data**
  - English–Spanish parallel corpus (Hugging Face OPUS100)  
  - Shared SentencePiece tokenizer from `Helsinki-NLP/opus-mt-en-es`

- **Training pipeline**
  - Teacher forcing  
  - Label smoothing  
  - Noam warmup / inverse-sqrt LR schedule  
  - Gradient clipping  
  - Validation BLEU with `sacrebleu`

- **Decoding**
  - Autoregressive **beam search**  
  - EOS-terminated generation  
  - Supports custom beam width and max length

- **Utilities**
  - CLI interface (`main.py`)
  - Sanity checks for masks, shapes, and tokenization
  - Full Jupyter notebook walkthrough

---

## 📁 Project Structure

```
├── data.py # Dataset + tokenizer + dataloader helpers
├── transformer.py # Full Transformer architecture
├── train.py # Training + validation + BLEU evaluation
├── utils.py # Beam search + translation utilities
├── main.py # CLI for train/eval/translate
├── sanity_check.py # Checks for masks, shapes, etc.
└── Translation_Transformer.ipynb # Notebook tutorial
```

---

## 📦 Installation

```bash
pip install torch datasets transformers sacrebleu tqdm sentencepiece

🗂 Data & Tokenization
Data loaded automatically via Hugging Face datasets
Tokenization via SentencePiece model: Helsinki-NLP/opus-mt-en-es
BOS/EOS added explicitly
Padding/truncation controlled by max_len
Train/validation splits handled in data.py
No manual dataset download required.

🏋️ Training
python main.py --mode train \
  --epochs 5 \
  --batch_size 64 \
  --max_len 96

Saves checkpoints (last + best)
Tracks validation BLEU
Supports resume training:
python main.py --mode train --resume

🧪 Evaluation
python main.py --mode eval \
  --checkpoint checkpoints/best.pt

Outputs BLEU and validation loss.

🌐 Translate a Sentence
python main.py --mode translate \
  --checkpoint checkpoints/best.pt \
  --text "This is a small example sentence."
```

Example:
EN: This is a small example sentence.
ES: Esta es una oración de ejemplo pequeña.

📓 Notebook

Translation_Transformer.ipynb includes:

Architecture explanation

Training examples

Visualization of attention and decoding

Sample translations

🛠 Future Improvements

Length penalty for beam search

Tokenizer training from scratch

Larger model variants (BERT-style embeddings or learned PE)

TensorBoard or W&B logging

Multi-GPU training


