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

.
├── data.py # Dataset + tokenizer + dataloader helpers
├── transformer.py # Full Transformer architecture
├── train.py # Training + validation + BLEU evaluation
├── utils.py # Beam search + translation utilities
├── main.py # CLI for train/eval/translate
├── sanity_check.py # Checks for masks, shapes, etc.
└── Translation_Transformer.ipynb # Notebook tutorial


