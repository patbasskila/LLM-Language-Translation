import torch
from data import tokenizer
from transformer import make_src_mask, make_tgt_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = tokenizer.pad_token_id
BOS_IDX = tokenizer.bos_token_id or tokenizer.cls_token_id or PAD_IDX  # fallback if BOS undefined
EOS_IDX = tokenizer.eos_token_id or tokenizer.sep_token_id or 0        # fallback if EOS undefined


def translate_sentence(sentence, model, max_len=64, beam_size=10):
    """
    Translate an input English/Spanish sentence using the trained model.
    
    Args:
        sentence (str): Input text to translate.
        model (nn.Module): Trained Transformer model.
        max_len (int): Max output length.
        beam_size (int): Beam width for beam search decoding.
    
    Returns:
        str: Best translated sentence as plain text.
    """
    model.eval()
    with torch.no_grad():
        # 1. Tokenize input (assume sentence is English → want Spanish)
        src_tokens = tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=max_len
        ).input_ids.to(DEVICE)

        # 2. Encode source
        src_mask = make_src_mask(src_tokens, PAD_IDX).to(DEVICE)
        enc_out = model.encode(src_tokens, src_mask)

        # 3. Initialize beam with <BOS>
        beam = [(torch.tensor([[BOS_IDX]], device=DEVICE), 0.0)]  # (sequence, log_prob)

        for _ in range(max_len):
            new_beam = []
            for seq, score in beam:
                if seq[0, -1].item() == EOS_IDX:
                    # Already ended, keep as is
                    new_beam.append((seq, score))
                    continue

                tgt_mask = make_tgt_mask(seq, PAD_IDX).to(DEVICE)
                out = model.decode(seq, enc_out, src_mask, tgt_mask)
                logits = model.generator(out[:, -1, :])  # last token
                log_probs = torch.log_softmax(logits, dim=-1)

                # Select top-k next tokens
                topk_log_probs, topk_ids = log_probs.topk(beam_size)

                for k in range(beam_size):
                    next_token = topk_ids[:, k].unsqueeze(0)  # shape (1,1)
                    next_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[0, k].item()
                    new_beam.append((next_seq, new_score))

            # Keep only the best `beam_size` candidates
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
            beam = new_beam

        # Pick the sequence with the highest score
        best_seq = beam[0][0]

        # 4. Decode IDs back to text
        return tokenizer.decode(best_seq.squeeze().tolist(), skip_special_tokens=True)
