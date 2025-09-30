import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------
# 1) POSITIONAL ENCODING
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Injects information about token position (0,1,2,…) into the model,
    since the Transformer has no recurrence or convolution to track order.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a (max_len x d_model) matrix of sinusoidal signals
        pe = torch.zeros(max_len, d_model)
        # pos: shape (max_len, 1), values 0,1,2,…, max_len-1
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        # Compute the denominators for the sine / cosine terms:
        # These decay exponentially from 1 to 1e-4 across the embedding dims
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        # Even indices: sin(pos * div_term), odd indices: cos(pos * div_term)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # Add batch dimension: shape now (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # Register as buffer so it's saved with the model (but not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        We simply add the first seq_len rows of pe to x.
        """
        seq_len = x.size(1)                 # number of tokens in this batch
        x = x + self.pe[:, :seq_len]        # broadcast add positional encodings
        return x


# ------------------------------------------------------------
# 2) MULTI-HEAD ATTENTION
# ------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention:
      - Project inputs to Q, K, V (queries, keys, values)
      - Split into multiple 'heads'
      - Scaled dot-product attention per head
      - Re-concatenate heads and final linear projection
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divisible by num_heads"
        self.d_k = d_model // num_heads      # dimension per head
        self.h = num_heads

        # Three linear layers to generate Q, K, V from input
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # Final projection back to d_model
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (batch, q_len, d_model)
        key: torch.Tensor,     # (batch, k_len, d_model)
        value: torch.Tensor,   # (batch, v_len, d_model)
        mask: Optional[torch.Tensor] = None  # (batch, 1, 1, k_len) or (batch, 1, q_len, k_len)
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # 1) Linear projections + reshape for multiple heads
        def project(x, linear):
            # x @ linear: (batch, seq_len, d_model)
            x = linear(x)
            # Break into heads: reshape → (batch, seq_len, h, d_k), then transpose
            return x.view(batch_size, -1, self.h, self.d_k) \
                    .transpose(1, 2)  # (batch, h, seq_len, d_k)

        Q = project(query, self.w_q)
        K = project(key,   self.w_k)
        V = project(value, self.w_v)

        # 2) Scaled dot-product attention
        # Compute raw scores: (batch, h, q_len, d_k) × (batch, h, d_k, k_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Masking: broadcast mask to (batch, 1, q_len, k_len)
        if mask is not None:
            # Guarantee mask has a broadcastable shape
            if mask.dim() == 2:
                # (batch, k_len) -> (batch, 1, 1, k_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, 1, k_len) or (batch, q_len, k_len) -> (batch, 1, q_len, k_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4 and mask.size(1) in (1,):
                # Already broadcastable
                pass
            else:
                raise ValueError(f"Unexpected mask shape {mask.shape}")

            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)    # attention weights
        weights = self.dropout(weights)        # optional dropout
        attn_output = torch.matmul(weights, V) # (batch, h, q_len, d_k)

        # 3) Concatenate heads and final linear
        attn_output = attn_output.transpose(1, 2) \
                                 .contiguous() \
                                 .view(batch_size, -1, self.h * self.d_k)
        # Back to (batch, seq_len, d_model)
        return self.w_o(attn_output)


# ------------------------------------------------------------
# 3) POSITION-WISE FEED-FORWARD
# ------------------------------------------------------------
class FeedForward(nn.Module):
    """
    Applies the same two-layer MLP (with ReLU) to each position separately.
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.net(x)


# ------------------------------------------------------------
# 4) ONE TRANSFORMER LAYER (ENCODER OR DECODER)
# ------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    A single layer in the encoder or decoder stack.
    Encoder layer: self-attention → add&norm → feed-forward → add&norm
    Decoder layer: same + encoder–decoder attention in the middle.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float,
        self_attn: bool = True,
        enc_dec_attn: bool = False
    ):
        super().__init__()
        # Optional self-attention (always on in both stacks)
        self.self_attn = MultiHeadAttention(d_model, heads, dropout) if self_attn else None
        # Only decoder layers have this second attention over encoder outputs
        self.enc_dec_attn = MultiHeadAttention(d_model, heads, dropout) if enc_dec_attn else None

        # Feed-forward
        self.ff = FeedForward(d_model, d_ff, dropout)

        # Three LayerNorms, one before each “sublayer”
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                               # current layer input
        enc_output: Optional[torch.Tensor] = None,     # only for decoder (cross-attn)
        self_mask: Optional[torch.Tensor] = None,      # <-- self-attn mask (src or tgt)
        cross_mask: Optional[torch.Tensor] = None      # <-- cross-attn mask (src)
    ) -> torch.Tensor:
        # 1) Self-attention block (use *self_mask* for both encoder and decoder)
        if self.self_attn is not None:
            attn = self.self_attn(x, x, x, mask=self_mask)  # <-- fixed: was always tgt_mask
            x = x + self.dropout(attn)   # residual connection
            x = self.norm1(x)            # layer norm

        # 2) Encoder–decoder attention (decoder only; attend over encoder outputs with *cross_mask*)
        if self.enc_dec_attn is not None and enc_output is not None:
            attn = self.enc_dec_attn(x, enc_output, enc_output, mask=cross_mask)
            x = x + self.dropout(attn)
            x = self.norm2(x)

        # 3) Feed-forward block
        ff = self.ff(x)
        x = x + self.dropout(ff)
        x = self.norm3(x)

        return x


# ------------------------------------------------------------
# 5) FULL TRANSFORMER MODEL
# ------------------------------------------------------------
class Transformer(nn.Module):
    """
    Puts it all together:
      - Input & target embeddings + positional encodings
      - N encoder layers
      - N decoder layers
      - Final linear “generator” to map to vocab logits
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        N: int = 6,
        heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        # Embedding layers + positional encodings
        self.src_embed = nn.Sequential(
            nn.Embedding(src_vocab, d_model),
            PositionalEncoding(d_model, max_len)
        )
        self.tgt_embed = nn.Sequential(
            nn.Embedding(tgt_vocab, d_model),
            PositionalEncoding(d_model, max_len)
        )

        # Create stacks of encoder / decoder layers
        self.enc_layers = nn.ModuleList([
            TransformerBlock(d_model, heads, d_ff, dropout,
                             self_attn=True, enc_dec_attn=False)
            for _ in range(N)
        ])
        self.dec_layers = nn.ModuleList([
            TransformerBlock(d_model, heads, d_ff, dropout,
                             self_attn=True, enc_dec_attn=True)
            for _ in range(N)
        ])

        # Final linear layer to produce logits for each target token
        self.generator = nn.Linear(d_model, tgt_vocab)
        # (optional) tie weights so embedding matrix = generator matrix
        self.generator.weight = self.tgt_embed[0].weight

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # 1) Embed source tokens → (batch, src_len, d_model)
        x = self.src_embed(src)
        # 2) Pass through each encoder layer in turn
        for layer in self.enc_layers:
            x = layer(x, self_mask=src_mask)  # <-- pass src_mask to encoder self-attn
        return x  # final encoder output

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        # 1) Embed target tokens
        x = self.tgt_embed(tgt)
        # 2) Pass through each decoder layer (with cross-attn to encoder)
        for layer in self.dec_layers:
            x = layer(x, enc_output, self_mask=tgt_mask, cross_mask=src_mask)  # <-- pass tgt_mask + src_mask
        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        # Full pass: encode + decode + project to vocab logits
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.generator(dec_out)


# ------------------------------------------------------------
# 6) MASKS: ignore padding & future tokens
# ------------------------------------------------------------
def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    src: (batch, src_len)
    Returns mask that is 1 for real tokens and 0 for PADs,
    shaped for attention broadcasting: (batch, 1, 1, src_len).
    """
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
    return mask

def make_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a mask that is 0 for any padding tokens or future tokens:
      - pad_mask: (batch, 1, 1, tgt_len)
      - subsequent_mask (no-peak): (1, 1, tgt_len, tgt_len) with ones on/below diagonal
    Combined → (batch, 1, tgt_len, tgt_len)
    """
    batch_size, tgt_len = tgt.size()
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
    nopeak = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))
    nopeak = nopeak.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
    return pad_mask & nopeak  # (batch, 1, tgt_len, tgt_len)


# ------------------------------------------------------------
# 7) LABEL SMOOTHING LOSS
# ------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    """
    Replaces the one-hot targets with a smoothed distribution:
      confidence on true class, rest spread over other classes.
    Helps regularize and avoid overconfidence.
    """

    def __init__(self, tgt_vocab: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.tgt_vocab = tgt_vocab

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (batch*seq, vocab) log probabilities
        target: (batch*seq) integer class labels
        Returns smoothed loss averaged over tokens.
        """
        true_dist = pred.data.clone()
        true_dist.fill_(self.smoothing / (self.tgt_vocab - 2))
        # Zero out padding positions
        mask = (target == self.padding_idx)
        target = target.masked_fill(mask, 0)
        # Place confidence mass on the true class
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        # Zero for padding rows
        true_dist.masked_fill_(mask.unsqueeze(1), 0.0)
        return self.criterion(pred, true_dist) / pred.size(0)


# ------------------------------------------------------------
# 8) TRAINING LOOP
# ------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pad_idx: int,
    device: torch.device,
    scheduler=None
) -> float:
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in data_loader:
        # Prepare inputs: decoder sees tgt[:-1], tries to predict tgt[1:]
        tgt_input = tgt_batch[:, :-1]
        targets = tgt_batch[:, 1:].contiguous().view(-1)

        # Build masks
        src_mask = make_src_mask(src_batch, pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx).to(device)

        optimizer.zero_grad()
        # Forward pass → logits: (batch, tgt_len-1, vocab)
        logits = model(
            src_batch.to(device),
            tgt_input.to(device),
            src_mask,
            tgt_mask
        )
        # Convert to log-probs and reshape for loss
        log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        loss = criterion(log_probs, targets.to(device))

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    # Return average loss per batch
    return total_loss / len(data_loader)


# ------------------------------------------------------------
# 9) EXAMPLE: SETUP + ONE EPOCH
# ------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters
    SRC_VOCAB_SIZE = 30000
    TGT_VOCAB_SIZE = 30000
    PAD_IDX = 0
    D_MODEL = 512
    N_LAYERS = 6
    HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    BATCH_SIZE = 64
    LR = 1e-4
    WARMUP_STEPS = 4000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy data loader: replace with real parallel corpus
    class DummyDataset(Dataset):
        def __len__(self): return 1000
        def __getitem__(self, idx):
            # Random integer sequences (no real language)
            return (
                torch.randint(1, SRC_VOCAB_SIZE, (20,)),  # src tokens
                torch.randint(1, TGT_VOCAB_SIZE, (20,))   # tgt tokens
            )
    data_loader = DataLoader(DummyDataset(), batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate model, optimizer, scheduler, loss
    model = Transformer(
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
        D_MODEL, N_LAYERS, HEADS, D_FF, DROPOUT
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9,0.98), eps=1e-9)
    # “Noam” schedule: warmup then inverse-sqrt decay
    def lr_lambda(step):
        step = max(step, 1)
        return (D_MODEL ** -0.5) * min(step ** -0.5, step * WARMUP_STEPS ** -1.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = LabelSmoothingLoss(TGT_VOCAB_SIZE, PAD_IDX, smoothing=0.1)

    # Run one epoch (for demonstration)
    avg_loss = train_epoch(
        model, data_loader,
        optimizer, criterion,
        PAD_IDX, DEVICE, scheduler
    )
    print(f"Average training loss: {avg_loss:.4f}")
