# phase2/model.py

import torch
import torch.nn as nn

# 🔡 Vocabulary Tokens
LAYER_TOKENS = [
    "conv", "relu", "batchnorm", "tanh", "sigmoid", "fc",
    "softmax", "residual", "mobilenet", "pool", "<PAD>", "<EOS>"
]

TOKEN_TO_IDX = {tok: idx for idx, tok in enumerate(LAYER_TOKENS)}
IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}
VOCAB_SIZE = len(LAYER_TOKENS)
PAD_IDX = TOKEN_TO_IDX["<PAD>"]
EOS_IDX = TOKEN_TO_IDX["<EOS>"]

# 🔍 Attention Module
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))  # [H]

    def forward(self, encoder_outputs):
        # encoder_outputs: [B, T, 2H]
        energy = torch.tanh(self.attn(encoder_outputs))  # [B, T, H]
        v = self.v.unsqueeze(0).unsqueeze(2)             # [1, H, 1]
        v = v.expand(encoder_outputs.size(0), -1, -1)    # [B, H, 1]
        attn_weights = torch.bmm(energy, v)              # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)# [B, T, 1]
        context = torch.sum(encoder_outputs * attn_weights, dim=1, keepdim=True)  # [B, 1, 2H]
        return context

# 🧐 Phase2 Model
class Phase2Model(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.attention = Attention(hidden_dim)

        self.decoder = nn.GRU(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, VOCAB_SIZE)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, max_len=50):
        """
        x: [B, T, input_dim] — OPi sequence
        Returns: [B, max_len, VOCAB_SIZE]
        """
        enc_out, _ = self.encoder(x)                    # [B, T, 2H]
        context = self.attention(enc_out)               # [B, 1, 2H]
        decoder_input = context.repeat(1, max_len, 1)   # [B, max_len, 2H]
        dec_out, _ = self.decoder(decoder_input)        # [B, max_len, H]
        logits = self.output_layer(dec_out)             # [B, max_len, V]
        return self.softmax(logits)
