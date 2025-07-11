# phase1/model.py

import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class Phase1Model(nn.Module):
    def __init__(self, input_dim=4, lstm_hidden=128, fcn_hidden=512, num_ops=10, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, lstm_hidden)
        
        self.lstm = nn.LSTM(input_size=lstm_hidden,
                            hidden_size=lstm_hidden,
                            batch_first=True,
                            bidirectional=True)

        self.attn = MultiHeadSelfAttention(dim=2*lstm_hidden, heads=nhead)

        self.fcn = nn.Sequential(
            nn.Linear(2*lstm_hidden, fcn_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fcn_hidden, num_ops),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)              # (B, T, lstm_hidden)
        x, _ = self.lstm(x)                # (B, T, 2*lstm_hidden)
        x = self.attn(x)                   # (B, T, 2*lstm_hidden)
        out = self.fcn(x)                  # (B, T, num_ops)
        return out
