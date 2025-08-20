# phase1/model.py

import torch
import torch.nn as nn

# 🎯 Multi-Head Self-Attention Block
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D]
        Returns:
            torch.Tensor: Output tensor of shape [B, T, D] with attention and residual normalization
        """
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

# 🧠 Phase1 Model for OPi Embedding
class Phase1Model(nn.Module):
    def __init__(self, input_dim=4, lstm_hidden=128, fcn_hidden=512, num_ops=10, nhead=4):
        """
        Args:
            input_dim (int): Dimension of each trace event [op_idx, start, end, duration]
            lstm_hidden (int): Hidden size for bidirectional LSTM
            fcn_hidden (int): Hidden size for FCN head
            num_ops (int): Number of unique operations
            nhead (int): Attention heads
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, lstm_hidden)  # 🔡 Project to LSTM space

        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.attn = MultiHeadSelfAttention(dim=2 * lstm_hidden, heads=nhead)

        self.fcn = nn.Sequential(
            nn.Linear(2 * lstm_hidden, fcn_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fcn_hidden, num_ops),
            nn.Sigmoid()  # ➡️ OPi per-timestep multi-label probabilities
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, T, 4] - kernel trace
        Returns:
            torch.Tensor: Output tensor [B, T, num_ops] - OPi embeddings
        """
        x = self.embedding(x)      # [B, T, H]
        x, _ = self.lstm(x)        # [B, T, 2H]
        x = self.attn(x)           # [B, T, 2H]
        out = self.fcn(x)          # [B, T, num_ops]
        return out
