# phase2/model.py

import torch
import torch.nn as nn

LAYER_TOKENS = [
    "conv", "relu", "batchnorm", "tanh", "sigmoid", "fc", 
    "softmax", "residual", "mobilenet", "pool", "<PAD>", "<EOS>"
]

TOKEN_TO_IDX = {tok: idx for idx, tok in enumerate(LAYER_TOKENS)}
IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}
VOCAB_SIZE = len(LAYER_TOKENS)

class Phase2Model(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=num_layers, batch_first=True, bidirectional=True)

        self.decoder = nn.GRU(input_size=hidden_dim*2, hidden_size=hidden_dim,
                              num_layers=1, batch_first=True)

        self.output_layer = nn.Linear(hidden_dim, VOCAB_SIZE)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, max_len=50):
        """
        x: [B, T, input_dim] — OPi sequence
        Returns: [B, max_len, VOCAB_SIZE]
        """
        enc_out, _ = self.encoder(x)  # [B, T, 2*H]
        # Use final timestep as context
        context = enc_out[:, -1:, :]  # [B, 1, 2*H]

        decoder_input = context.repeat(1, max_len, 1)  # [B, max_len, 2*H]
        dec_out, _ = self.decoder(decoder_input)       # [B, max_len, H]
        logits = self.output_layer(dec_out)            # [B, max_len, V]
        return self.softmax(logits)
