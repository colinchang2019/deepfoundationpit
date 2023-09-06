# -*- encoding: utf-8 -*-
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

from config import config



class Informer4(nn.Module):
    def __init__(self, input_dim=12, output_dim=12, sequence_len=config.sequences_in, d_model=512, n_heads=8, e_layers=8, d_layers=8, d_ff=1024):
        # input_dim=12, output_dim=12, sequence_len=6, d_model=256, n_heads=8, e_layers=4, d_layers=4, d_ff=512： 0.003312
        # input_dim=12, output_dim=12, sequence_len=6, d_model=512, n_heads=8, e_layers=8, d_layers=8, d_ff=1024: 0.00268
        super(Informer4, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, output_dim)
        )

        self.enc_self_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads) for _ in range(e_layers)
        ])

        self.dec_self_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads) for _ in range(d_layers)
        ])

        self.dec_enc_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads) for _ in range(d_layers)
        ])

        self.enc_feedforwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, d_model),
                nn.Dropout(0.1)
            ) for _ in range(e_layers)
        ])

        self.dec_feedforwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, d_model),
                nn.Dropout(0.1)
            ) for _ in range(d_layers)
        ])

        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_len, d_model))
        nn.init.normal_(self.positional_encoding, std=0.02)

        self.output_projection = nn.Sequential(
            nn.Linear(sequence_len, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1)
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        # Encode the input sequence
        x = self.encoder(x)

        # Add the positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Apply the encoder self-attention mechanism to the input sequence
        for i in range(len(self.enc_self_attns)):
            residual = x
            x, _ = self.enc_self_attns[i](x, x, x)
            x = x + residual
            x = x + self.enc_feedforwards[i](x)

        # Decode the output sequence
        for i in range(len(self.dec_self_attns)):
            # Apply the decoder self-attention mechanism to the output sequence
            residual = x
            output, _ = self.dec_self_attns[i](x, x, x)
            output = output + residual
            output = output + self.dec_feedforwards[i](output)

            # Apply the encoder-decoder attention mechanism to the output sequence
            residual = x
            output, _ = self.dec_enc_attns[i](output, x, x)
            x = residual + output

        # Decode the final output sequence
        x = self.decoder(x)

        # Project the output sequence to the desired output sequence length
        x = self.output_projection(x.transpose(1, 2)).transpose(1, 2)

        return x
