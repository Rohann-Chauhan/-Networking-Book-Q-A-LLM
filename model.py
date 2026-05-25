import torch.nn as nn

from embeddings import (
    InputEmbedding,
    PositionalEncoding
)

from normalization import LayerNormalization

from decoder import DecoderBlock

from config import *

class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        self.embedding = InputEmbedding(
            d_model,
            vocab_size
        )

        self.pos_encoding = PositionalEncoding(
            d_model,
            max_seq_len,
            dropout
        )

        self.layers = nn.ModuleList([

            DecoderBlock(
                d_model,
                num_heads,
                d_ff,
                dropout
            )

            for _ in range(num_layers)

        ])

        self.norm = LayerNormalization(
            d_model
        )

        self.projection = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        x,
        mask
    ):

        x = self.embedding(x)

        x = self.pos_encoding(x)

        for layer in self.layers:

            x = layer(x, mask)

        x = self.norm(x)

        return self.projection(x)