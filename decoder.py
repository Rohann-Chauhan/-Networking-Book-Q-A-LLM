import torch.nn as nn

from multiheadattention import MultiHeadAttentionBlock
from feedforward import FeedForward
from residual_connection import ResidualConnection

class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        h,
        d_ff,
        dropout
    ):

        super().__init__()

        self.attention = MultiHeadAttentionBlock(
            d_model,
            h,
            dropout
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.residuals = nn.ModuleList([

            ResidualConnection(
                d_model,
                dropout
            )

            for _ in range(2)

        ])

    def forward(
        self,
        x,
        mask
    ):

        x = self.residuals[0](

            x,

            lambda x:
            self.attention(
                x,
                x,
                x,
                mask
            )

        )

        x = self.residuals[1](

            x,
            self.feed_forward

        )

        return x