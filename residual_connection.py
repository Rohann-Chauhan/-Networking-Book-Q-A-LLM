import torch.nn as nn

from normalization import LayerNormalization

class ResidualConnection(nn.Module):

    def __init__(
        self,
        d_model,
        dropout
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm = LayerNormalization(
            d_model
        )

    def forward(
        self,
        x,
        sublayer
    ):

        return x + self.dropout(
            sublayer(
                self.norm(x)
            )
        )