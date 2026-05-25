import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout
    ):

        super().__init__()

        self.linear_1 = nn.Linear(
            d_model,
            d_ff
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(
            d_ff,
            d_model
        )

    def forward(self, x):

        return self.linear_2(

            self.dropout(

                F.gelu(
                    self.linear_1(x)
                )

            )

        )