import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(
        self,
        d_model,
        eps=1e-6
    ):

        super().__init__()

        self.gamma = nn.Parameter(
            torch.ones(d_model)
        )

        self.beta = nn.Parameter(
            torch.zeros(d_model)
        )

        self.eps = eps

    def forward(self, x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        std = x.std(
            dim=-1,
            keepdim=True
        )

        return (

            self.gamma
            * (x - mean)
            / (std + self.eps)

            + self.beta

        )