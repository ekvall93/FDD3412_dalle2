""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

import torch.nn as nn
import torch
class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) network

    This MLP implementation allows for customizable depth, expansion factor for hidden layers,
    and an optional normalization layer after each hidden layer.

    Attributes:
        dim_in (int): The dimension of the input layer.
        dim_out (int): The dimension of the output layer.
        expansion_factor (float): Factor to determine the size of the hidden layers.
        depth (int): The number of hidden layers in the MLP.
        norm (bool): Flag to include normalization layers after each hidden layer.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        expansion_factor: float = 2.,
        depth: int = 2,
        norm: bool = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        # Initialize layers of the MLP
        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        # Add additional hidden layers based on depth
        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        # Output layer
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the MLP.
        """
        return self.net(x.float())