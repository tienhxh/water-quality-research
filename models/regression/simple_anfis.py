from typing import Any, Dict, List

import torch
from torch import nn
from models.base_pytorch_model import BaseModel  


class GaussianLayer(nn.Module):
    """Fuzzification layer using Gaussian membership function."""

    def __init__(self, input_dim, num_rules):
        """
        Initializes the GaussianLayer.

        Args:
            input_dim: The number of input features.
            num_rules: The number of rules (and membership functions) per input.
        """
        super(GaussianLayer, self).__init__()
        # Initialize mean and sigma randomly
        self.mean = nn.Parameter(torch.randn(input_dim, num_rules))
        self.sigma = nn.Parameter(torch.abs(torch.randn(input_dim, num_rules)))  # Sigma must be positive
        self.epsilon = 1e-6  # Small value to prevent division by zero

    def forward(self, x):
        """
        Calculates the membership values using the Gaussian function.

        Args:
            x: The input tensor of shape (batch_size, input_dim).

        Returns:
            A tensor of membership values of shape (batch_size, input_dim, num_rules).
        """
        x = x.unsqueeze(2)  # (batch_size, input_dim, 1)
        mean = self.mean.unsqueeze(0)  # (1, input_dim, num_rules)
        sigma = self.sigma.unsqueeze(0)  # (1, input_dim, num_rules)
        sigma = torch.clamp(sigma, min=self.epsilon)  # Ensure sigma is not zero
        return torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2))  # (batch_size, input_dim, num_rules)


class ANFISNetwork(nn.Module):
    def __init__(self, input_dim: int, num_rules: int, output_dim: int):
        """
        Initializes the ANFISNetwork.

        Args:
            input_dim: The number of input features.
            num_rules: The number of rules per input.
            output_dim: The number of output features.
        """
        super(ANFISNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.output_dim = output_dim

        # Layer 1: Fuzzification with Gaussian membership functions
        self.fuzzification = GaussianLayer(input_dim, num_rules)

        # Layer 4: Consequent parameters (linear coefficients for each rule)
        self.consequent_layer = nn.Linear(num_rules, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ANFIS network.

        Args:
            x: The input tensor of shape (batch_size, input_dim).

        Returns:
            The output tensor of shape (batch_size, output_dim).
        """
        # Layer 1: Fuzzification
        membership = self.fuzzification(x)  # (batch_size, input_dim, num_rules)

        # Layer 2: Rule strength (product t-norm)
        rule_strength = torch.prod(membership, dim=1)  # (batch_size, num_rules)

        # Layer 3: Normalization
        normalized_strength = rule_strength / (
            torch.sum(rule_strength, dim=1, keepdim=True) + 1e-12
        )  # Avoid division by zero

        # Layer 4: Consequent layer (linear combination)
        rule_output = self.consequent_layer(normalized_strength)  # (batch_size, output_dim)

        return rule_output


class BasicANFIS(BaseModel):
    def __init__(self, input_dim: int, num_rules: int, output_dim: int):
        """
        Initializes the BasicANFIS model.

        Args:
            input_dim: The number of input features.
            num_rules: The number of rules per input.
            output_dim: The number of output features.
        """
        self.num_rules = num_rules
        self.output_dim = output_dim
        super(BasicANFIS, self).__init__(input_dim)

        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        """Builds the ANFIS network."""
        return ANFISNetwork(self.input_dim, self.num_rules, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicANFIS model.

        Args:
            x:  The input tensor.

        Returns:
            The output tensor.
        """
        return self.model(x)  # Don't use squeeze here

    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}