from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from models.base_pytorch_model import BaseModel

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        """
        Basic MLP network.
        :param input_dim: Number of input features.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param output_dim: Number of output features.
        :param num_layer: Number of hidden layers.
        """
        super(MLPNetwork, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.linear_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.input_layer(input)
        input = self.relu(input)
        for linear_layer in self.linear_layers:
            input = linear_layer(input)
            input = self.relu(input)
        return self.output_layer(input)

class BasicMLP(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        """
        Basic MLP model.
        :param input_dim: Number of input features.
        :param hidden_dim: Number of neurons in the hidden layer. Example values: 32, 64, 128, 256, 512.
        :param output_dim: Number of output features, equal to 1 for regression tasks.
        :param num_layers: Number of hidden layers. Must be a positive integer, 
                           typically between 1 and 5 for small to medium-sized models.
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        super(BasicMLP, self).__init__(input_dim)

        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        return MLPNetwork(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.view(batch.shape[0], -1)  # (batch_size, num_features)
        return self.model(batch)
    
    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}
