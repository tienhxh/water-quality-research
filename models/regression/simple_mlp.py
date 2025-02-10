from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from models.base_pytorch_model import BaseModel

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Basic MLP network.
        :param input_dim: Number of input features.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param output_dim: Number of output features.
        """
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)

class BasicMLP(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Basic MLP model.
        :param input_dim: Number of input features.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param output_dim: Number of output features.
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        super(BasicMLP, self).__init__(input_dim)

        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        return MLPNetwork(self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.view(batch.shape[0], -1)  # Flatten input
        return self.model(batch)
    
    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}
