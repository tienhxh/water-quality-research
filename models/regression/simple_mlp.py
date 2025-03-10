from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from models.base_pytorch_model import BaseModel  


class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, seed=42):
        super(MLPNetwork, self).__init__()

        # Fix the seed when initializing the model
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

        # Initialize weights deterministically
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class BasicMLP(BaseModel):
    def __init__(self, input_dim: int, seed=42):
        self.seed = seed  
        super(BasicMLP, self).__init__(input_dim)  
        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        return MLPNetwork(self.input_dim, self.seed)  

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.view(batch.shape[0], -1)  # (batch_size, num_features)
        return self.model(batch)

    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}
