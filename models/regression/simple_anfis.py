from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from models.base_pytorch_model import BaseModel
from torch.utils.data.dataloader import DataLoader


class ANFISNetwork(nn.Module):
    def __init__(self, input_dim: int, num_rules: int, output_dim: int):
        super(ANFISNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.output_dim = output_dim

        self.membership_layer = nn.Linear(input_dim, num_rules)
        self.rule_layer = nn.Linear(num_rules, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membership = torch.sigmoid(self.membership_layer(x))
        rule_output = self.rule_layer(membership)
        return rule_output

class BasicANFIS(BaseModel):
    def __init__(self, input_dim: int, num_rules: int, output_dim: int):
        self.num_rules = num_rules
        self.output_dim = output_dim
        super(BasicANFIS, self).__init__(input_dim)
        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        return ANFISNetwork(self.input_dim, self.num_rules, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()

    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}

