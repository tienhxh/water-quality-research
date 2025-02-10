from typing import Any, Dict, List
import torch
from torch import nn
from models.base_pytorch_model import BaseModel

class TransformerNetwork(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, output_dim: int, num_heads: int = 2, num_layers: int = 2):
        """
        Transformer-based regression model.
        :param input_dim: Number of input features.
        :param model_dim: Hidden dimension of the Transformer.
        :param output_dim: Number of output features.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of encoder layers.
        """
        super(TransformerNetwork, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # Transform input into a vector of size model_dim
        x = x.unsqueeze(1)  # Add sequence length dimension = 1 (batch_size, seq_len=1, model_dim)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # Use the last vector as output

class BasicTransformer(BaseModel):
    def __init__(self, input_dim: int, model_dim: int, output_dim: int, num_heads: int = 2, num_layers: int = 2):
        """
        Basic Transformer model.
        """
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        super(BasicTransformer, self).__init__(input_dim)
        
        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        return TransformerNetwork(self.input_dim, self.model_dim, self.output_dim, self.num_heads, self.num_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.view(batch.shape[0], -1)  # Flatten input
        return self.model(batch)
    
    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor], **kwargs) -> Dict[str, Any]:
        return {}
