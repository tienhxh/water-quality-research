from typing import Any, Dict, List
import torch
from torch import nn
from models.base_pytorch_model import BaseModel

class TransformerNetwork(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, output_dim: int, num_heads: int = 2, num_layers: int = 2, seq_length: int = 10):
        """
        Transformer-based regression model.
        :param input_dim: Number of input features.
        :param model_dim: Hidden dimension of the Transformer.
        :param output_dim: Number of output features.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of encoder layers.
        :param seq_length: Length of the input sequences.
        """
        super(TransformerNetwork, self).__init__()
        self.seq_length = seq_length
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # Transform input into model_dim
        x = x.view(x.shape[0], self.seq_length, -1)  # Ensure correct shape (batch_size, seq_length, model_dim)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # Use the last time step for prediction

class BasicTransformer(BaseModel):
    def __init__(self, input_dim: int, model_dim: int, output_dim: int, num_heads: int = 2, num_layers: int = 2, seq_length: int = 10):
        """
        Basic Transformer model with variable sequence length.
        """
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_length = seq_length
        super(BasicTransformer, self).__init__(input_dim)

        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        return TransformerNetwork(self.input_dim, self.model_dim, self.output_dim, self.num_heads, self.num_layers, self.seq_length)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.view(batch.shape[0], self.seq_length, -1)  # Ensure shape matches Transformer expectations
        return self.model(batch)

    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor], **kwargs) -> Dict[str, Any]:
        return {}
