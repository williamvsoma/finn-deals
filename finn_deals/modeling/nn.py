from typing import List, Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass, field, fields, asdict
import numpy as np
import torch
import torch.nn as nn


@dataclass
class SentimentModelConfig:
    vocab_size: int = 10_000
    seq_max_len: int = 100
    positional_encoding_type: Literal["sinusoidal", "learned", "none"] = "sinusoidal"
    embedding_dim: int = 32
    padding_idx: int = 0
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [64])  

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of the config.
        """
        return asdict(self)
    
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        items = [(f.name, getattr(self, f.name)) for f in fields(self)]
        max_key = max(len(k) for k, _ in items)

        lines = [
            f"{k.ljust(max_key)} : {v}"
            for k, v in items
        ]
        return f"{cls_name}(\n  " + "\n  ".join(lines) + "\n)"

class BoundedParameter(nn.Module):

    def __init__(
        self, 
        init_value: float = 0.05, 
        min_val: float = 0.001, 
        max_val: float = 1.0,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

        # Convert initial value to uncontrained space
        # Using sigmoid: param = min + (max - min)*sigmoid(raw)
        init_raw = torch.logit(torch.tensor(
            (init_value - min_val) / (max_val - min_val)
        ))

        self.raw_param = nn.Parameter(init_raw)

    def forward(self):
        # Map unconstrained parameter to [min_val, max_val]
        normalized = torch.sigmoid(self.raw_param)
        return self.min_val + (self.max_val - self.min_val)*normalized

class MLP(nn.Module):
    """Basic Multilayer Perceptron Module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i +1]))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PositionalEncoder(nn.Module):
    
    def __init__(
        self,
        embedding_dim: int,
        max_len: int = 100,
        encoding_type: Literal["sinusoidal", "learned", "none"] = "sinusoidal",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.encoding_type = encoding_type

        if encoding_type == "sinusoidal":
            # Create winusoidal positional encoding
            pe = torch.zeros(max_len, embedding_dim)
            position = torch.arange(0, max_len , dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embedding_dim, 2).float() *
                (-np.log(10_000.0) / embedding_dim)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # Register as buffer (not a parameter, but part of state)
            self.register_buffer("pe", pe)
        elif encoding_type == "learned":
            # Learned positional embedding
            self.pe = nn.Parameter(torch.randn(max_len, embedding_dim) * 0.02)
        elif encoding_type == "none":
            # No positional encoding - register a dummy bufferfor consistency
            self.register_buffer("pe", torch.zeros(max_len, embedding_dim))
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}. Must be 'sinusoidal', 'learnedæ, or 'none'.") 

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Length of the sequence

        Returns:
            postitional encoding: (seq_len, embedding_dim)
        """
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )

        return self.pe[:seq_len, :]

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over sequence dimension.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.score = nn.Linear(embedding_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) embeddings
            mask: (B, S) 1 for valid tokens, 0 for padding

        Returns:
            pooled: (B, D)
        """
        # Compute raw attention scores
        scores = self.score(x).squeeze(-1)  # (B, S)

        # Mask padding tokens
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Normalize
        weights = torch.softmax(scores, dim=1)  # (B, S)

        # Weighted sum
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled


class SentimentModel(nn.Module):

    def __init__(self, config: SentimentModelConfig = SentimentModelConfig()):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )
        
        self.positional_encoding = PositionalEncoder(
            embedding_dim=config.embedding_dim,
            max_len=config.seq_max_len,
            encoding_type=config.positional_encoding_type
        )

        self.attention_pool = AttentionPooling(
            embedding_dim=config.embedding_dim,
        )
        
        self.mlp = MLP(
            input_dim=config.embedding_dim,
            hidden_dims=config.mlp_hidden_layers,
            output_dim=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S)
        Returns
            prediction: (B,)
        """
        B, S = x.shape

        emb = self.embedding(x) # (B, S, D)
        pe = self.positional_encoding(S) # (S, D)
        emb = emb + pe.unsqueeze(0)

        mask = (x != self.config.padding_idx) # (B, S)

        # mean pooling (masked)
        # emb = emb * mask
        # lengths = mask.sum(dim=1).clamp(min=1)
        # pooled = emb.sum(dim=1) / lengths # (B, D)

        # Attetion pooling
        pooled = self.attention_pool(emb, mask) # (B, D)

        prediction = self.mlp(pooled).squeeze(-1) # (B,)
        return prediction