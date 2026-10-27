from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field, fields, asdict
import numpy as np
import torch
import torch.nn as nn


@dataclass
class SentimentModelConfig:
    num_features: int
    vocab_size: int = 10_000
    seq_max_len: int = 100
    positional_encoding_type: Literal["sinusoidal", "learned", "none"] = "sinusoidal"
    embedding_dim: int = 32
    padding_idx: int = 0
    num_time_features: int = 2
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
    
class TimeEncoder(nn.Module):
    """
    Time2Vec (Kazemi et al., 2019).

    Maps scalar time features to a learned vector with one linear component
    and ``output_dim - 1`` periodic (sinusoidal) components:

        t2v(τ)[0]   = ω₀·τ + φ₀           (linear / trend)
        t2v(τ)[1:]  = sin(ω·τ + φ)         (periodic / seasonality)

    Args:
        input_dim:  Number of raw time features (e.g. 1 for a single timestamp).
        output_dim: Total dimensionality of the output representation.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim

        # Linear (trend) component  →  (input_dim, 1)
        self.linear_weight = nn.Parameter(torch.randn(input_dim, 1) * 0.02)
        self.linear_bias = nn.Parameter(torch.zeros(1))

        # Periodic (seasonal) components  →  (input_dim, output_dim - 1)
        n_periodic = output_dim - 1
        if n_periodic > 0:
            self.periodic_weight = nn.Parameter(
                torch.randn(input_dim, n_periodic) * 0.02
            )
            self.periodic_bias = nn.Parameter(torch.zeros(n_periodic))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(*, input_dim)`` raw time features.

        Returns:
            ``(*, output_dim)`` Time2Vec representation.
        """
        # Linear trend term  →  (*, 1)
        linear = x @ self.linear_weight + self.linear_bias

        if self.output_dim == 1:
            return linear

        # Periodic terms  →  (*, output_dim - 1)
        periodic = torch.sin(x @ self.periodic_weight + self.periodic_bias)

        return torch.cat([linear, periodic], dim=-1)

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

    def __init__(self, config: SentimentModelConfig):
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

        self.time_encoder = TimeEncoder(
            input_dim=config.num_time_features,
            output_dim=config.embedding_dim,
        )
    
        
        self.mlp = MLP(
            input_dim=config.embedding_dim,
            hidden_dims=config.mlp_hidden_layers,
            output_dim=1,
        )
    
    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
        time_emb = self.time_encoder(x2) # (B, D)

        prediction = self.mlp(pooled + time_emb).squeeze(-1) # (B,)
        return prediction


# ─── New multi-modal model for DataPipeline output ──────────────────────────────

@dataclass
class DealPricingModelConfig:
    # Text branch
    vocab_size: int = 10_000
    seq_max_len: int = 64
    positional_encoding_type: Literal["sinusoidal", "learned", "none"] = "sinusoidal"
    text_embedding_dim: int = 32
    padding_idx: int = 0

    # Tabular branch (numeric + numeric_log + temporal + binary + categorical_low)
    num_tabular_features: int = 30

    # Categorical high branch (one embedding per column)
    categorical_high_vocab_sizes: List[int] = field(default_factory=list)
    categorical_embedding_dim: int = 8

    # Fusion
    fusion_dim: int = 64
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        items = [(f.name, getattr(self, f.name)) for f in fields(self)]
        max_key = max(len(k) for k, _ in items)
        lines = [f"{k.ljust(max_key)} : {v}" for k, v in items]
        return f"{cls_name}(\n  " + "\n  ".join(lines) + "\n)"


class DealPricingModel(nn.Module):
    """
    Multi-modal regression model for price prediction.

    Inputs (from DataPipeline):
        text_ids:       (B, S)   padded token ids
        tabular:        (B, T)   numeric + numeric_log + temporal + binary + categorical_low
        cat_high:       (B, C)   integer-encoded high-cardinality categoricals

    Output:
        prediction:     (B,)     predicted scaled price
    """

    def __init__(self, config: DealPricingModelConfig):
        super().__init__()
        self.config = config

        # ── Text branch ──
        self.text_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.text_embedding_dim,
            padding_idx=config.padding_idx,
        )
        self.positional_encoding = PositionalEncoder(
            embedding_dim=config.text_embedding_dim,
            max_len=config.seq_max_len,
            encoding_type=config.positional_encoding_type,
        )
        self.attention_pool = AttentionPooling(config.text_embedding_dim)
        self.text_proj = nn.Linear(config.text_embedding_dim, config.fusion_dim)

        # ── Tabular branch ──
        self.tabular_proj = nn.Sequential(
            nn.Linear(config.num_tabular_features, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
        )

        # ── Categorical high branch ──
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vs, config.categorical_embedding_dim)
            for vs in config.categorical_high_vocab_sizes
        ])
        cat_total_dim = len(config.categorical_high_vocab_sizes) * config.categorical_embedding_dim
        self.cat_proj = nn.Linear(cat_total_dim, config.fusion_dim) if cat_total_dim > 0 else None

        # ── Fusion MLP ──
        num_branches = 2 + (1 if cat_total_dim > 0 else 0)
        self.mlp = MLP(
            input_dim=config.fusion_dim * num_branches,
            hidden_dims=config.mlp_hidden_layers,
            output_dim=1,
            dropout=config.dropout,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        tabular: torch.Tensor,
        cat_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_ids: (B, S) padded token ids
            tabular:  (B, T) all dense features
            cat_high: (B, C) high-cardinality categorical indices

        Returns:
            (B,) predictions
        """
        B, S = text_ids.shape

        # Text branch
        emb = self.text_embedding(text_ids)  # (B, S, D)
        pe = self.positional_encoding(S)     # (S, D)
        emb = emb + pe.unsqueeze(0)
        mask = (text_ids != self.config.padding_idx)
        text_vec = self.attention_pool(emb, mask)  # (B, D)
        text_vec = self.text_proj(text_vec)        # (B, fusion_dim)

        # Tabular branch
        tab_vec = self.tabular_proj(tabular)  # (B, fusion_dim)

        # Categorical high branch
        branches = [text_vec, tab_vec]
        if self.cat_proj is not None and cat_high is not None:
            cat_embs = [
                emb_layer(cat_high[:, i])
                for i, emb_layer in enumerate(self.cat_embeddings)
            ]
            cat_concat = torch.cat(cat_embs, dim=-1)  # (B, C*cat_emb_dim)
            cat_vec = self.cat_proj(cat_concat)        # (B, fusion_dim)
            branches.append(cat_vec)

        # Fusion
        fused = torch.cat(branches, dim=-1)  # (B, fusion_dim * num_branches)
        prediction = self.mlp(fused).squeeze(-1)  # (B,)
        return prediction