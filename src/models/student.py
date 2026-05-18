"""
Student model: Compact transformer for intent classification.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class StudentModel(nn.Module):
    """
    Compact transformer student model for intent classification.

    Architecture:
        - Embedding layer with positional encoding
        - 2-4 transformer encoder layers
        - Hidden size 256-384
        - Classification head on [CLS]-equivalent (first token)
        - ~5-15M parameters
    """

    def __init__(
        self,
        vocab_size: int = 30522,  # BERT tokenizer vocab size
        num_labels: int = 7,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        intermediate_size: int = 512,
        max_length: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_length, dropout)
        self.embedding_norm = nn.LayerNorm(hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Returns:
            dict with keys: logits, loss (if labels provided), hidden_state
        """
        # Create padding mask for transformer (True = ignore)
        src_key_padding_mask = ~attention_mask.bool()

        # Embed tokens
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.embedding_norm(x)

        # Transformer encoding
        hidden_state = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )

        # Use first token (CLS equivalent) for classification
        cls_output = hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        result = {
            "logits": logits,
            "hidden_state": hidden_state,
            "cls_embedding": cls_output,
        }

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Return model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


def create_student_small(num_labels: int = 7) -> StudentModel:
    """Create a small student model (~5M params)."""
    return StudentModel(
        num_labels=num_labels,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
    )


def create_student_medium(num_labels: int = 7) -> StudentModel:
    """Create a medium student model (~10M params)."""
    return StudentModel(
        num_labels=num_labels,
        hidden_size=384,
        num_layers=3,
        num_heads=6,
        intermediate_size=768,
    )


def create_student_large(num_labels: int = 7) -> StudentModel:
    """Create a larger student model (~15M params)."""
    return StudentModel(
        num_labels=num_labels,
        hidden_size=384,
        num_layers=4,
        num_heads=6,
        intermediate_size=1024,
    )
