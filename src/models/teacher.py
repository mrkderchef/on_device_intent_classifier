"""
Teacher model: BERT-base for intent classification.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class TeacherModel(nn.Module):
    """
    BERT-base teacher model for intent classification.

    Architecture:
        - BERT-base-uncased (~110M parameters)
        - 12 transformer layers
        - Hidden size 768
        - Classification head on [CLS] token
    """

    def __init__(
        self,
        num_labels: int = 7,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Returns:
            dict with keys: logits, loss (if labels provided), hidden_states
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        result = {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
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
