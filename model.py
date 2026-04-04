"""
Toxic Comment Classifier - TextCNN Model
Kim (2014) Convolutional Neural Network for sentence classification.
Supports multi-label toxic comment detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TextCNN(nn.Module):
    """
    TextCNN for multi-label toxic comment classification.

    Architecture:
        Embedding → [Conv1d(kernel_size=k) → ReLU → MaxPool] × K filters
        → Concat → Dropout → FC → Sigmoid

    References:
        Kim, Y. (2014). Convolutional neural networks for sentence classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: List[int] = [2, 3, 4, 5],
        num_classes: int = 6,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings=None,
        freeze_embeddings: bool = False,
    ):
        """
        Args:
            vocab_size:           size of vocabulary
            embed_dim:            embedding dimension
            num_filters:          number of filters per kernel size
            filter_sizes:         list of kernel widths (n-gram windows)
            num_classes:          number of output labels (6 for Jigsaw)
            dropout:              dropout probability before FC layer
            pad_idx:              padding token index (for embedding)
            pretrained_embeddings: optional pre-trained embedding matrix (vocab_size, embed_dim)
            freeze_embeddings:    whether to freeze embedding weights
        """
        super(TextCNN, self).__init__()

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Parallel convolutional filters
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=fs,
                padding=0,
            )
            for fs in filter_sizes
        ])

        # Dropout + fully connected output
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) — token ids
        Returns:
            logits: (batch, num_classes) — raw scores before sigmoid
        """
        # (batch, seq_len) → (batch, seq_len, embed_dim) → (batch, embed_dim, seq_len)
        embedded = self.embedding(x).permute(0, 2, 1)

        # Each conv: (batch, num_filters, seq_len - filter_size + 1) → MaxPool → (batch, num_filters)
        pooled = []
        for conv in self.convs:
            activated = F.relu(conv(embedded))              # (B, num_filters, L')
            pooled_feat = F.max_pool1d(activated, activated.size(2)).squeeze(2)  # (B, num_filters)
            pooled.append(pooled_feat)

        # Concatenate all filter outputs: (batch, num_filters * len(filter_sizes))
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        return self.fc(dropped)  # raw logits


class ToxicDataset(torch.utils.data.Dataset):
    """PyTorch dataset for toxic comment classification."""

    def __init__(self, X: "np.ndarray", y: "np.ndarray" = None):
        import numpy as np
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
