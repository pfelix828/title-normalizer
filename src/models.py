"""Neural network architectures for job title classification.

Two models:
1. BiLSTM — word embeddings + bidirectional LSTM + dual classification heads
2. CNNClassifier — word embeddings + 1D convolutions + dual classification heads

Both are multi-task: predict seniority level AND job function simultaneously.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with multi-task classification heads.

    Architecture:
        Input (token IDs) → Embedding → BiLSTM → Concat final hidden states
        → Dropout → Seniority head (Linear)
                  → Function head (Linear)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_seniority: int = 10,
        num_function: int = 10,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.seniority_head = nn.Linear(hidden_dim * 2, num_seniority)
        self.function_head = nn.Linear(hidden_dim * 2, num_function)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Token IDs, shape (batch_size, seq_length)

        Returns:
            (seniority_logits, function_logits) — each shape (batch_size, num_classes)
        """
        embeds = self.embedding(x)  # (B, L, E)
        _, (hidden, _) = self.lstm(embeds)  # hidden: (2*layers, B, H)
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, 2*H)
        hidden = self.dropout(hidden)
        return self.seniority_head(hidden), self.function_head(hidden)


class CNNClassifier(nn.Module):
    """1D CNN with multi-task classification heads.

    Uses multiple kernel sizes to capture n-gram patterns in title tokens.

    Architecture:
        Input → Embedding → [Conv1D(k=2), Conv1D(k=3), Conv1D(k=4)] → MaxPool
        → Concat → Dropout → Seniority head
                             → Function head
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
        num_seniority: int = 10,
        num_function: int = 10,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        concat_dim = num_filters * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.seniority_head = nn.Linear(concat_dim, num_seniority)
        self.function_head = nn.Linear(concat_dim, num_function)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(x)  # (B, L, E)
        embeds = embeds.permute(0, 2, 1)  # (B, E, L) for Conv1d

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(embeds))  # (B, F, L')
            c = c.max(dim=2).values  # (B, F) — global max pooling
            conv_outs.append(c)

        concat = torch.cat(conv_outs, dim=1)  # (B, F*len(kernels))
        concat = self.dropout(concat)
        return self.seniority_head(concat), self.function_head(concat)
