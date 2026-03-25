"""Training pipeline for job title classifiers."""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .generate_data import generate_dataset
from .dataset import build_vocab_and_datasets
from .models import BiLSTMClassifier, CNNClassifier
from .evaluate import evaluate, print_report


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    n_samples: int = 20000
    noise_level: float = 1.0
    max_length: int = 12
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Model
    model_type: str = "bilstm"  # "bilstm" or "cnn"
    embed_dim: int = 64
    hidden_dim: int = 128
    num_filters: int = 64
    kernel_sizes: tuple[int, ...] = (2, 3, 4)
    num_layers: int = 1
    dropout: float = 0.3

    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5
    seed: int = 42

    # Device
    device: str = ""

    def __post_init__(self):
        if not self.device:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"


def build_model(config: TrainConfig, vocab_size: int) -> nn.Module:
    """Instantiate model from config."""
    if config.model_type == "bilstm":
        return BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_seniority=10,
            num_function=10,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif config.model_type == "cnn":
        return CNNClassifier(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            num_seniority=10,
            num_function=10,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        seniority = batch["seniority"].to(device)
        function = batch["function"].to(device)

        optimizer.zero_grad()
        sen_logits, func_logits = model(tokens)
        loss = criterion(sen_logits, seniority) + criterion(func_logits, function)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float]:
    """Validate model. Returns (loss, seniority_accuracy, function_accuracy)."""
    model.eval()
    total_loss = 0.0
    sen_correct = 0
    func_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            seniority = batch["seniority"].to(device)
            function = batch["function"].to(device)

            sen_logits, func_logits = model(tokens)
            loss = criterion(sen_logits, seniority) + criterion(func_logits, function)
            total_loss += loss.item()

            sen_correct += (sen_logits.argmax(1) == seniority).sum().item()
            func_correct += (func_logits.argmax(1) == function).sum().item()
            total += len(tokens)

    n = len(dataloader)
    return total_loss / n, sen_correct / total, func_correct / total


def train(config: Optional[TrainConfig] = None) -> dict:
    """Full training pipeline.

    Returns:
        Dict with model, vocab, config, and metrics.
    """
    if config is None:
        config = TrainConfig()

    print(f"Device: {config.device}")
    print(f"Model: {config.model_type}")
    print(f"Generating {config.n_samples} titles...")

    # Generate data
    records = generate_dataset(
        n_samples=config.n_samples,
        noise_level=config.noise_level,
        seed=config.seed,
    )

    # Build datasets
    train_ds, val_ds, test_ds, vocab = build_vocab_and_datasets(
        records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        max_length=config.max_length,
        seed=config.seed,
    )
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # Model
    model = build_model(config, len(vocab)).to(config.device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\nTraining for up to {config.epochs} epochs (patience={config.patience})...\n")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Sen Acc':>8} {'Func Acc':>9} {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, sen_acc, func_acc = validate(model, val_loader, criterion, config.device)

        elapsed = time.time() - t0
        print(f"{epoch:>5} {train_loss:>11.4f} {val_loss:>9.4f} {sen_acc:>7.1%} {func_acc:>8.1%} {elapsed:>5.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(config.device)

    # Final evaluation
    print("\n" + "=" * 55)
    print("Final evaluation on test set:")
    print("=" * 55)
    metrics = evaluate(model, test_loader, config.device)
    print_report(metrics)

    return {
        "model": model,
        "vocab": vocab,
        "config": config,
        "metrics": metrics,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }


if __name__ == "__main__":
    results = train()
