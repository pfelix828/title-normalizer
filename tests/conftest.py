"""Shared test fixtures."""

import pytest
from src.generate_data import generate_dataset
from src.dataset import build_vocab_and_datasets, Vocabulary, tokenize


@pytest.fixture
def small_dataset():
    """Generate a small dataset for fast tests."""
    return generate_dataset(n_samples=500, seed=42)


@pytest.fixture
def train_test_vocab(small_dataset):
    """Build train/val/test datasets and vocabulary."""
    train_ds, val_ds, test_ds, vocab = build_vocab_and_datasets(
        small_dataset, train_ratio=0.70, val_ratio=0.15, seed=42
    )
    return train_ds, test_ds, vocab


@pytest.fixture
def train_val_test_vocab(small_dataset):
    """Build train/val/test datasets and vocabulary (full 4-tuple)."""
    train_ds, val_ds, test_ds, vocab = build_vocab_and_datasets(
        small_dataset, train_ratio=0.70, val_ratio=0.15, seed=42
    )
    return train_ds, val_ds, test_ds, vocab
