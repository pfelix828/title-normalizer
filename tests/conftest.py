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
    """Build train/test datasets and vocabulary."""
    train_ds, test_ds, vocab = build_vocab_and_datasets(
        small_dataset, train_ratio=0.8, seed=42
    )
    return train_ds, test_ds, vocab
