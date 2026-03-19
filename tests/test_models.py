"""Tests for neural network models."""

import torch
from src.models import BiLSTMClassifier, CNNClassifier


class TestBiLSTMClassifier:
    def setup_method(self):
        self.model = BiLSTMClassifier(
            vocab_size=100, embed_dim=32, hidden_dim=64,
            num_seniority=10, num_function=10,
        )

    def test_output_shapes(self):
        x = torch.randint(0, 100, (4, 12))
        sen, func = self.model(x)
        assert sen.shape == (4, 10)
        assert func.shape == (4, 10)

    def test_single_sample(self):
        x = torch.randint(0, 100, (1, 12))
        sen, func = self.model(x)
        assert sen.shape == (1, 10)
        assert func.shape == (1, 10)

    def test_gradients_flow(self):
        x = torch.randint(0, 100, (4, 12))
        sen, func = self.model(x)
        loss = sen.sum() + func.sum()
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestCNNClassifier:
    def setup_method(self):
        self.model = CNNClassifier(
            vocab_size=100, embed_dim=32, num_filters=32,
            num_seniority=10, num_function=10,
        )

    def test_output_shapes(self):
        x = torch.randint(0, 100, (4, 12))
        sen, func = self.model(x)
        assert sen.shape == (4, 10)
        assert func.shape == (4, 10)

    def test_single_sample(self):
        x = torch.randint(0, 100, (1, 12))
        sen, func = self.model(x)
        assert sen.shape == (1, 10)
        assert func.shape == (1, 10)

    def test_gradients_flow(self):
        x = torch.randint(0, 100, (4, 12))
        sen, func = self.model(x)
        loss = sen.sum() + func.sum()
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad:
                assert p.grad is not None
