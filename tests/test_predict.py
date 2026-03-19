"""Tests for prediction / inference."""

from src.generate_data import generate_dataset
from src.dataset import build_vocab_and_datasets
from src.models import BiLSTMClassifier
from src.predict import predict_title, predict_batch


def _make_trained_model():
    """Build a small model with vocab for testing."""
    records = generate_dataset(n_samples=200, seed=42)
    train_ds, _, vocab = build_vocab_and_datasets(records, train_ratio=0.8, seed=42)
    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=16, hidden_dim=32,
        num_seniority=10, num_function=10,
    )
    return model, vocab


class TestPredictTitle:
    def test_returns_required_keys(self):
        model, vocab = _make_trained_model()
        result = predict_title("Senior Software Engineer", model, vocab)
        assert "seniority" in result
        assert "function" in result
        assert "seniority_confidence" in result
        assert "function_confidence" in result

    def test_confidence_between_0_and_1(self):
        model, vocab = _make_trained_model()
        result = predict_title("VP of Marketing", model, vocab)
        assert 0.0 <= result["seniority_confidence"] <= 1.0
        assert 0.0 <= result["function_confidence"] <= 1.0

    def test_handles_messy_input(self):
        model, vocab = _make_trained_model()
        result = predict_title("  Sr. Dir.\tMktg (EMEA)  ", model, vocab)
        assert isinstance(result["seniority"], str)
        assert isinstance(result["function"], str)


class TestPredictBatch:
    def test_batch_returns_correct_count(self):
        model, vocab = _make_trained_model()
        titles = ["CTO", "Sales Rep", "Senior Data Scientist"]
        results = predict_batch(titles, model, vocab)
        assert len(results) == 3

    def test_batch_matches_single(self):
        model, vocab = _make_trained_model()
        title = "Engineering Manager"
        single = predict_title(title, model, vocab)
        batch = predict_batch([title], model, vocab)
        assert single["seniority"] == batch[0]["seniority"]
        assert single["function"] == batch[0]["function"]
