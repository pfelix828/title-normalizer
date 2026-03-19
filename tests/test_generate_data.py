"""Tests for synthetic data generation."""

from src.generate_data import (
    generate_dataset,
    generate_messy_title,
    SENIORITY_LEVELS,
    FUNCTIONS,
    TITLE_TEMPLATES,
)
import random


class TestDataGeneration:
    def test_generates_correct_count(self):
        records = generate_dataset(n_samples=100, seed=42)
        assert len(records) == 100

    def test_record_has_required_fields(self):
        records = generate_dataset(n_samples=10, seed=42)
        required = {"raw_title", "canonical_title", "seniority", "function", "seniority_id", "function_id"}
        for r in records:
            assert required.issubset(r.keys())

    def test_seniority_ids_valid(self):
        records = generate_dataset(n_samples=200, seed=42)
        valid_ids = set(SENIORITY_LEVELS.values())
        for r in records:
            assert r["seniority_id"] in valid_ids

    def test_function_ids_valid(self):
        records = generate_dataset(n_samples=200, seed=42)
        valid_ids = set(FUNCTIONS.values())
        for r in records:
            assert r["function_id"] in valid_ids

    def test_reproducibility(self):
        r1 = generate_dataset(n_samples=50, seed=42)
        r2 = generate_dataset(n_samples=50, seed=42)
        assert [r["raw_title"] for r in r1] == [r["raw_title"] for r in r2]

    def test_different_seeds_differ(self):
        r1 = generate_dataset(n_samples=50, seed=42)
        r2 = generate_dataset(n_samples=50, seed=99)
        assert [r["raw_title"] for r in r1] != [r["raw_title"] for r in r2]

    def test_noise_level_zero_preserves_canonical(self):
        records = generate_dataset(n_samples=100, noise_level=0.0, seed=42)
        for r in records:
            assert r["raw_title"] == r["canonical_title"]

    def test_covers_multiple_seniorities(self):
        records = generate_dataset(n_samples=1000, seed=42)
        seniorities = {r["seniority"] for r in records}
        assert len(seniorities) >= 5

    def test_covers_multiple_functions(self):
        records = generate_dataset(n_samples=1000, seed=42)
        functions = {r["function"] for r in records}
        assert len(functions) >= 5

    def test_messy_title_returns_string(self):
        rng = random.Random(42)
        result = generate_messy_title("Senior Data Scientist", "senior", rng)
        assert isinstance(result, str)
        assert len(result) > 0
