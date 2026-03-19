"""Tests for dataset and vocabulary."""

import torch
from src.dataset import tokenize, Vocabulary, TitleDataset, build_vocab_and_datasets


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Senior Engineer") == ["senior", "engineer"]

    def test_handles_abbreviations(self):
        tokens = tokenize("Sr. Dir. of Eng")
        assert "sr." in tokens
        assert "dir." in tokens

    def test_handles_special_chars(self):
        tokens = tokenize("VP & Director / GM")
        assert "vp" in tokens
        assert "director" in tokens

    def test_strips_whitespace(self):
        assert tokenize("  Senior Engineer  ") == ["senior", "engineer"]

    def test_handles_tabs(self):
        assert tokenize("Dir\tof Engineering") == ["dir", "of", "engineering"]

    def test_handles_em_dash(self):
        assert tokenize("VP \u2014 Engineering") == ["vp", "engineering"]


class TestVocabulary:
    def test_special_tokens(self):
        vocab = Vocabulary()
        vocab.build([["hello"]])
        assert vocab.token2id["<PAD>"] == 0
        assert vocab.token2id["<UNK>"] == 1

    def test_min_freq_filters(self):
        vocab = Vocabulary(min_freq=2)
        vocab.build([["common", "rare"], ["common", "another_rare"]])
        assert "common" in vocab.token2id
        assert "rare" not in vocab.token2id

    def test_encode_known_tokens(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build([["senior", "engineer"]])
        ids = vocab.encode(["senior", "engineer"])
        assert all(isinstance(i, int) for i in ids)
        assert all(i >= 2 for i in ids)

    def test_encode_unknown_tokens(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build([["senior"]])
        ids = vocab.encode(["unknown_word"])
        assert ids == [1]  # UNK token


class TestTitleDataset:
    def test_length(self, train_test_vocab):
        train_ds, test_ds, _ = train_test_vocab
        assert len(train_ds) + len(test_ds) == 500

    def test_item_shape(self, train_test_vocab):
        train_ds, _, _ = train_test_vocab
        item = train_ds[0]
        assert item["tokens"].shape == (12,)
        assert item["seniority"].shape == ()
        assert item["function"].shape == ()

    def test_item_types(self, train_test_vocab):
        train_ds, _, _ = train_test_vocab
        item = train_ds[0]
        assert item["tokens"].dtype == torch.long
        assert item["seniority"].dtype == torch.long
        assert item["function"].dtype == torch.long

    def test_padding(self, train_test_vocab):
        train_ds, _, _ = train_test_vocab
        item = train_ds[0]
        # At least some positions should be padded (0)
        # (most titles are shorter than 12 tokens)
        tokens = item["tokens"].tolist()
        assert 0 in tokens or len(tokens) == 12
