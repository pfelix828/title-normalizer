"""PyTorch Dataset and vocabulary for job title classification."""

import re
from collections import Counter

import torch
from torch.utils.data import Dataset


def tokenize(title: str) -> list[str]:
    """Tokenize a job title into lowercase words.

    Handles abbreviations, punctuation, and extra whitespace.
    """
    title = title.lower().strip()
    title = re.sub(r"[&/–]", " ", title)
    title = re.sub(r"[^\w\s.-]", "", title)
    title = re.sub(r"\s+", " ", title)
    tokens = title.split()
    return tokens


class Vocabulary:
    """Maps tokens to integer IDs.

    Reserves 0 for padding and 1 for unknown tokens.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2id: dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.id2token: dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, tokenized_titles: list[list[str]]) -> "Vocabulary":
        """Build vocabulary from tokenized titles."""
        counter = Counter(tok for tokens in tokenized_titles for tok in tokens)
        for token, count in counter.most_common():
            if count >= self.min_freq:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
        return self

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens to IDs."""
        unk_id = self.token2id[self.UNK_TOKEN]
        return [self.token2id.get(t, unk_id) for t in tokens]

    def __len__(self) -> int:
        return len(self.token2id)


class TitleDataset(Dataset):
    """PyTorch Dataset for job title classification.

    Each sample contains tokenized + padded title IDs and two labels:
    seniority level and job function.
    """

    def __init__(
        self,
        records: list[dict],
        vocab: Vocabulary,
        max_length: int = 12,
    ):
        self.records = records
        self.vocab = vocab
        self.max_length = max_length

        # Pre-tokenize and encode all titles
        self.encoded = []
        for r in records:
            tokens = tokenize(r["raw_title"])
            ids = vocab.encode(tokens)
            # Truncate or pad
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [0] * (max_length - len(ids))
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.tensor(self.encoded[idx], dtype=torch.long),
            "seniority": torch.tensor(self.records[idx]["seniority_id"], dtype=torch.long),
            "function": torch.tensor(self.records[idx]["function_id"], dtype=torch.long),
        }


def build_vocab_and_datasets(
    records: list[dict],
    train_ratio: float = 0.8,
    max_length: int = 12,
    min_freq: int = 2,
    seed: int = 42,
) -> tuple[TitleDataset, TitleDataset, Vocabulary]:
    """Split records and build vocabulary from training set only.

    Returns:
        (train_dataset, test_dataset, vocabulary)
    """
    import random
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    split = int(len(shuffled) * train_ratio)
    train_records = shuffled[:split]
    test_records = shuffled[split:]

    # Build vocab from training data only
    train_tokens = [tokenize(r["raw_title"]) for r in train_records]
    vocab = Vocabulary(min_freq=min_freq).build(train_tokens)

    train_ds = TitleDataset(train_records, vocab, max_length)
    test_ds = TitleDataset(test_records, vocab, max_length)

    return train_ds, test_ds, vocab
