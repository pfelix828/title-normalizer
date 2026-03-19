"""Inference: classify raw job titles using a trained model."""

import torch
import torch.nn as nn

from .dataset import tokenize, Vocabulary
from .generate_data import SENIORITY_LEVELS, FUNCTIONS


SENIORITY_NAMES = {v: k for k, v in SENIORITY_LEVELS.items()}
FUNCTION_NAMES = {v: k for k, v in FUNCTIONS.items()}


def predict_title(
    title: str,
    model: nn.Module,
    vocab: Vocabulary,
    max_length: int = 12,
    device: str = "cpu",
) -> dict:
    """Classify a single raw job title.

    Returns:
        Dict with predicted seniority, function, and confidence scores.
    """
    model.eval()
    tokens = tokenize(title)
    ids = vocab.encode(tokens)

    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids = ids + [0] * (max_length - len(ids))

    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        sen_logits, func_logits = model(x)

    sen_probs = torch.softmax(sen_logits, dim=1)[0]
    func_probs = torch.softmax(func_logits, dim=1)[0]

    sen_id = sen_probs.argmax().item()
    func_id = func_probs.argmax().item()

    return {
        "raw_title": title,
        "seniority": SENIORITY_NAMES[sen_id],
        "seniority_confidence": sen_probs[sen_id].item(),
        "function": FUNCTION_NAMES[func_id],
        "function_confidence": func_probs[func_id].item(),
        "seniority_probs": {SENIORITY_NAMES[i]: p.item() for i, p in enumerate(sen_probs)},
        "function_probs": {FUNCTION_NAMES[i]: p.item() for i, p in enumerate(func_probs)},
    }


def predict_batch(
    titles: list[str],
    model: nn.Module,
    vocab: Vocabulary,
    max_length: int = 12,
    device: str = "cpu",
) -> list[dict]:
    """Classify a batch of raw job titles.

    Returns:
        List of prediction dicts.
    """
    model.eval()

    encoded = []
    for title in titles:
        tokens = tokenize(title)
        ids = vocab.encode(tokens)
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [0] * (max_length - len(ids))
        encoded.append(ids)

    x = torch.tensor(encoded, dtype=torch.long).to(device)

    with torch.no_grad():
        sen_logits, func_logits = model(x)

    sen_probs = torch.softmax(sen_logits, dim=1)
    func_probs = torch.softmax(func_logits, dim=1)

    results = []
    for i, title in enumerate(titles):
        sen_id = sen_probs[i].argmax().item()
        func_id = func_probs[i].argmax().item()
        results.append({
            "raw_title": title,
            "seniority": SENIORITY_NAMES[sen_id],
            "seniority_confidence": sen_probs[i][sen_id].item(),
            "function": FUNCTION_NAMES[func_id],
            "function_confidence": func_probs[i][func_id].item(),
        })

    return results
