"""Evaluation metrics for multi-task title classification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from .generate_data import SENIORITY_LEVELS, FUNCTIONS


# Reverse mappings for human-readable labels
SENIORITY_NAMES = {v: k for k, v in SENIORITY_LEVELS.items()}
FUNCTION_NAMES = {v: k for k, v in FUNCTIONS.items()}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Compute per-class precision, recall, F1 for both tasks.

    Returns dict with keys: seniority_metrics, function_metrics,
    seniority_accuracy, function_accuracy, combined_accuracy
    """
    model.eval()

    all_sen_preds = []
    all_sen_labels = []
    all_func_preds = []
    all_func_labels = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            seniority = batch["seniority"]
            function = batch["function"]

            sen_logits, func_logits = model(tokens)

            all_sen_preds.extend(sen_logits.argmax(1).cpu().tolist())
            all_sen_labels.extend(seniority.tolist())
            all_func_preds.extend(func_logits.argmax(1).cpu().tolist())
            all_func_labels.extend(function.tolist())

    sen_metrics = _per_class_metrics(all_sen_labels, all_sen_preds, SENIORITY_NAMES)
    func_metrics = _per_class_metrics(all_func_labels, all_func_preds, FUNCTION_NAMES)

    sen_acc = sum(p == l for p, l in zip(all_sen_preds, all_sen_labels)) / len(all_sen_labels)
    func_acc = sum(p == l for p, l in zip(all_func_preds, all_func_labels)) / len(all_func_labels)
    combined_acc = sum(
        sp == sl and fp == fl
        for sp, sl, fp, fl in zip(all_sen_preds, all_sen_labels, all_func_preds, all_func_labels)
    ) / len(all_sen_labels)

    return {
        "seniority_metrics": sen_metrics,
        "function_metrics": func_metrics,
        "seniority_accuracy": sen_acc,
        "function_accuracy": func_acc,
        "combined_accuracy": combined_acc,
        "n_samples": len(all_sen_labels),
    }


def _per_class_metrics(
    labels: list[int], preds: list[int], names: dict[int, str]
) -> list[dict]:
    """Compute precision, recall, F1 per class."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for label, pred in zip(labels, preds):
        support[label] += 1
        if pred == label:
            tp[label] += 1
        else:
            fp[pred] += 1
            fn[label] += 1

    results = []
    for class_id in sorted(names.keys()):
        name = names[class_id]
        t = tp[class_id]
        precision = t / (t + fp[class_id]) if (t + fp[class_id]) > 0 else 0.0
        recall = t / (t + fn[class_id]) if (t + fn[class_id]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({
            "class_id": class_id,
            "name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[class_id],
        })

    return results


def print_report(metrics: dict) -> None:
    """Print a formatted classification report."""
    print(f"\nSeniority Classification (accuracy: {metrics['seniority_accuracy']:.1%})")
    print(f"{'Class':<25} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8}")
    print("-" * 55)
    for m in metrics["seniority_metrics"]:
        print(f"{m['name']:<25} {m['precision']:>5.1%} {m['recall']:>6.1%} {m['f1']:>5.1%} {m['support']:>8}")

    print(f"\nFunction Classification (accuracy: {metrics['function_accuracy']:.1%})")
    print(f"{'Class':<25} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8}")
    print("-" * 55)
    for m in metrics["function_metrics"]:
        print(f"{m['name']:<25} {m['precision']:>5.1%} {m['recall']:>6.1%} {m['f1']:>5.1%} {m['support']:>8}")

    print(f"\nBoth correct (seniority AND function): {metrics['combined_accuracy']:.1%}")
    print(f"Total test samples: {metrics['n_samples']}")
