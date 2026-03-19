"""Baseline model: TF-IDF + Logistic Regression.

Provides a classical ML comparison for the neural network models.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from .generate_data import generate_dataset, SENIORITY_LEVELS, FUNCTIONS

SENIORITY_NAMES = {v: k for k, v in SENIORITY_LEVELS.items()}
FUNCTION_NAMES = {v: k for k, v in FUNCTIONS.items()}


def train_baseline(
    n_samples: int = 20000,
    noise_level: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """Train TF-IDF + Logistic Regression baselines for both tasks.

    Returns dict with models, vectorizer, and metrics.
    """
    import random

    records = generate_dataset(n_samples=n_samples, noise_level=noise_level, seed=seed)

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    split = int(len(shuffled) * train_ratio)
    train_records = shuffled[:split]
    test_records = shuffled[split:]

    train_titles = [r["raw_title"] for r in train_records]
    test_titles = [r["raw_title"] for r in test_records]

    train_sen = [r["seniority_id"] for r in train_records]
    test_sen = [r["seniority_id"] for r in test_records]
    train_func = [r["function_id"] for r in train_records]
    test_func = [r["function_id"] for r in test_records]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(train_titles)
    X_test = vectorizer.transform(test_titles)

    # Seniority classifier
    sen_model = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    sen_model.fit(X_train, train_sen)
    sen_preds = sen_model.predict(X_test)
    sen_acc = accuracy_score(test_sen, sen_preds)

    # Function classifier
    func_model = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    func_model.fit(X_train, train_func)
    func_preds = func_model.predict(X_test)
    func_acc = accuracy_score(test_func, func_preds)

    # Combined accuracy
    combined_acc = sum(
        sp == sl and fp == fl
        for sp, sl, fp, fl in zip(sen_preds, test_sen, func_preds, test_func)
    ) / len(test_sen)

    print("=" * 55)
    print("BASELINE: TF-IDF + Logistic Regression")
    print("=" * 55)
    print(f"\nSeniority accuracy: {sen_acc:.1%}")
    print(f"Function accuracy:  {func_acc:.1%}")
    print(f"Both correct:       {combined_acc:.1%}")
    print(f"\nTrain: {len(train_records)} | Test: {len(test_records)}")
    print(f"TF-IDF features: {X_train.shape[1]}")

    return {
        "vectorizer": vectorizer,
        "seniority_model": sen_model,
        "function_model": func_model,
        "seniority_accuracy": sen_acc,
        "function_accuracy": func_acc,
        "combined_accuracy": combined_acc,
    }


if __name__ == "__main__":
    train_baseline()
