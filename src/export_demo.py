"""Train the demo model, save weights, and export everything the browser demo needs.

Outputs:
    models/<type>.pt          — checkpoint (state_dict + config + vocab)
    models/metrics.json       — test-set metrics from this training run
    docs/model/model.onnx     — fixed-shape (1, 12) ONNX export for onnxruntime-web
    docs/model/vocab.json     — token2id, label names, max_length
    docs/model/fixtures.json  — tokenizer + end-to-end parity fixtures for the JS side

Run: python -m src.export_demo [bilstm|cnn]   (default: cnn)
"""

import json
from pathlib import Path

import numpy as np
import torch

from .dataset import tokenize
from .generate_data import SENIORITY_LEVELS, FUNCTIONS
from .train import TrainConfig, train

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DEMO_MODEL_DIR = ROOT / "docs" / "model"

# Deliberately messy inputs to pin down tokenizer parity between Python and JS.
ADVERSARIAL_TITLES = [
    "Sr. Dir. of Eng, EMEA",
    "VP Sales & Marketing",
    "Chief People Officer",
    "sr software engineer ii",
    "Engineering Manager — Platform",
    "Product Mgr / Growth",
    "DIRECTOR, DATA ANALYTICS",
    "Head of Design (Brand)",
    "Développeur Senior",
    "Sr.  Account   Executive",
    "C.F.O.",
    "Staff ML Engineer, Infra & Tools",
    "principal pm – payments",
    "Jr. HR Generalist!!",
    "General Counsel 🏛️",
    "Senior Vice President, Global Customer Operations and Strategic Partnerships",
    "engineer",
    "???",
    "VP\tFinance",
    "Mkt Ops Lead",
]


def encode_for_model(title: str, token2id: dict, max_length: int) -> list[int]:
    tokens = tokenize(title)
    unk = token2id["<UNK>"]
    ids = [token2id.get(t, unk) for t in tokens]
    return ids[:max_length] + [0] * max(0, max_length - len(ids))


def main() -> None:
    import sys

    model_type = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    torch.manual_seed(42)
    config = TrainConfig(model_type=model_type)
    result = train(config)

    model = result["model"].cpu().eval()
    vocab = result["vocab"]
    test_ds = result["test_dataset"]
    max_length = config.max_length

    MODELS_DIR.mkdir(exist_ok=True)
    DEMO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Checkpoint ---
    torch.save(
        {
            "state_dict": model.state_dict(),
            "token2id": vocab.token2id,
            "max_length": max_length,
            "model_type": config.model_type,
            "embed_dim": config.embed_dim,
            "hidden_dim": config.hidden_dim,
            "num_filters": config.num_filters,
            "kernel_sizes": config.kernel_sizes,
            "num_layers": config.num_layers,
        },
        MODELS_DIR / f"{config.model_type}.pt",
    )

    # --- 2. Test metrics from this run ---
    metrics = result["metrics"]
    serializable = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float, str, dict, list))
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(serializable, indent=2, default=float))

    # --- 3. ONNX export (fixed batch=1, length=12; TorchScript exporter for stable op mapping) ---
    dummy = torch.zeros((1, max_length), dtype=torch.long)
    onnx_path = DEMO_MODEL_DIR / "model.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["tokens"],
        output_names=["seniority_logits", "function_logits"],
        opset_version=17,
        dynamo=False,
    )
    print(f"ONNX written: {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    # --- 4. Vocab + labels for the JS side ---
    sen_names = [None] * len(SENIORITY_LEVELS)
    for name, idx in SENIORITY_LEVELS.items():
        sen_names[idx] = name
    func_names = [None] * len(FUNCTIONS)
    for name, idx in FUNCTIONS.items():
        func_names[idx] = name
    (DEMO_MODEL_DIR / "vocab.json").write_text(
        json.dumps(
            {
                "token2id": vocab.token2id,
                "max_length": max_length,
                "seniority_names": sen_names,
                "function_names": func_names,
            }
        )
    )

    # --- 5. Parity fixtures: 100 test titles + adversarial set, with PyTorch outputs ---
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(test_ds.records), size=100, replace=False)
    fixture_titles = [test_ds.records[i]["raw_title"] for i in sample_idx] + ADVERSARIAL_TITLES

    fixtures = []
    for title in fixture_titles:
        ids = encode_for_model(title, vocab.token2id, max_length)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            sen_logits, func_logits = model(x)
        sen_probs = torch.softmax(sen_logits, dim=1)[0]
        func_probs = torch.softmax(func_logits, dim=1)[0]
        fixtures.append(
            {
                "title": title,
                "ids": ids,
                "seniority": sen_names[int(sen_probs.argmax())],
                "function": func_names[int(func_probs.argmax())],
                "seniority_confidence": round(float(sen_probs.max()), 6),
                "function_confidence": round(float(func_probs.max()), 6),
            }
        )
    (DEMO_MODEL_DIR / "fixtures.json").write_text(json.dumps(fixtures))
    print(f"Fixtures written: {len(fixtures)} cases")

    # --- 6. ONNX parity vs PyTorch on the full test set ---
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    mismatches = 0
    max_prob_diff = 0.0
    for i in range(len(test_ds)):
        ids = test_ds.encoded[i]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            sen_t, func_t = model(x)
        sen_o, func_o = sess.run(None, {"tokens": np.array([ids], dtype=np.int64)})
        if int(sen_t.argmax()) != int(np.argmax(sen_o)) or int(func_t.argmax()) != int(
            np.argmax(func_o)
        ):
            mismatches += 1
        max_prob_diff = max(
            max_prob_diff,
            float(np.abs(torch.softmax(sen_t, 1).numpy() - _softmax(sen_o)).max()),
            float(np.abs(torch.softmax(func_t, 1).numpy() - _softmax(func_o)).max()),
        )
    print(f"\nONNX parity on {len(test_ds)} test rows: {mismatches} argmax mismatches, "
          f"max softmax diff {max_prob_diff:.2e}")
    if mismatches:
        raise SystemExit("ONNX export does not match PyTorch — do not ship.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
