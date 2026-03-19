"""Generate PDF evaluation report with visualizations."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .generate_data import generate_dataset
from .dataset import build_vocab_and_datasets
from .train import train, TrainConfig
from .evaluate import evaluate, SENIORITY_NAMES, FUNCTION_NAMES


def _plot_accuracy_comparison(ax, results: dict):
    """Bar chart comparing model accuracies."""
    models = list(results.keys())
    sen_acc = [results[m]["seniority_accuracy"] for m in models]
    func_acc = [results[m]["function_accuracy"] for m in models]
    comb_acc = [results[m]["combined_accuracy"] for m in models]

    x = np.arange(len(models))
    w = 0.25

    ax.bar(x - w, sen_acc, w, label="Seniority", color="#4C72B0")
    ax.bar(x, func_acc, w, label="Function", color="#55A868")
    ax.bar(x + w, comb_acc, w, label="Both Correct", color="#C44E52")

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.98, 1.002)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def _plot_per_class_f1(ax, metrics: list[dict], title: str):
    """Horizontal bar chart of per-class F1 scores."""
    names = [m["name"] for m in metrics]
    f1s = [m["f1"] for m in metrics]
    supports = [m["support"] for m in metrics]

    colors = ["#4C72B0" if f >= 0.99 else "#C44E52" for f in f1s]
    y = np.arange(len(names))
    bars = ax.barh(y, f1s, color=colors, edgecolor="white", linewidth=0.5)

    for i, (f1, sup) in enumerate(zip(f1s, supports)):
        ax.text(f1 - 0.003, i, f"{f1:.1%} (n={sup})", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.95, 1.005)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def _plot_confusion_heatmap(ax, metrics: list[dict], title: str):
    """Plot precision vs recall scatter for each class."""
    names = [m["name"] for m in metrics]
    prec = [m["precision"] for m in metrics]
    rec = [m["recall"] for m in metrics]

    ax.scatter(rec, prec, s=80, c="#4C72B0", edgecolors="white", zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (rec[i], prec[i]), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")

    ax.plot([0.95, 1.01], [0.95, 1.01], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlim(0.95, 1.005)
    ax.set_ylim(0.95, 1.005)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def generate_report(output_path: str = "data/evaluation_report.pdf"):
    """Train all models and produce a multi-page PDF report."""
    print("=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)

    # Train BiLSTM
    print("\n[1/3] Training BiLSTM...")
    bilstm_results = train(TrainConfig(model_type="bilstm"))

    # Train CNN
    print("\n[2/3] Training CNN...")
    cnn_results = train(TrainConfig(model_type="cnn"))

    # Run baseline
    print("\n[3/3] Running baseline...")
    from .baseline import train_baseline
    baseline_results = train_baseline()
    baseline_metrics = {
        "seniority_accuracy": baseline_results["seniority_accuracy"],
        "function_accuracy": baseline_results["function_accuracy"],
        "combined_accuracy": baseline_results["combined_accuracy"],
    }

    all_results = {
        "TF-IDF + LogReg": baseline_metrics,
        "BiLSTM": bilstm_results["metrics"],
        "CNN": cnn_results["metrics"],
    }

    # Generate PDF
    print(f"\nGenerating PDF report: {output_path}")
    with PdfPages(output_path) as pdf:
        # Page 1: Title page and accuracy comparison
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle("Job Title Normalizer — Evaluation Report", fontsize=16, fontweight="bold", y=0.98)

        axes[0].axis("off")
        summary_text = (
            f"Dataset: 20,000 synthetic titles (16,000 train / 4,000 test)\n"
            f"Tasks: 10 seniority levels × 10 job functions\n"
            f"Models: TF-IDF + Logistic Regression, BiLSTM, CNN\n\n"
        )
        for name, metrics in all_results.items():
            summary_text += (
                f"{name:20s}  "
                f"Seniority: {metrics['seniority_accuracy']:.1%}  "
                f"Function: {metrics['function_accuracy']:.1%}  "
                f"Combined: {metrics['combined_accuracy']:.1%}\n"
            )
        axes[0].text(0.05, 0.5, summary_text, transform=axes[0].transAxes,
                     fontsize=11, verticalalignment="center", fontfamily="monospace")

        _plot_accuracy_comparison(axes[1], all_results)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: BiLSTM per-class F1
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle("BiLSTM — Per-Class F1 Scores", fontsize=14, fontweight="bold")
        _plot_per_class_f1(axes[0], bilstm_results["metrics"]["seniority_metrics"], "Seniority")
        _plot_per_class_f1(axes[1], bilstm_results["metrics"]["function_metrics"], "Function")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: CNN per-class F1
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle("CNN — Per-Class F1 Scores", fontsize=14, fontweight="bold")
        _plot_per_class_f1(axes[0], cnn_results["metrics"]["seniority_metrics"], "Seniority")
        _plot_per_class_f1(axes[1], cnn_results["metrics"]["function_metrics"], "Function")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Precision vs Recall scatter
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Precision vs Recall by Class", fontsize=14, fontweight="bold")
        _plot_confusion_heatmap(axes[0, 0], bilstm_results["metrics"]["seniority_metrics"], "BiLSTM — Seniority")
        _plot_confusion_heatmap(axes[0, 1], bilstm_results["metrics"]["function_metrics"], "BiLSTM — Function")
        _plot_confusion_heatmap(axes[1, 0], cnn_results["metrics"]["seniority_metrics"], "CNN — Seniority")
        _plot_confusion_heatmap(axes[1, 1], cnn_results["metrics"]["function_metrics"], "CNN — Function")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nReport saved to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()
