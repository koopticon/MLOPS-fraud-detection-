"""Metrics and diagnostic plots tuned for a highly imbalanced problem.

Accuracy is deliberately *not* the headline: with ~0.2% frauds a model that
predicts "never fraud" scores 99.8% accuracy and catches nothing. The metric
that matters is the precision-recall trade-off, summarised by PR-AUC.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return the metric set we log to MLflow for every run/batch."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def best_threshold(y_true, y_prob) -> float:
    """Pick the probability cutoff that maximises F1 on the PR curve."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    # thr has length len(prec)-1; align indices.
    return float(thr[max(0, np.argmax(f1[:-1]))]) if len(thr) else 0.5


def save_diagnostics(y_true, y_prob, out_dir, prefix: str = "eval") -> list:
    """Save a PR curve and a confusion matrix; return the file paths."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2)
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"Precision-Recall (AP={ap:.3f})", xlim=(0, 1), ylim=(0, 1.02))
    fig.tight_layout()
    pr_path = out_dir / f"{prefix}_pr_curve.png"
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    paths.append(str(pr_path))

    thr = best_threshold(y_true, y_prob)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["legit", "fraud"], yticklabels=["legit", "fraud"],
           xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix @ thr={thr:.3f}")
    fig.tight_layout()
    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    paths.append(str(cm_path))
    return paths
