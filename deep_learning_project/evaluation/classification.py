"""Classification evaluation: metrics, confusion matrix, and training history plots."""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple

from deep_learning_project.utils.io import save_metrics_json, save_figure


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model over the loader and return (y_true, y_pred) as numpy arrays.

    Supports two batch formats:
        - (X, y)            -- regular loaders
        - (X, lengths, y)   -- sequence loaders with packed padding
    """
    model.eval()

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                X, y = batch
                X = X.to(device)
                logits = model(X)
            else:
                X, lengths, y = batch
                X, lengths = X.to(device), lengths.to(device)
                logits = model(X, lengths)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_true.append(y.numpy() if not isinstance(y, np.ndarray) else y)
            all_pred.append(preds)

    return np.concatenate(all_true), np.concatenate(all_pred)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return float((y_true == y_pred).mean())


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> float:
    """
    Weighted precision.

    precision_c = TP_c / (TP_c + FP_c)

    Classes that are never predicted receive precision=0.
    """
    classes = np.unique(y_true)
    support = np.array([(y_true == c).sum() for c in classes], dtype=float)
    total_support = support.sum()

    precisions = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        denom = tp + fp
        precisions.append(tp / denom if denom > 0 else 0.0)

    if average == "weighted":
        return float(np.dot(precisions, support) / total_support) if total_support > 0 else 0.0
    elif average == "macro":
        return float(np.mean(precisions))
    else:
        raise ValueError(f"Unsupported average='{average}'. Use 'weighted' or 'macro'.")


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> float:
    """
    Weighted recall.

    recall_c = TP_c / (TP_c + FN_c) = TP_c / support_c
    """
    classes = np.unique(y_true)
    support = np.array([(y_true == c).sum() for c in classes], dtype=float)
    total_support = support.sum()

    recalls = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        sup_c = (y_true == c).sum()
        recalls.append(tp / sup_c if sup_c > 0 else 0.0)

    if average == "weighted":
        return float(np.dot(recalls, support) / total_support) if total_support > 0 else 0.0
    elif average == "macro":
        return float(np.mean(recalls))
    else:
        raise ValueError(f"Unsupported average='{average}'. Use 'weighted' or 'macro'.")


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> float:
    """
    Weighted F1.

    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)

    Classes with both precision and recall equal to 0 receive f1=0.
    """
    classes = np.unique(y_true)
    support = np.array([(y_true == c).sum() for c in classes], dtype=float)
    total_support = support.sum()

    f1s = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        sup_c = (y_true == c).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / sup_c if sup_c > 0 else 0.0
        denom = prec + rec
        f1s.append(2 * prec * rec / denom if denom > 0 else 0.0)

    if average == "weighted":
        return float(np.dot(f1s, support) / total_support) if total_support > 0 else 0.0
    elif average == "macro":
        return float(np.mean(f1s))
    else:
        raise ValueError(f"Unsupported average='{average}'. Use 'weighted' or 'macro'.")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute confusion matrix using pure numpy.

    cm[i, j] = number of samples with true label i predicted as j.

    Args:
        y_true:    Ground-truth class indices.
        y_pred:    Predicted class indices.
        n_classes: Total number of classes. Inferred from data if None.

    Returns:
        Integer ndarray of shape (n_classes, n_classes).
    """
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def classification_report(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run the model, compute all metrics, and persist a JSON summary.

    Returns:
        dict with keys:
            "accuracy", "precision", "recall", "f1",
            "confusion_matrix" (ndarray), "class_names" (list)
    """
    y_true, y_pred = get_predictions(model, loader, device)

    n_classes = int(y_true.max()) + 1
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    acc = compute_accuracy(y_true, y_pred)
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)

    report = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "class_names": class_names,
    }

    # Persist a JSON-serialisable version (ndarray -> list)
    json_report = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in report.items()}
    save_metrics_json(json_report, "classification_report.json")

    return report


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_filename: Optional[str] = None,
) -> Figure:
    """
    Plot a confusion matrix as an annotated seaborn heatmap.

    Args:
        cm:            Confusion matrix of shape (n_classes, n_classes).
        class_names:   Labels for each class.
        title:         Figure title.
        save_filename: If given, the figure is saved via save_figure().

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()

    if save_filename is not None:
        save_figure(fig, save_filename)

    return fig


def plot_training_history(
    history: Dict,
    title: str = "Training History",
    save_filename: Optional[str] = None,
) -> Figure:
    """
    Plot training and validation loss and accuracy over epochs.

    Left panel:  loss curves (train + val)
    Right panel: accuracy curves (train + val)

    Args:
        history:       Dict with keys "train_loss", "val_loss",
                       "train_acc", "val_acc".
        title:         Overall figure title.
        save_filename: If given, the figure is saved via save_figure().

    Returns:
        matplotlib Figure object.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Loss panel
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax_loss.plot(epochs, history["val_loss"], label="Val Loss", marker="o", markersize=3)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    # Accuracy panel
    ax_acc.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax_acc.plot(epochs, history["val_acc"], label="Val Acc", marker="o", markersize=3)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    fig.tight_layout()

    if save_filename is not None:
        save_figure(fig, save_filename)

    return fig
