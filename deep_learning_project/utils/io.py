"""I/O helpers for saving metrics, figures, and model checkpoints."""

import csv
import json
import pathlib

import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Root output directory — anchored to *this file's* location so the path is
# correct regardless of the notebook's working directory.
# deep_learning_project/utils/io.py  ->  parents[2] == deep-learning/
# ---------------------------------------------------------------------------
OUTPUTS_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2] / "outputs"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure(directory: pathlib.Path) -> pathlib.Path:
    """Create *directory* (and parents) if it does not exist, then return it."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_metrics_json(metrics: dict, filename: str) -> pathlib.Path:
    """Serialize *metrics* to a JSON file inside ``outputs/metrics/``.

    Parameters
    ----------
    metrics:
        A JSON-serialisable dictionary of metric names to values.
    filename:
        Output filename, e.g. ``"mlp_results.json"``.

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
    out_dir = _ensure(OUTPUTS_ROOT / "metrics")
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved to {path}")
    return path


def save_metrics_csv(rows: list, filename: str) -> pathlib.Path:
    """Write a list of dicts as a CSV file inside ``outputs/metrics/``.

    Parameters
    ----------
    rows:
        List of dicts where every dict shares the same keys (column names).
    filename:
        Output filename, e.g. ``"training_history.csv"``.

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    out_dir = _ensure(OUTPUTS_ROOT / "metrics")
    path = out_dir / filename
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {path}")
    return path


def save_figure(fig, filename: str, dpi: int = 150) -> pathlib.Path:
    """Save a matplotlib figure to ``outputs/figures/``.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` instance.
    filename:
        Output filename, e.g. ``"confusion_matrix.png"``.
    dpi:
        Resolution in dots per inch. Default is 150.

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
    out_dir = _ensure(OUTPUTS_ROOT / "figures")
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to {path}")
    return path


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_loss: float,
    filename: str,
) -> pathlib.Path:
    """Save a training checkpoint to ``outputs/models/``.

    The checkpoint dictionary contains:
    ``model_state_dict``, ``optimizer_state_dict``, ``epoch``, ``val_loss``.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` whose ``state_dict`` will be saved.
    optimizer:
        A ``torch.optim.Optimizer`` whose ``state_dict`` will be saved.
    epoch:
        Current (or last completed) training epoch number.
    val_loss:
        Validation loss at the time of saving.
    filename:
        Output filename, e.g. ``"best_mlp.pt"``.

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
    out_dir = _ensure(OUTPUTS_ROOT / "models")
    path = out_dir / filename
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
    return path


def load_checkpoint(model, filename: str, device) -> dict:
    """Load a checkpoint from ``outputs/models/`` into *model* in-place.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` that matches the architecture of the saved
        checkpoint.  Its weights are updated in-place.
    filename:
        Checkpoint filename, e.g. ``"best_mlp.pt"``.
    device:
        The ``torch.device`` to map the checkpoint tensors to.

    Returns
    -------
    dict
        The full checkpoint dictionary (includes ``model_state_dict``,
        ``optimizer_state_dict``, ``epoch``, and ``val_loss``).
    """
    path = OUTPUTS_ROOT / "models" / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint
