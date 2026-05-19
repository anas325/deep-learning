"""Training utilities for MLP, RNN-based classifiers, and Seq2Seq models."""

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple

from deep_learning_project.utils.io import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Run one full pass over the training data.

    Returns:
        (avg_loss, accuracy) as floats.
    """
    model.train()

    total_loss = 0.0
    n_correct = 0
    n_total = 0

    for batch in loader:
        # Support both (X, y) and (X, lengths, y) batch formats
        if len(batch) == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
            logits = model(X)
        else:
            X, lengths, y = batch
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            logits = model(X, lengths)

        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        n_correct += (preds == y).sum().item()
        n_total += y.size(0)

    avg_loss = total_loss / n_total
    accuracy = n_correct / n_total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on a data loader without updating weights.

    Returns:
        (avg_loss, accuracy) as floats.
    """
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                X, y = batch
                X, y = X.to(device), y.to(device)
                logits = model(X)
            else:
                X, lengths, y = batch
                X, lengths, y = X.to(device), lengths.to(device), y.to(device)
                logits = model(X, lengths)

            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)

    avg_loss = total_loss / n_total
    accuracy = n_correct / n_total
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 50,
    grad_clip: Optional[float] = None,
    checkpoint_filename: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Full training loop for classification models (MLP or RNN-based).

    Tracks per-epoch train/val loss and accuracy. Saves a checkpoint whenever
    the validation loss improves.

    Returns:
        history dict with keys:
            "train_loss", "train_acc", "val_loss", "val_acc", "best_epoch"
    """
    history: Dict[str, List] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
    }

    best_val_loss = float("inf")
    epoch_iter = range(1, epochs + 1)

    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    for epoch in epoch_iter:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch

            if checkpoint_filename is not None:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    checkpoint_filename,
                )

        if verbose:
            epoch_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
            )

    return history


def train_seq2seq(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    epochs: int = 30,
    teacher_force_ratio: float = 0.5,
    grad_clip: float = 1.0,
    checkpoint_filename: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Training loop for Seq2Seq models.

    Each batch from the loader is expected to be a 4-tuple:
        (src_padded, src_lengths, tgt_padded, tgt_lengths)

    Validation is run with teacher_force_ratio=0.0 (greedy decoding).
    Gradient clipping is always applied (essential for RNN stability).

    Returns:
        history dict with keys:
            "train_loss", "train_acc", "val_loss", "val_acc", "best_epoch"
        Note: "acc" values are always 0.0 for seq2seq (token-level accuracy
        is not tracked here; use BLEU or similar post-hoc).
    """
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    history: Dict[str, List] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
    }

    best_val_loss = float("inf")
    epoch_iter = range(1, epochs + 1)

    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Seq2Seq Training", unit="epoch")

    for epoch in epoch_iter:
        # ---- Training ----
        model.train()
        train_total_loss = 0.0
        train_n_tokens = 0

        for src_padded, src_lengths, tgt_padded, tgt_lengths in train_loader:
            src_padded = src_padded.to(device)
            src_lengths = src_lengths.to(device)
            tgt_padded = tgt_padded.to(device)
            tgt_lengths = tgt_lengths.to(device)

            # logits: (batch, tgt_len-1, vocab)
            logits = model(src_padded, src_lengths, tgt_padded, teacher_force_ratio)

            batch_size, tgt_len_minus1, vocab_size = logits.shape

            # Reshape for CrossEntropyLoss
            logits_2d = logits.reshape(batch_size * tgt_len_minus1, vocab_size)
            targets_1d = tgt_padded[:, 1:].reshape(batch_size * tgt_len_minus1)

            loss = criterion(logits_2d, targets_1d)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Count non-pad tokens for loss averaging
            n_tokens = (targets_1d != pad_idx).sum().item()
            train_total_loss += loss.item() * n_tokens
            train_n_tokens += n_tokens

        train_loss = train_total_loss / max(train_n_tokens, 1)

        # ---- Validation (greedy, no teacher forcing) ----
        model.eval()
        val_total_loss = 0.0
        val_n_tokens = 0

        with torch.no_grad():
            for src_padded, src_lengths, tgt_padded, tgt_lengths in val_loader:
                src_padded = src_padded.to(device)
                src_lengths = src_lengths.to(device)
                tgt_padded = tgt_padded.to(device)
                tgt_lengths = tgt_lengths.to(device)

                logits = model(src_padded, src_lengths, tgt_padded, teacher_force_ratio=0.0)

                batch_size, tgt_len_minus1, vocab_size = logits.shape
                logits_2d = logits.reshape(batch_size * tgt_len_minus1, vocab_size)
                targets_1d = tgt_padded[:, 1:].reshape(batch_size * tgt_len_minus1)

                loss = criterion(logits_2d, targets_1d)

                n_tokens = (targets_1d != pad_idx).sum().item()
                val_total_loss += loss.item() * n_tokens
                val_n_tokens += n_tokens

        val_loss = val_total_loss / max(val_n_tokens, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(0.0)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(0.0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch

            if checkpoint_filename is not None:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    checkpoint_filename,
                )

        if verbose:
            epoch_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
            )

    return history
