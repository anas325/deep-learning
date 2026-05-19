"""Breast Cancer dataset loading, preprocessing, and DataLoader creation."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class BreastCancerDataset(Dataset):
    """PyTorch Dataset wrapping breast cancer features and labels."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        # X: float32 (N, 30), y: long (N,)
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_breast_cancer_loaders(
    val_size: float = 0.15,
    test_size: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, int, int]:
    """
    Loads, normalizes, splits, and wraps the breast cancer dataset.

    Returns:
        train_loader, val_loader, test_loader,
        scaler (fitted on train only),
        n_features (30),
        n_classes (2)

    Pipeline:
      1. load_breast_cancer() -> X (569, 30), y (569,)
      2. train_test_split stratified by y (test_size)
      3. train_test_split remainder stratified by y (val / (1 - test_size))
      4. StandardScaler fit on train only, transform all three splits
      5. Convert to float32 (X) and long (y) tensors
      6. Wrap in BreastCancerDataset -> DataLoader(shuffle=True for train only)
    """
    # 1. Load raw data
    data = load_breast_cancer()
    X: np.ndarray = data.data          # (569, 30)
    y: np.ndarray = data.target        # (569,)
    n_features: int = X.shape[1]       # 30
    n_classes: int = len(np.unique(y)) # 2

    # 2. Split off test set (stratified)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    # 3. Split remainder into train / val (stratified)
    # val fraction relative to the train+val pool
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val_size,
        stratify=y_trainval,
        random_state=seed,
    )

    # 4. Normalise: fit scaler on train only, transform all splits
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # 5. Convert to tensors
    def to_tensors(
        X_arr: np.ndarray, y_arr: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.long)
        return X_t, y_t

    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_val_t,   y_val_t   = to_tensors(X_val,   y_val)
    X_test_t,  y_test_t  = to_tensors(X_test,  y_test)

    # 6. Build Datasets and DataLoaders
    train_dataset = BreastCancerDataset(X_train_t, y_train_t)
    val_dataset   = BreastCancerDataset(X_val_t,   y_val_t)
    test_dataset  = BreastCancerDataset(X_test_t,  y_test_t)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, scaler, n_features, n_classes
