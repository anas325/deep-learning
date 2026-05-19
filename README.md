# Deep Learning Project — MLP, CNN, RNN/LSTM/GRU/Seq2Seq

A three-part PyTorch project that walks through the most important neural-network families from first principles, with clean, reusable utilities for reproducibility, I/O, training, and evaluation.

---

## Description

This project is organized into three self-contained learning modules, each delivered as a Jupyter notebook backed by importable Python source code:

| Part | Topic | Dataset |
|------|-------|---------|
| 1 | **Tabular classification with MLP** | sklearn Breast Cancer (binary) |
| 2 | **Image classification with CNN** | CIFAR-10 (10-class) |
| 3 | **Sequence modeling with RNN / LSTM / GRU / Seq2Seq** | Synthetic sentiment + number-translation |

Each part demonstrates data loading, model construction, training loops with validation, evaluation metrics, and result persistence.

---

## Project Structure

```
deep-learning/
├── requirements.txt
├── setup.py
├── README.md
│
├── notebooks/
│   ├── 01_mlp_breast_cancer.ipynb       # Part 1 — MLP on tabular data
│   ├── 02_cnn_cifar10.ipynb             # Part 2 — CNN on CIFAR-10
│   └── 03_rnn_lstm_gru_seq2seq.ipynb    # Part 3 — RNN / LSTM / GRU / Seq2Seq
│
├── outputs/
│   ├── figures/                         # Saved matplotlib figures (.png)
│   ├── metrics/                         # Saved JSON / CSV metric files
│   └── models/                          # Saved model checkpoints (.pt)
│
└── deep_learning_project/
    ├── __init__.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── tabular.py                   # BreastCancerDataset, get_breast_cancer_loaders
    │   └── text.py                      # Vocabulary, sentiment & seq2seq loaders
    │
    ├── models/
    │   ├── __init__.py
    │   ├── mlp.py                       # MLP, build_sequential_mlp, init strategies
    │   ├── cnn.py                       # LeNetCNN, ConvBlock (PyTorch nn.Module)
    │   ├── cnn_manual.py                # Pure-NumPy convolution / pooling reference
    │   ├── rnn.py                       # VanillaRNN, LSTMModel, GRUModel
    │   └── seq2seq.py                   # Encoder, Decoder, Seq2Seq
    │
    ├── training/
    │   ├── __init__.py
    │   └── train_mlp.py                 # train_one_epoch, evaluate, train, train_seq2seq
    │
    ├── evaluation/
    │   ├── __init__.py
    │   └── classification.py            # Metrics + confusion matrix + history plots
    │
    └── utils/
        ├── __init__.py
        ├── reproducibility.py           # set_seed, get_device
        └── io.py                        # save/load helpers anchored to outputs/
```

---

## Installation

Clone the repository and install the package in editable mode so that all notebooks can import `deep_learning_project` directly:

```bash
pip install -e .
```

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

> **PyTorch with CUDA** — if you need GPU support, install PyTorch separately following the official instructions at <https://pytorch.org/get-started/locally/> before running the command above.

---

## Running the Notebooks

Launch Jupyter from the project root:

```bash
jupyter notebook
# or
jupyter lab
```

Then open the desired notebook from the `notebooks/` directory:

### Part 1 — MLP on Breast Cancer (tabular)

```
notebooks/01_mlp_breast_cancer.ipynb
```

Covers: dataset loading with `sklearn`, custom `Dataset`/`DataLoader`, fully-connected MLP architectures, weight initialization strategies (Xavier, normal, constant), training with cross-entropy loss, and evaluation with accuracy / precision / recall / F1 / confusion matrix.

### Part 2 — CNN on CIFAR-10 (images)

```
notebooks/02_cnn_cifar10.ipynb
```

Covers: `torchvision` CIFAR-10 download and augmentation, manual 2-D cross-correlation and pooling (NumPy reference), `LeNetCNN` built from `ConvBlock` modules, per-class accuracy, and training-history plots.

### Part 3 — RNN / LSTM / GRU / Seq2Seq (sequences)

```
notebooks/03_rnn_lstm_gru_seq2seq.ipynb
```

Covers: synthetic sentiment classification (positive / negative sentences), a custom `Vocabulary` with padding, `VanillaRNN` vs `LSTMModel` vs `GRUModel` comparison, and an encoder-decoder `Seq2Seq` model trained on a number-translation task.

---

## Datasets

| Dataset | Source | Task | Size |
|---------|--------|------|------|
| **Breast Cancer Wisconsin** | `sklearn.datasets.load_breast_cancer` | Binary classification (malignant / benign) | 569 samples, 30 features |
| **CIFAR-10** | `torchvision.datasets.CIFAR10` (auto-downloaded) | 10-class image classification | 50 000 train / 10 000 test, 32×32 RGB |
| **Synthetic Sentiment** | Generated in `deep_learning_project/data/text.py` | Binary sentiment classification | Configurable, default 2 000 samples |
| **Synthetic Number Translation** | Generated in `deep_learning_project/data/text.py` | Seq2seq (digit string → word string) | Configurable, default 2 000 pairs |

---

## Requirements

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `torch` | 2.1.0 | Core deep-learning framework |
| `torchvision` | 0.16.0 | CIFAR-10 dataset + transforms |
| `scikit-learn` | 1.3.0 | Breast Cancer dataset, train/test split |
| `numpy` | 1.24.0 | Numerical operations, manual convolution |
| `matplotlib` | 3.7.0 | Plotting training curves and figures |
| `seaborn` | 0.12.0 | Styled confusion-matrix heatmaps |
| `pandas` | 2.0.0 | CSV metric export and display |
| `jupyter` | 1.0.0 | Notebook server |
| `ipykernel` | 6.0.0 | Jupyter Python kernel |
| `tqdm` | 4.65.0 | Progress bars in training loops |
