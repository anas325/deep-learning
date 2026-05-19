"""RNN, LSTM, and GRU classifiers for text sequence classification."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple


class BaseRNNClassifier(nn.Module):
    """
    Abstract base for RNN-based text classifiers.
    Subclasses implement _run_rnn() which extracts a context vector from the sequence.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout_p: float = 0.3,
        bidirectional: bool = False,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len), lengths: (batch,)
        embedded = self.dropout(self.embedding(x))
        context = self._run_rnn(embedded, lengths)
        return self.classifier(self.dropout(context))

    def _run_rnn(self, embedded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VanillaRNN(BaseRNNClassifier):
    """Uses nn.RNN. Context = last hidden state."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes,
                 dropout_p=0.3, bidirectional=False, pad_idx=0):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, num_classes,
                         dropout_p, bidirectional, pad_idx)
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_p if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def _run_rnn(self, embedded, lengths):
        # Use pack_padded_sequence for efficiency (sequences sorted by length desc)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.rnn(packed)
        # hidden: (num_layers * num_directions, batch, hidden_dim)
        # Take the last layer's hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            context = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            context = hidden[-1]
        return context


class LSTMModel(BaseRNNClassifier):
    """Uses nn.LSTM. Context = final hidden state (not cell state)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes,
                 dropout_p=0.3, bidirectional=False, pad_idx=0):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, num_classes,
                         dropout_p, bidirectional, pad_idx)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_p if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def _run_rnn(self, embedded, lengths):
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, _cell) = self.rnn(packed)
        if self.bidirectional:
            context = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            context = hidden[-1]
        return context


class GRUModel(BaseRNNClassifier):
    """Uses nn.GRU. Context = final hidden state."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes,
                 dropout_p=0.3, bidirectional=False, pad_idx=0):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, num_classes,
                         dropout_p, bidirectional, pad_idx)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_p if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def _run_rnn(self, embedded, lengths):
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.rnn(packed)
        if self.bidirectional:
            context = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            context = hidden[-1]
        return context
