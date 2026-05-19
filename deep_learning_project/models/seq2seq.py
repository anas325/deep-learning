"""Encoder-Decoder Seq2Seq model with teacher forcing."""

import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import List, Tuple


class Encoder(nn.Module):
    """LSTM encoder. Reads source sequence, returns final hidden + cell states."""

    def __init__(
        self,
        src_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout_p: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout_p if num_layers > 1 else 0)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src: (batch, src_len)
        # Returns: (hidden, cell) each (num_layers, batch, hidden_dim)
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, cell) = self.rnn(packed)
        return hidden, cell


class Decoder(nn.Module):
    """LSTM decoder. One step at a time."""

    def __init__(
        self,
        tgt_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout_p: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout_p if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(
        self,
        tgt_token: torch.Tensor,       # (batch,)
        hidden: torch.Tensor,           # (num_layers, batch, hidden_dim)
        cell: torch.Tensor,             # (num_layers, batch, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: (logits: (batch, tgt_vocab_size), hidden, cell)
        tgt_token = tgt_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(tgt_token))  # (batch, 1, embed_dim)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        logits = self.fc_out(output.squeeze(1))  # (batch, tgt_vocab_size)
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model with teacher forcing.

    Wraps Encoder and Decoder, handles the decode loop.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        tgt_vocab_size: int,
        sos_idx: int = 2,
        eos_idx: int = 3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    def forward(
        self,
        src: torch.Tensor,               # (batch, src_len)
        src_lengths: torch.Tensor,       # (batch,)
        tgt: torch.Tensor,               # (batch, tgt_len) -- includes SOS
        teacher_force_ratio: float = 0.5,
    ) -> torch.Tensor:
        # Returns logits: (batch, tgt_len-1, tgt_vocab_size)
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        hidden, cell = self.encoder(src, src_lengths)

        # First input to decoder is <SOS>
        decoder_input = tgt[:, 0]  # (batch,)

        outputs = []
        for t in range(1, tgt_len):
            logits, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(logits.unsqueeze(1))  # (batch, 1, vocab)

            # Teacher forcing: feed ground truth or model prediction
            if random.random() < teacher_force_ratio:
                decoder_input = tgt[:, t]
            else:
                decoder_input = logits.argmax(dim=1)

        return torch.cat(outputs, dim=1)  # (batch, tgt_len-1, vocab)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,           # (batch, src_len)
        src_lengths: torch.Tensor,   # (batch,)
        max_len: int = 20,
    ) -> List[List[int]]:
        """
        Greedy decoding (no teacher forcing). Returns list of token index lists.
        Stops when <EOS> is predicted or max_len is reached.
        """
        self.eval()
        batch_size = src.size(0)
        hidden, cell = self.encoder(src, src_lengths)

        decoder_input = torch.full((batch_size,), self.sos_idx,
                                   dtype=torch.long, device=src.device)

        decoded = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            logits, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predicted = logits.argmax(dim=1)  # (batch,)

            for i in range(batch_size):
                if not finished[i]:
                    token = predicted[i].item()
                    if token == self.eos_idx:
                        finished[i] = True
                    else:
                        decoded[i].append(token)

            if all(finished):
                break

            decoder_input = predicted

        return decoded
