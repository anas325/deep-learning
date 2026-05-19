"""
Self-contained NLP data pipeline for sentiment classification and seq2seq.
No external downloads required.
"""

import random
import re
from collections import Counter
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Token-to-index mapping with special tokens PAD=0, UNK=1, SOS=2, EOS=3."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self) -> None:
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        for tok in [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
            idx = len(self.token2idx)
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

    def build_from_texts(self, texts: List[str], min_freq: int = 1) -> None:
        """Tokenizes all texts, adds tokens appearing >= min_freq times."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        for token, freq in sorted(counter.items(), key=lambda x: -x[1]):
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def tokenize(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def encode(
        self, text: str, add_sos: bool = False, add_eos: bool = False
    ) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token2idx.get(t, self.token2idx[self.UNK_TOKEN]) for t in tokens]
        if add_sos:
            ids = [self.token2idx[self.SOS_TOKEN]] + ids
        if add_eos:
            ids = ids + [self.token2idx[self.EOS_TOKEN]]
        return ids

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        special = {
            self.token2idx[t]
            for t in [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        }
        tokens = []
        for idx in indices:
            if skip_special and idx in special:
                continue
            tokens.append(self.idx2token.get(idx, self.UNK_TOKEN))
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------

def make_sentiment_dataset(
    n_samples: int = 1000, seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Generates a synthetic sentiment dataset with balanced classes.

    Returns:
        texts  – list of n_samples strings
        labels – list of n_samples ints (0 = negative, 1 = positive)
    """
    rng = random.Random(seed)

    positive_words = [
        "excellent", "amazing", "wonderful", "fantastic", "great",
        "outstanding", "brilliant", "superb", "perfect", "magnificent",
        "delightful", "impressive", "exceptional", "remarkable", "splendid",
        "joyful", "beautiful", "marvelous", "incredible", "positive",
        "good", "nice", "happy", "love", "best",
    ]
    negative_words = [
        "terrible", "awful", "horrible", "dreadful", "disgusting",
        "pathetic", "miserable", "disappointing", "poor", "bad",
        "worst", "hate", "ugly", "boring", "useless",
        "dull", "mediocre", "inferior", "frustrating", "unpleasant",
        "negative", "annoying", "stupid", "sad", "failure",
    ]
    neutral_words = [
        "the", "a", "is", "was", "this", "that", "very", "quite",
        "really", "movie", "product", "service", "experience", "time",
        "day", "place", "thing", "person", "book", "film",
    ]

    texts: List[str] = []
    labels: List[int] = []

    n_per_class = n_samples // 2

    for label in [0, 1]:
        for _ in range(n_per_class):
            length = rng.randint(8, 15)
            words: List[str] = []
            for _ in range(length):
                roll = rng.random()
                if label == 1:
                    # ~70 % positive, ~30 % from neutral + negative pool
                    if roll < 0.70:
                        words.append(rng.choice(positive_words))
                    elif roll < 0.85:
                        words.append(rng.choice(neutral_words))
                    else:
                        words.append(rng.choice(negative_words))
                else:
                    # ~70 % negative, ~30 % from neutral + positive pool
                    if roll < 0.70:
                        words.append(rng.choice(negative_words))
                    elif roll < 0.85:
                        words.append(rng.choice(neutral_words))
                    else:
                        words.append(rng.choice(positive_words))
            texts.append(" ".join(words))
            labels.append(label)

    # Shuffle together so classes are interleaved
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def make_number_translation_dataset() -> Tuple[List[str], List[str]]:
    """
    Returns source/target pairs for numbers 0-99.

    Returns:
        sources – ["0", "1", ..., "99"]
        targets – ["zero", "one", ..., "ninety nine"]
    """
    ones = [
        "", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = [
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety",
    ]

    sources: List[str] = []
    targets: List[str] = []

    for n in range(100):
        sources.append(str(n))
        if n == 0:
            word = "zero"
        elif n < 20:
            word = ones[n]
        else:
            t = tens[n // 10]
            o = ones[n % 10]
            word = (t + " " + o).strip() if o else t
        targets.append(word)

    return sources, targets


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    """Maps raw texts to encoded integer sequences with a class label."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_len: Optional[int] = None,
    ) -> None:
        self.token_ids: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        for text, label in zip(texts, labels):
            ids = vocab.encode(text)
            if max_len is not None:
                ids = ids[:max_len]
            # Guard against empty sequences after truncation
            if len(ids) == 0:
                ids = [vocab.token2idx[vocab.UNK_TOKEN]]
            self.token_ids.append(torch.tensor(ids, dtype=torch.long))
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[idx], self.labels[idx]


class Seq2SeqDataset(Dataset):
    """Source/target pairs where targets are wrapped with SOS and EOS tokens."""

    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
    ) -> None:
        self.src_ids: List[torch.Tensor] = []
        self.tgt_ids: List[torch.Tensor] = []

        for src, tgt in zip(sources, targets):
            src_encoded = src_vocab.encode(src)
            if len(src_encoded) == 0:
                src_encoded = [src_vocab.token2idx[src_vocab.UNK_TOKEN]]
            self.src_ids.append(torch.tensor(src_encoded, dtype=torch.long))

            tgt_encoded = tgt_vocab.encode(tgt, add_sos=True, add_eos=True)
            self.tgt_ids.append(torch.tensor(tgt_encoded, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src_ids[idx], self.tgt_ids[idx]


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_classification(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate for TextClassificationDataset.

    Sorts by sequence length descending (required for pack_padded_sequence),
    pads all sequences to the max length in the batch.

    Returns:
        padded_seqs : (B, max_len)  – LongTensor
        lengths     : (B,)          – LongTensor of original lengths
        labels      : (B,)          – LongTensor
    """
    # Sort by descending length
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    seqs, labels = zip(*batch)

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    # pad_sequence expects (L, B) by default; batch_first=True gives (B, L)
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    labels_t = torch.stack(labels)

    return padded_seqs, lengths, labels_t


def collate_seq2seq(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate for Seq2SeqDataset.

    Returns:
        src_padded  : (B, src_max)  – LongTensor
        src_lengths : (B,)          – LongTensor
        tgt_padded  : (B, tgt_max)  – LongTensor
        tgt_lengths : (B,)          – LongTensor
    """
    src_seqs, tgt_seqs = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)

    return src_padded, src_lengths, tgt_padded, tgt_lengths


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def get_sentiment_loaders(
    n_samples: int = 1000,
    val_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
    min_vocab_freq: int = 1,
) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Full pipeline for sentiment classification.

    1. make_sentiment_dataset
    2. Stratified train / val split
    3. Vocabulary built on train texts only
    4. TextClassificationDataset for each split
    5. DataLoaders with collate_classification

    Returns:
        train_loader, val_loader, vocab
    """
    texts, labels = make_sentiment_dataset(n_samples=n_samples, seed=seed)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=val_size,
        stratify=labels,
        random_state=seed,
    )

    vocab = Vocabulary()
    vocab.build_from_texts(train_texts, min_freq=min_vocab_freq)

    train_dataset = TextClassificationDataset(train_texts, train_labels, vocab)
    val_dataset   = TextClassificationDataset(val_texts,   val_labels,   vocab)

    collate_fn = partial(collate_classification, pad_idx=vocab.token2idx[Vocabulary.PAD_TOKEN])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, vocab


def get_seq2seq_loaders(
    batch_size: int = 16,
    val_size: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Full pipeline for number translation (seq2seq).

    1. make_number_translation_dataset
    2. Train / val split
    3. src_vocab built on train sources; tgt_vocab built on train targets
    4. Seq2SeqDataset for each split
    5. DataLoaders with collate_seq2seq

    Returns:
        train_loader, val_loader, src_vocab, tgt_vocab
    """
    sources, targets = make_number_translation_dataset()

    train_src, val_src, train_tgt, val_tgt = train_test_split(
        sources, targets,
        test_size=val_size,
        random_state=seed,
    )

    src_vocab = Vocabulary()
    src_vocab.build_from_texts(train_src)

    tgt_vocab = Vocabulary()
    tgt_vocab.build_from_texts(train_tgt)

    train_dataset = Seq2SeqDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset   = Seq2SeqDataset(val_src,   val_tgt,   src_vocab, tgt_vocab)

    collate_fn = partial(
        collate_seq2seq,
        pad_idx=src_vocab.token2idx[Vocabulary.PAD_TOKEN],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, src_vocab, tgt_vocab
