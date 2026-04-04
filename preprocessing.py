"""
Toxic Comment Classifier - NLP Preprocessing Pipeline
Tokenization, vocabulary building, padding, and text cleaning
"""

import re
import string
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
import joblib
import os


class TextPreprocessor:
    """
    Complete NLP preprocessing pipeline:
    clean → tokenize → build vocab → encode → pad/truncate
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_vocab: int = 50_000, max_len: int = 200, min_freq: int = 2):
        """
        Args:
            max_vocab: maximum vocabulary size (most frequent kept)
            max_len:   fixed sequence length (pad/truncate)
            min_freq:  minimum token frequency to enter vocabulary
        """
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.min_freq = min_freq

        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        self.vocab_size = 2
        self.fitted = False

    # ─── Cleaning ─────────────────────────────────────────────────────────────
    def clean(self, text: str) -> str:
        """Normalize raw text."""
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)        # remove URLs
        text = re.sub(r"<.*?>", " ", text)                          # strip HTML
        text = re.sub(r"[^\x00-\x7F]+", " ", text)                 # remove non-ASCII
        text = re.sub(r"\d+", " <NUM> ", text)                      # normalize numbers
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return self.clean(text).split()

    # ─── Vocabulary ───────────────────────────────────────────────────────────
    def fit(self, texts: List[str]):
        """Build vocabulary from a corpus."""
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        # Filter by frequency, truncate to max_vocab
        vocab_tokens = [
            tok for tok, freq in counter.most_common(self.max_vocab - 2)
            if freq >= self.min_freq
        ]
        for idx, tok in enumerate(vocab_tokens, start=2):
            self.word2idx[tok] = idx
            self.idx2word[idx] = tok

        self.vocab_size = len(self.word2idx)
        self.fitted = True
        print(f"Vocabulary built: {self.vocab_size} tokens (max_len={self.max_len})")
        return self

    # ─── Encoding ─────────────────────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        """Convert text → list of token ids (no padding)."""
        tokens = self.tokenize(text)
        return [self.word2idx.get(tok, self.UNK_IDX) for tok in tokens]

    def encode_and_pad(self, text: str) -> List[int]:
        """Encode, truncate to max_len, and pad with PAD_IDX."""
        ids = self.encode(text)[:self.max_len]
        ids += [self.PAD_IDX] * (self.max_len - len(ids))
        return ids

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts → numpy array (N, max_len)."""
        return np.array([self.encode_and_pad(t) for t in texts], dtype=np.int64)

    # ─── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "vocab_size": self.vocab_size,
            "max_vocab": self.max_vocab,
            "max_len": self.max_len,
            "min_freq": self.min_freq,
        }, os.path.join(path, "preprocessor.pkl"))

    def load(self, path: str):
        data = joblib.load(os.path.join(path, "preprocessor.pkl"))
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]
        self.vocab_size = data["vocab_size"]
        self.max_vocab = data["max_vocab"]
        self.max_len = data["max_len"]
        self.min_freq = data["min_freq"]
        self.fitted = True
        return self


# ─── Label Schema ─────────────────────────────────────────────────────────────
LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def binarize_labels(df, threshold: float = 0.5):
    """Convert label probabilities to binary (for multi-label tasks)."""
    return (df[LABEL_COLUMNS] >= threshold).astype(int)
