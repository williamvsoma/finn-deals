import re
from typing import Iterable, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset


class WordPieceTokenizer:
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}

    @staticmethod
    def _normalize(text: object) -> str:
        if text is None:
            return ""
        if isinstance(text, float) and np.isnan(text):
            return ""
        normalized = re.sub(r"[_\W]+", " ", str(text).lower())
        return normalized.strip()

    def _tokenize_one(self, text: object) -> List[str]:
        normalized = self._normalize(text)
        return normalized.split() if normalized else []

    def fit(self, vocab: Iterable[str]):
        """
        Build word → id and id → word lookup tables.
        """
        tokens = [token for text in vocab for token in self._tokenize_one(text)]
        vocab = sorted(set(tokens))  # deterministic ordering
        vocab = [self.pad_token, self.unk_token] + vocab

        self.encoder = {word: idx for idx, word in enumerate(vocab)}
        self.decoder = {idx: word for idx, word in enumerate(vocab)}
        return self

    def _tokenize(self, x: Iterable[str]) -> List[List[str]]:
        """
        Lowercase and split each string in x.
        """
        return [self._tokenize_one(s) for s in x]

    def __call__(self, x: Iterable[str]) -> List[List[int]]:
        """
        Convert text → token_ids.
        """
        tokenized = self._tokenize(x)
        return [
            [self.encoder.get(word, self.encoder[self.unk_token]) for word in sentence]
            for sentence in tokenized
        ]

    @property
    def pad_token_id(self) -> int:
        return self.encoder[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.encoder[self.unk_token]

    def __len__(self):
        """
        Vocab size.
        """
        return len(self.encoder)


class EmbeddingEncoder:
    def __init__(self, unk_token: str = "<UNK>", missing_token: str = "<MISSING>"):
        self.unk_token = unk_token
        self.missing_token = missing_token
        self.unk_idx = 0
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}

    def _key(self, item: object) -> str:
        if item is None:
            return self.missing_token
        if isinstance(item, float) and np.isnan(item):
            return self.missing_token
        return str(item)

    def fit(self, numeric_col: Iterable[object]):

        vocab = sorted({self._key(item) for item in numeric_col})

        self.encoder = {self.unk_token: self.unk_idx}
        self.encoder.update({word: idx for idx, word in enumerate(vocab, start=1)})
        self.decoder = {idx: word for word, idx in self.encoder.items()}

        return self

    def __call__(self, x: Iterable[object]) -> np.ndarray:
        return np.array([self.encoder.get(self._key(item), self.unk_idx) for item in x])


class Log1pMinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        x = np.log1p(x)
        self.min = np.min(x)
        self.max = np.max(x)

        if self.max == self.min:
            self.max += 1e-9

        return self

    def transform(self, x):
        assert (self.max is not None) and (self.min is not None), (
            "You need to fit the scaler first!"
        )
        x = np.log1p(x)
        return (x - self.min) / (self.max - self.min)

    def fit_transform(self, x):
        _ = self.fit(x)
        return self.transform(x)

    def untransform(self, x):
        x = x * (self.max - self.min) + self.min
        return np.expm1(x)


class DealDataset(Dataset):
    """
    PyTorch Dataset for DealPricingModel.

    Holds pre-padded text ids, tabular features, categorical high indices, and targets.
    """

    def __init__(
        self,
        text_ids: torch.Tensor,
        tabular: torch.Tensor,
        cat_high: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.text_ids = text_ids
        self.tabular = tabular
        self.cat_high = cat_high
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            self.text_ids[idx],
            self.tabular[idx],
            self.cat_high[idx],
            self.targets[idx],
        )
