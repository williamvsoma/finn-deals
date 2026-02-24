from typing import Iterable, List
import numpy as np

class WordPieceTokenizer:
    
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}

    def fit(self, vocab: Iterable[str]):
        """
        Build word → id and id → word lookup tables.
        """
        vocab = sorted(set(vocab))  # deterministic ordering
        vocab = [self.pad_token, self.unk_token] + vocab  # add unknown token first

        self.encoder = {word: idx for idx, word in enumerate(vocab)}
        self.decoder = {idx: word for idx, word in enumerate(vocab)}
        return self

    def _tokenize(self, x: Iterable[str]) -> List[List[str]]:
        """
        Lowercase and split each string in x.
        """
        return [s.lower().split() for s in x]

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
    
    def __len__(self,):
        """
        Vocab size.
        """
        return len(self.encoder)

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
        assert (self.max is not None) and (self.min is not None), "You need to fit the scaler first!"
        x = np.log1p(x)
        return (x - self.min) / (self.max - self.min)

    def fit_transform(self, x):
        _ = self.fit(x)
        return self.transform(x)

    def untransform(self, x):
        x = x*(self.max - self.min) + self.min
        return np.expm1(x)