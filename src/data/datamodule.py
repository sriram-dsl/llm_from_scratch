from __future__ import annotations

from pathlib import Path
from typing import Type

import torch

from .tokenizer import CharacterTokenizer
from .dataset import CharacterDataset
from .dataloader import MiniDataLoader


class CharacterDataModule:

    def __init__(
        self,
        file_path: str,
        dataset_cls: Type[CharacterDataset],
        block_size: int,
        batch_size: int,
        train_split: float = 0.9,
        shuffle: bool = True,
    ) -> None:

        self.file_path = Path(file_path)
        self.datasetcls = dataset_cls
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.shuffle = shuffle

        self._setup()

    def _setup(self) -> None:
 
        text = self.file_path.read_text(encoding="utf-8")
        self.tokenizer = CharacterTokenizer.from_text(text)
        encoded = self.tokenizer.encode(text)

        self.data = torch.tensor(
            encoded,
            dtype=torch.long,
        )
        split_idx = int(len(self.data) * self.train_split)

        train_tokens = self.data[:split_idx]
        val_tokens = self.data[split_idx:]

        self.train_dataset = self.datasetcls(
            train_tokens,
            self.block_size,
        )

        self.val_dataset = self.datasetcls(
            val_tokens,
            self.block_size,
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def train_dataloader(self) -> MiniDataLoader:

        return MiniDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> MiniDataLoader:

        return MiniDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )