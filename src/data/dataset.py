from __future__ import annotations
from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import Dataset

class CharacterDataset(Dataset, ABC) : 

    def __init__(self, data:Tensor, block_size: int) :
        self.data  = data 
        self.block_size = block_size

    def __len__(self) -> int :
        return self.data.size(0) - self.block_size

    @abstractmethod
    def __getitem__(self, index:int) -> tuple[Tensor, Tensor] :
        pass


class NextTokenDataset(CharacterDataset) :

    def __getitem__(self, index : int) -> tuple[Tensor, Tensor] :

        x = self.data[index : index +self.block_size]
        y = self.data[index + self.block_size]
        return x, y


class Sequencedataset(CharacterDataset) :

    def __getitem__(self, index:int) -> tuple[Tensor, Tensor] :
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]
        return x, y

