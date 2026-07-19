from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset

class CharacterDataset(Dataset) : 

    def __init__(self, data:Tensor, block_size: int) :
        self.data  = data 
        self.block_data = block_size

    def __len__(self) -> int :
        return self.data.size(0) - self.block_size
    
    def __getitem__(self, index:int) -> tuple[Tensor, Tensor] :
        x = self.data[index: index + self.block_size]
        y = self.data[index + 1: index + 1 + self.block_size]

        return x, y