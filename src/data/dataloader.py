from __future__ import annotations

import random
import torch
from torch import Tensor

class MiniDataLoader :

    def __init__(
            self,
            dataset,
            batch_size: int,
            shuffle: bool = True,
            drop_last: bool = False,
    ) -> None :
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) :
        self.indices = list(range(len(self.dataset)))
        if self.shuffle : 
            random.shuffle(self.indices)

        self.current_position = 0

        return self

    def __next__(self)  -> tuple[Tensor, Tensor] :

        if self.current_position >= len(self.indices) :
            raise StopIteration
        
        end_position = self.current_position + self.batch_size
        batch_indices = self.indices[self.current_position :  end_position]

        if self.drop_last and len(batch_indices) < self.batch_size :
            raise StopIteration
        
        batch_x = []
        batch_y = []

        for index in batch_indices : 
            x, y = self.dataset[index]
            batch_x.append(x)
            batch_y.append(y)

        batch_x = torch.stack(batch_x)
        batch_y = torch.stack(batch_y)

        self.current_position = end_position

        return batch_x, batch_y