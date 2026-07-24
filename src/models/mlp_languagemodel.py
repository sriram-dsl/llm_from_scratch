from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.lm import LanguageModel


class MLPModel(nn.Module, LanguageModel):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        block_size: int,
    ) -> None:

        super().__init__()

        self._block_size = block_size
        self._embedding = nn.Embedding(vocab_size, embedding_dim)
        self._hidden = nn.Linear(block_size * embedding_dim, hidden_dim)
        self._output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        x = self._embedding(inputs)
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self._hidden(x))
        logits = self._output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:

        logits, _ = self.forward(context)
        probs = F.softmax(logits,dim=-1)
        return probs

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            context_window = context[:, -self._block_size:]
            probs = self.predict(context_window)
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            context = torch.cat((context, next_token), dim=1)

        return context