import torch
import torch.nn as nn
from models.lm import LanguageModel
from ..losses.loss import CrossEntropyLoss
from ..data.tokenizer import CharacterTokenizer
import torch.nn.functional as F


class NeuralBigramModel(nn.Module, LanguageModel):

    def __init__(self, vocab_size):
        super().__init__()
        self._vocab_size = vocab_size
        self._embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        logits = self._embedding(inputs)
        loss = None
        if targets is not None: 
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def predict(self, context):
        logits , _ = self.forward(context)
        probs = F.softmax(logits, dim = -1)
        return probs
    @torch.no_grad()
    def generate(self, context, max_new_tokens):
    
        for i in range(max_new_tokens) : 
            probs = self.predict(context)
            next_token = torch.multinomial(probs[:, -1, :], num_samples=1)
            context = torch.cat((context, next_token), dim=1)

        return context

