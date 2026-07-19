from __future__ import annotations
from dataclass import dataclass

@dataclass
class CharacterTokensizer:
    stoi : dict[str, int]
    itos : dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer" :
        vocab = sorted(set(text)) 

        stoi = {c:i for i, c in enumerate(vocab)}
        itos = {i:c for i, c in enumerate(vocab)}
        return cls(stoi=stoi, itos=itos)
    
    @property
    def vocab_size(self) -> int :
        return len(self.stoi)
    
    def encode(self, text: str) -> int :
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens: list[int]) -> str :
        return "".join([self.itos[idx] for idx in tokens])
    
    