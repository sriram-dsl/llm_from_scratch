import torch


class StatisticalBigramModel:
    '''
    A simple class to predict the next token with the probability 
    '''

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.counts = torch.zeros(
            (vocab_size, vocab_size),
            dtype=torch.long,
        )
        self.transition_probs = None

    def fit(self, dataloader) -> None:
        for batch_x, batch_y in dataloader :
            for x, y in zip(batch_x, batch_y) :
                self.counts[x, y] += 1
        self._compute_transition_probability()

    def _compute_transition_probability(self) -> None : 
        counts = self.counts.float()
        row_sums = counts.sum(dim=1, keepdim=True)
        self.transition_probs = counts / row_sums.clamp(min=1)  # Avoid division by zero
 
    def predict(self, token: int) -> int:
        if self.transition_probs is None:
            raise RuntimeError(
                "The model has not been fitted yet. Call fit() first."
            )
        probs = self.transition_probs[token]
        if probs.sum() == 0:
            return torch.randint(self.vocab_size, (1,)).item()
        return torch.multinomial(probs, num_samples=1).item()

    def generate(self, token: int, tokenizer, max_length: int = 100) -> str:

        generated_tokens = [token] 
        for _ in range(max_length - 1) :
            next_token = self.predict(generated_tokens[-1])
            generated_tokens.append(next_token)

        return tokenizer.decode(generated_tokens)
        