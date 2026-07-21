from abc import ABC, abstractmethod


class LanguageModel(ABC):
    """
    Base contract for all language models.
    """

    @abstractmethod
    def predict(self, context):
        """
        Predict the next-token distribution for the given context.
        """
        pass

    @abstractmethod
    def generate(self, context, max_new_tokens: int):
        """
        Generate a continuation from the given context.
        """
        pass