import torch
import torch.nn as nn

from ..data.datamodule import CharacterDataModule


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        datamodule: CharacterDataModule,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self._model = model.to(device)
        self._datamodule = datamodule
        self._optimizer = optimizer
        self._device = device

    def fit(self, epochs: int) -> None:

        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    def train_one_epoch(self) -> float:
        self._model.train()

        losses = []

        train_loader = self._datamodule.train_dataloader()

        for inputs, targets in train_loader:

            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            self._optimizer.zero_grad()
            _, loss = self._model(inputs, targets)
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())

        return sum(losses) / len(losses)

    @torch.no_grad()
    def validate(self) -> float:
        self._model.eval()

        losses = []
        val_loader = self._datamodule.val_dataloader()

        for inputs, targets in val_loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            _, loss = self._model(inputs, targets)
            losses.append(loss.item())

        return sum(losses) / len(losses)