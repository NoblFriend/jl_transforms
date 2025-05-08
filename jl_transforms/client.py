import torch
from catalyst import dl  # type: ignore[import]
from torch import nn as nn


class ClientRunner(dl.Runner):
    def __init__(self, model, loader, global_weights, lambda_reg):
        super().__init__()
        self.model = model
        self.loader = loader
        self.global_weights = global_weights
        self.lambda_reg = lambda_reg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_model(self):  # noqa: WPS615
        return self.model.to(self.device)

    def get_loaders(self):
        return {"train": self.loader}

    def get_optimizer(self, model):
        return torch.optim.SGD(
            model.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"],
            nesterov=True,
        )

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def handle_batch(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(inputs)
        cross_entropy_loss = self.criterion(logits, targets)

        l2_penalty = self._compute_l2_penalty()

        self.batch = {
            "loss": cross_entropy_loss + self.lambda_reg * l2_penalty,
            "logits": logits,
            "targets": targets,
        }

    def _compute_l2_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0, dtype=torch.float32, device=self.device)
        for name, parameter in self.model.named_parameters():
            global_param = self.global_weights[name].to(self.device)
            penalty += (parameter - global_param).pow(2).sum()
        return penalty


def get_epoch_offset(round_idx, local_epoch_fn):
    return sum(local_epoch_fn(round_idx) for round_idx in range(round_idx))
