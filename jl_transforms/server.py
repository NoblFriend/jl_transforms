import copy

import torch
from catalyst import dl  # type: ignore[import]


def average_weights(weights_list):
    avg = copy.deepcopy(weights_list[0])
    for key in avg:
        for weight in weights_list[1:]:
            avg[key] += weight[key]
        avg[key] /= len(weights_list)
    return avg


class ServerRunner(dl.Runner):
    def __init__(self, model, loader, mode: str):
        super().__init__()
        self.model = model
        self.loader = loader
        self.mode = mode
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_model(self):  # noqa: WPS615
        return self.model.to(self.device)

    def get_loaders(self):
        return {self.mode: self.loader}

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.1)  # dummy

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def handle_batch(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
        self.batch = {"loss": loss, "logits": logits, "targets": targets}
