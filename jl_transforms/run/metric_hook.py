from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

if TYPE_CHECKING:
    from jl_transforms.run.runner import Runner

from jl_transforms.run.hooks import Hook


class MetricHook(Hook, ABC):
    def __init__(self, name: str):
        self.name = name

    def on_epoch_start(self, runner: "Runner"):
        self.reset()

    def on_batch_end(self, runner: "Runner"):
        metric_value = self.update(runner)
        if metric_value is not None:
            runner.batch_metrics[self.name] = metric_value

    def on_epoch_end(self, runner: "Runner"):
        runner.epoch_metrics[self.name] = self.compute()

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self, runner: "Runner"): ...

    @abstractmethod
    def compute(self): ...


class AccuracyHook(MetricHook):
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        input_key: str = "logits",
        target_key: str = "targets",
    ):
        name = f"acc@{top_k}"
        super().__init__(name=name)
        self.input_key = input_key
        self.target_key = target_key
        self.name = name

        self.metric = MulticlassAccuracy(num_classes=num_classes, top_k=top_k)

    def on_train_start(self, runner: "Runner"):
        self.metric.to(runner.device)

    def reset(self):
        self.metric.reset()

    def update(self, runner: "Runner"):
        logits = runner.batch_state[self.input_key]
        targets = runner.batch_state[self.target_key]
        return self.metric.update(logits, targets)

    def compute(self):
        return self.metric.compute().item()


class LossHook(MetricHook):
    def __init__(self, input_key: str = "loss"):
        super().__init__(name="loss")
        self.input_key = input_key
        self.metric = MeanMetric()

    def on_train_start(self, runner: "Runner"):
        self.metric.to(runner.device)

    def reset(self):
        self.metric.reset()

    def update(self, runner: "Runner"):
        loss = runner.batch_state[self.input_key]
        return self.metric.update(loss.detach())

    def compute(self):
        return self.metric.compute().item()
