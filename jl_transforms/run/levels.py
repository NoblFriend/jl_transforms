from enum import Enum
from typing import Any


class Level(str, Enum):
    TRAIN = "train"
    EPOCH = "epoch"
    BATCH = "batch"


class LevelStatesMixin:
    train_state: dict[str, Any]
    epoch_state: dict[str, Any]
    batch_state: dict[str, Any]
    epoch_metrics: dict[str, Any]
    batch_metrics: dict[str, Any]
    state: dict[Level, dict[str, Any]]

    def __init__(self) -> None:
        metrics_key = "metrics"
        self.train_state = {}
        self.epoch_state = {metrics_key: {}}
        self.batch_state = {metrics_key: {}}
        self.epoch_metrics = self.epoch_state[metrics_key]
        self.batch_metrics = self.batch_state[metrics_key]
        self.state = {
            Level.TRAIN: self.train_state,
            Level.EPOCH: self.epoch_state,
            Level.BATCH: self.batch_state,
        }
