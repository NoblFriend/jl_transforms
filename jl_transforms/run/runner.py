from typing import Literal

import torch
from tqdm import tqdm  # type: ignore[import]

from jl_transforms.run.hooks import Hook, run_hook
from jl_transforms.run.internal_hooks import eval_hooks, train_hooks
from jl_transforms.run.levels import LevelStatesMixin

Mode = Literal["train", "val", "test"]


class Runner(LevelStatesMixin):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    loaders: dict[Mode, torch.utils.data.DataLoader]
    device: torch.device
    epoch: int
    _internal_hooks: list[Hook]
    hooks: list[Hook]
    name: str

    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self._internal_hooks = []
        self.hooks = []
        self.name = "dummy_name"

    def train(self, num_epochs: int) -> None:
        self._internal_hooks = train_hooks
        self.model.train()
        self._run_train(num_epochs)
        self._internal_hooks = []

    def evaluate(self, mode: Mode) -> None:
        self._internal_hooks = eval_hooks
        self.model.eval()
        with torch.no_grad():
            self._run_epoch(mode)
        self._internal_hooks = []

    @run_hook
    def _run_train(self, num_epochs: int) -> None:
        pbar = tqdm(
            range(num_epochs),
            desc=f"({self.name}) Train",
            unit="epoch",
            leave=True,
        )
        self.train_state.update(
            pbar=pbar,
        )
        for _ in pbar:
            self._run_epoch("train")
            self.epoch += 1

    @run_hook
    def _run_epoch(self, mode: Mode) -> None:
        pbar = tqdm(
            self.loaders[mode],
            desc=f"({self.name}) {mode.title()} Epoch",
            unit="batch",
            leave=False,
        )
        self.epoch_state.update(
            mode=mode,
            pbar=pbar,
        )
        for batch in pbar:
            self.batch_state.update(
                raw=batch,
            )
            self._run_batch()

    @run_hook
    def _run_batch(self) -> None:
        self.batch_state.update(logits=self.model(self.batch_state["inputs"]))
