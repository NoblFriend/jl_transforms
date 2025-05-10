from typing import TYPE_CHECKING

import wandb
from jl_transforms.run.hooks import Hook

if TYPE_CHECKING:
    from jl_transforms.run.runner import Runner


class TqdmLoggerHook(Hook):
    def on_batch_end(self, runner: "Runner"):
        metrics = runner.batch_metrics
        metrics["loss"] = runner.batch_state["loss"].item()
        pbar = runner.epoch_state["pbar"]
        pbar.set_postfix(
            {name: f"{metric:.4f}" for name, metric in metrics.items()}
        )

    def on_epoch_end(self, runner: "Runner"):
        metrics = runner.epoch_metrics
        pbar = runner.train_state["pbar"]
        pbar.set_postfix(
            {name: f"{metric:.4f}" for name, metric in metrics.items()}
        )


class WandbLoggerHook(Hook):
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        if self.prefix != "":
            self.prefix = f"{self.prefix}/"

    def on_epoch_end(self, runner: "Runner"):
        metrics = runner.epoch_metrics
        if not metrics:
            return
        full_prefix = f"{self.prefix}{runner.epoch_state['mode']}"
        logged = {
            f"{full_prefix}/{name}" if self.prefix else name: metric
            for name, metric in metrics.items()
        }

        wandb.log(logged, step=runner.epoch)
