import wandb
from catalyst.core.callback import Callback  # type: ignore[import]
from catalyst.core.callback import CallbackOrder


class FederatedLoggerCallback(Callback):
    def __init__(
        self,
        namespace: str,
        mode: str,
        epoch: int,
        round_idx: int | None = None,
        log_round: bool = False,
    ):
        super().__init__(order=CallbackOrder.logging + 1)
        self.namespace = namespace
        self.mode = mode
        self.epoch = epoch
        self.round_idx = round_idx
        self.log_round = log_round

    def on_loader_end(self, runner):
        metrics = runner.loader_metrics

        wandb.log(
            {
                f"{self.namespace}/epoch/{self.mode}/{metric}": metric_val
                for metric, metric_val in metrics.items()
            },
            step=self.epoch,
        )

        if self.log_round and self.round_idx is not None:
            wandb.log(
                {
                    f"{self.namespace}/round/{self.mode}/{metric}": metric_val
                    for metric, metric_val in metrics.items()
                },
                step=self.round_idx,
            )
