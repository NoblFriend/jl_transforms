import hydra
import wandb
from catalyst import dl  # type: ignore[import]
from omegaconf import DictConfig, OmegaConf

from jl_transforms.callbacks import FederatedLoggerCallback
from jl_transforms.client import ClientRunner, get_epoch_offset
from jl_transforms.data.data import get_client_loaders
from jl_transforms.misc import local_epoch_fn
from jl_transforms.models import get_resnet18_cifar
from jl_transforms.server import ServerRunner, average_weights


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):  # noqa: WPS210
    wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )

    client_loaders, test_loader = get_client_loaders(cfg)
    global_model = get_resnet18_cifar()
    global_weights = global_model.state_dict()

    for round_idx in range(cfg.num_rounds):
        local_epochs = local_epoch_fn(round_idx)
        epoch_offset = get_epoch_offset(round_idx, local_epoch_fn)
        client_weights = []

        for client_idx, loader in enumerate(client_loaders):
            model = get_resnet18_cifar()
            model.load_state_dict(global_weights)
            runner = ClientRunner(
                model, loader, global_weights, cfg.lambda_reg
            )
            runner.train(
                num_epochs=local_epochs,
                logdir=None,
                callbacks=[
                    FederatedLoggerCallback(
                        f"client_{client_idx}", "train", epoch_offset + epoch
                    )
                    for epoch in range(local_epochs)
                ]
                + [
                    dl.AccuracyCallback(),
                    dl.CriterionCallback(),
                ],
                hparams={"lr": cfg.lr, "momentum": cfg.momentum},
            )
            client_weights.append(model.state_dict())

        global_weights = average_weights(client_weights)
        global_model.load_state_dict(global_weights)

        for mode, loader in (
            ("train", client_loaders[0]),
            ("test", test_loader),
        ):
            runner = ServerRunner(global_model, loader, mode)
            runner.evaluate(
                callbacks=[
                    FederatedLoggerCallback(
                        namespace="server",
                        mode=mode,
                        epoch=epoch_offset + local_epochs - 1,
                        round_idx=round_idx,
                        log_round=True,
                    ),
                    dl.AccuracyCallback(),
                    dl.CriterionCallback(),
                ]
            )

    wandb.finish()
