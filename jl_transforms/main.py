from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm  # type: ignore[import]

import wandb
from jl_transforms.data_loader.data import get_client_loaders
from jl_transforms.misc import average_weights, local_epoch_fn
from jl_transforms.models import get_resnet18_cifar
from jl_transforms.run.logger_hook import TqdmLoggerHook, WandbLoggerHook
from jl_transforms.run.metric_hook import AccuracyHook, LossHook
from jl_transforms.run.runner import Runner


def setup_wandb(cfg: DictConfig) -> None:
    wandb.login()
    wandb.init(
        project="impacts-dropout",
        config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
    )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):  # noqa: WPS210
    setup_wandb(cfg)
    client_loaders, test_loader = get_client_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = get_resnet18_cifar().to(device)
    global_weights = global_model.state_dict()

    client_runners = []
    for idx, loader in enumerate(client_loaders):
        model = get_resnet18_cifar().cpu()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum
        )

        def make_loss_fn(model_ref, global_weights_ref):  # noqa: WPS430
            def loss_fn(logits, targets):  # noqa: WPS430
                base = torch.nn.functional.cross_entropy(logits, targets)
                l2 = sum(
                    torch.norm(weight - global_weights_ref[name]) ** 2
                    for name, weight in model_ref.named_parameters()
                )
                return base + cfg.train.lambda_reg / 2 * l2

            return loss_fn

        runner = Runner()
        runner.model = model
        runner.optimizer = optimizer
        runner.loss_fn = make_loss_fn(model, global_weights)
        runner.device = device
        runner.loaders = {"train": loader}
        runner.name = f"client_{idx}"
        runner.hooks = [
            TqdmLoggerHook(),
            WandbLoggerHook(prefix=runner.name),
            AccuracyHook(num_classes=10),
            LossHook(),
        ]
        client_runners.append(runner)

    global_runner = Runner()
    global_runner.model = global_model
    global_runner.device = device
    global_runner.loaders = {"test": test_loader}
    global_runner.name = "server"
    global_runner.hooks = [
        WandbLoggerHook(prefix="server"),
        AccuracyHook(num_classes=10),
    ]

    for round_k in tqdm(
        range(cfg.train.num_rounds), desc="📦 Communication Rounds"
    ):
        local_epochs = local_epoch_fn(round_k)
        tqdm.write(f"➡️ Round {round_k} | Local epochs = {local_epochs}")

        for runner in tqdm(client_runners, desc="➡️ Clients", leave=False):
            runner.model.load_state_dict(global_weights)
            runner.train(num_epochs=local_epochs)

        global_weights = average_weights(
            [client.model.state_dict() for client in client_runners]
        )
        global_runner.model.load_state_dict(global_weights)
        global_runner.epoch = round_k
        global_runner.evaluate("test")


if __name__ == "__main__":
    main()
