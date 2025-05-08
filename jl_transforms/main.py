import hydra
import torch
from catalyst import dl  # type: ignore[import]
from omegaconf import DictConfig, OmegaConf
from torch import nn as nn
from tqdm import tqdm  # type: ignore[import]

import wandb
from jl_transforms.data_loader.data import get_client_loaders
from jl_transforms.misc import average_weights, local_epoch_fn
from jl_transforms.models import get_resnet18_cifar


def train_client(  # noqa: WPS211
    model, loader, global_weights, local_epochs, lambda_reg, lr, momentum
):
    runner = dl.SupervisedRunner(
        input_key="features",
        output_key="logits",
        target_key="targets",
        loss_key="loss",
    )

    def loss_fn(logits, targets):  # noqa: WPS430
        base_loss = nn.CrossEntropyLoss()(logits, targets)
        l2_reg = sum(
            torch.norm(parameter - global_weights[name]) ** 2
            for name, parameter in model.named_parameters()
        )
        return base_loss + lambda_reg / 2 * l2_reg

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    runner.train(
        model=model,
        loaders={"train": loader},
        criterion=loss_fn,
        optimizer=optimizer,
        num_epochs=local_epochs,
        callbacks=[
            dl.CriterionCallback(
                input_key="logits", target_key="targets", metric_key="loss"
            ),
            dl.OptimizerCallback(metric_key="loss"),
        ],
        logdir=None,
        verbose=True,
    )


def evaluate(model, loader):  # noqa: WPS210
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):  # noqa: WPS210
    wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )

    client_loaders, test_loader = get_client_loaders(cfg)
    num_clients = cfg.train.num_clients
    lambda_reg = cfg.train.lambda_reg
    lr = cfg.train.lr
    momentum = cfg.train.momentum
    num_rounds = cfg.train.num_rounds

    client_models = [get_resnet18_cifar() for _ in range(num_clients)]
    global_model = get_resnet18_cifar()
    global_weights = global_model.state_dict()

    round_bar = tqdm(range(num_rounds), desc="📦 Communication Rounds")

    for round_k in round_bar:
        local_epochs = local_epoch_fn(round_k)
        round_bar.set_postfix(local_epochs=local_epochs)

        client_bar = tqdm(
            range(num_clients),
            desc=f"➡️ Clients in Round {round_k}",
            leave=False,
        )
        for client_idx in client_bar:
            client_bar.set_description(f"➡️ Client {client_idx} training")
            train_client(
                model=client_models[client_idx],
                loader=client_loaders[client_idx],
                global_weights=global_weights,
                local_epochs=local_epochs,
                lambda_reg=lambda_reg,
                lr=lr,
                momentum=momentum,
            )

        tqdm.write("\n🔗 Aggregating client weights...")
        client_weights = [model.state_dict() for model in client_models]
        global_weights = average_weights(client_weights)
        global_model.load_state_dict(global_weights)

        acc = evaluate(global_model, test_loader)
        tqdm.write(
            f"\n🧪 Server test accuracy after round {round_k}: {acc:.4f}"
        )
        wandb.log({"server/acc": acc}, step=round_k)


if __name__ == "__main__":
    main()
