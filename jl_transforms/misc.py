import copy
import os
import random

import numpy as np
import seaborn as sns  # type: ignore[import]
import torch
from matplotlib import pyplot as plt  # type: ignore[import]

import wandb  # type: ignore[import]


def local_epoch_fn(round_idx: int) -> int:
    return 4 * (round_idx + 1)


def average_weights(weights_list):
    avg = copy.deepcopy(weights_list[0])
    for key in avg:
        for weight in weights_list[1:]:
            avg[key] += weight[key]
        avg[key] /= len(weights_list)
    return avg


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_class_client_heatmap(
    table: np.ndarray, cfg, prefix: str = "class_distribution"
):
    os.makedirs("plots", exist_ok=True)
    fname = f"plots/{prefix}_heatmap.png"

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(table, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    ax.set_title(
        "Class-Client Distribution (mode={})".format(cfg.data.split_mode)
    )
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Class ID")
    plt.savefig(fname)
    plt.close()

    wandb.log({f"{prefix}_heatmap": wandb.Image(fname)})
