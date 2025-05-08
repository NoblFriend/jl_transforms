from typing import Literal

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import]

from jl_transforms.data_loader.distribution import generate_distribution_matrix
from jl_transforms.data_loader.splitter import apply_matrix_to_dataset
from jl_transforms.misc import log_class_client_heatmap


def load_cifar10(
    data_root: str,
) -> dict[Literal["train", "test"], datasets.CIFAR10]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return {
        "train": datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform
        ),
        "test": datasets.CIFAR10(
            data_root, train=False, download=True, transform=transform
        ),
    }


def get_client_loaders(cfg: DictConfig):
    dataset_map = load_cifar10(cfg.data.root)
    matrix = generate_distribution_matrix(cfg)
    client_subsets = apply_matrix_to_dataset(dataset_map["train"], matrix)

    log_class_client_heatmap(matrix, cfg)

    train_loaders = [
        DataLoader(
            subset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=2,
        )
        for subset in client_subsets
    ]
    test_loader = DataLoader(
        dataset_map["test"],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loaders, test_loader
