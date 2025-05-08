import numpy as np
from omegaconf import DictConfig


def generate_distribution_matrix(cfg: DictConfig) -> np.ndarray:
    generators = {
        "table": _generate_table_matrix,
        "uniform": _generate_uniform_matrix,
        "dirichlet": _generate_dirichlet_matrix,
    }

    mode = cfg.data.split_mode
    if mode not in generators:
        raise ValueError("Unknown split_mode: {}".format(mode))

    return generators[mode](cfg)


def _generate_table_matrix(cfg: DictConfig) -> np.ndarray:
    table = cfg.data.table
    if table is None:
        raise ValueError("Table must be provided in 'table' mode")

    matrix = np.array(table, dtype=np.float64)
    shape = (cfg.data.num_classes, cfg.train.num_clients)
    if matrix.shape != shape:
        raise ValueError(
            "Expected shape {}, got {}".format(shape, matrix.shape)
        )

    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def _generate_uniform_matrix(cfg: DictConfig) -> np.ndarray:
    num_classes = cfg.data.num_classes
    num_clients = cfg.train.num_clients
    return np.full(
        (num_classes, num_clients), 1.0 / num_clients, dtype=np.float64
    )


def _generate_dirichlet_matrix(cfg: DictConfig) -> np.ndarray:
    matrix = np.zeros(
        (cfg.data.num_classes, cfg.train.num_clients), dtype=np.float64
    )
    for class_id in range(cfg.data.num_classes):
        proportions = np.random.dirichlet(
            (cfg.data.alpha,) * cfg.train.num_clients
        )
        matrix[class_id] = proportions
    return matrix
