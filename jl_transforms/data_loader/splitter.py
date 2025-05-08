import numpy as np
from torch.utils.data import Dataset, Subset


def apply_matrix_to_dataset(
    dataset: Dataset,
    matrix: np.ndarray,
) -> list[Subset]:
    targets = np.array(dataset.targets)  # type: ignore[attr-defined]
    class_indices = _group_indices_by_class(targets, matrix.shape[0])
    client_indices = _split_indices_by_matrix(class_indices, matrix)
    return [Subset(dataset, indices) for indices in client_indices]


def _group_indices_by_class(
    targets: np.ndarray,
    num_classes: int,
) -> list[np.ndarray]:
    return [
        np.where(targets == class_id)[0] for class_id in range(num_classes)
    ]


def _split_indices_by_matrix(
    class_indices: list[np.ndarray],
    matrix: np.ndarray,
) -> list[list[int]]:
    num_clients = matrix.shape[1]
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id, indices in enumerate(class_indices):
        _assign_indices_to_clients(client_indices, indices, matrix[class_id])

    return client_indices


def _assign_indices_to_clients(
    client_indices: list[list[int]],
    indices: np.ndarray,
    proportions: np.ndarray,
) -> None:
    np.random.shuffle(indices)
    counts = (proportions * len(indices)).astype(int)
    counts[-1] = len(indices) - np.sum(counts[:-1])  # noqa: WPS221

    start = 0
    for client_id, count in enumerate(counts):
        selected = indices[start : start + count]  # noqa: E203
        client_indices[client_id].extend(selected)
        start += count
