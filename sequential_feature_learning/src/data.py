from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


COARSE_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
    "trees", "vehicles 1", "vehicles 2",
]
TASK_NAMES = {name.replace(" ", "_").replace("-", "_"): name for name in COARSE_NAMES}
TRAINING_TASKS = ["aquatic_mammals", "fish", "large_carnivores", "small_mammals"]
PROBE_TASKS = ["medium_sized_mammals", "vehicles_2"]
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class CoarseTaskDataset(Dataset):
    def __init__(self, base: Dataset, coarse_labels: list[int], selected_coarse: int):
        self.base = base
        self.indices = [i for i, label in enumerate(coarse_labels) if label == selected_coarse]
        self.fine_labels = [int(base.targets[i]) for i in self.indices]
        self.classes = sorted(set(self.fine_labels))
        if len(self.classes) != 5:
            raise ValueError(f"Expected five fine classes, found {len(self.classes)}")
        self.to_local = {fine: local for local, fine in enumerate(self.classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, _ = self.base[self.indices[item]]
        return image, self.to_local[self.fine_labels[item]]


def _coarse_labels(dataset: datasets.CIFAR100) -> list[int]:
    # CIFAR-100's official Python archive stores the coarse label parallel to each fine label.
    split_file = "train" if dataset.train else "test"
    path = Path(dataset.root) / dataset.base_folder / split_file
    with path.open("rb") as handle:
        record = pickle.load(handle, encoding="bytes")
    coarse = record.get(b"coarse_labels")
    fine = record.get(b"fine_labels")
    if coarse is None or fine is None:
        raise RuntimeError(f"Official coarse/fine labels were not found in {path}")
    if list(map(int, fine)) != list(map(int, dataset.targets)):
        raise RuntimeError("CIFAR-100 fine-label ordering differs from torchvision dataset ordering")
    return list(map(int, coarse))


def make_transforms():
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, evaluation


def task_id(name: str) -> int:
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in TASK_NAMES:
        raise ValueError(f"Unknown CIFAR-100 coarse task {name!r}; choices: {sorted(TASK_NAMES)}")
    return COARSE_NAMES.index(TASK_NAMES[key])


def make_task_dataset(root: str, name: str, train: bool, download: bool = True,
                      augment: bool = False) -> CoarseTaskDataset:
    train_tf, eval_tf = make_transforms()
    base = datasets.CIFAR100(root=root, train=train, transform=train_tf if augment else eval_tf,
                             download=download)
    return CoarseTaskDataset(base, _coarse_labels(base), task_id(name))


def stratified_indices(dataset: CoarseTaskDataset, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    labels = np.asarray([dataset.to_local[fine] for fine in dataset.fine_labels])
    for label in range(5):
        candidates = np.flatnonzero(labels == label)
        if per_class > len(candidates):
            raise ValueError(f"Requested {per_class} examples for class {label}, only {len(candidates)} available")
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return sorted(selected)


def task_loader(root: str, name: str, train: bool, batch_size: int, workers: int,
                download: bool, augment: bool, limit_per_class: int | None = None,
                seed: int = 0, shuffle: bool | None = None) -> tuple[DataLoader, CoarseTaskDataset]:
    dataset = make_task_dataset(root, name, train, download, augment)
    if limit_per_class is not None:
        dataset = Subset(dataset, stratified_indices(dataset, limit_per_class, seed))
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=(train if shuffle is None else shuffle), num_workers=workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return loader, dataset


def wrapped_targets(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [dataset.dataset[i][1] for i in dataset.indices]
    return [dataset[i][1] for i in range(len(dataset))]
