"""Dataset loading and stratified splits for ImageFolder weather classification."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def stratified_split(
    dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
):
    import numpy as np

    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(random_seed)
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)
        n_cls = len(cls_indices)
        n_train = int(round(train_ratio * n_cls))
        n_val = int(round(val_ratio * n_cls))
        n_test = n_cls - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_train = n_cls - n_val
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train : n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val :])
    return train_idx, val_idx, test_idx


class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        if not isinstance(x, __import__("torch").Tensor):
            from torchvision.transforms import functional as F

            x = F.to_tensor(x)
        return x, y


def weather_transforms():
    """Train / eval transforms matching the original notebooks."""
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform_train, transform_eval


def build_imagefolder_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
):
    """Return train/val/test loaders, num_classes, and class names."""
    import torch

    transform_train, transform_eval = weather_transforms()
    full_dataset = datasets.ImageFolder(root=data_root, transform=None)
    num_classes = len(full_dataset.classes)
    train_idx, val_idx, test_idx = stratified_split(full_dataset, random_seed=42)

    train_ds = SubsetWithTransform(full_dataset, train_idx, transform_train)
    val_ds = SubsetWithTransform(full_dataset, val_idx, transform_eval)
    test_ds = SubsetWithTransform(full_dataset, test_idx, transform_eval)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    dl_kw: dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kw)
    return train_loader, val_loader, test_loader, num_classes, full_dataset.classes
