import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Callable, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    """Dataset class for medical images (X-rays, MRI).
    
    Args:
        root_dir: Root directory containing class folders.
        transform: Transform to apply to images.
        target_classes: List of class names to load. None for all.
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_classes: Optional[List[str]] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        if target_classes:
            self.classes = [c for c in self.classes if c.name in target_classes]
        
        self.class_to_idx = {c.name: i for i, c in enumerate(self.classes)}
        
        skipped_files = []
        for class_dir in self.classes:
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    try:
                        img = Image.open(img_path)
                        img.verify()
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir.name])
                    except Exception as e:
                        skipped_files.append((str(img_path), str(e)))
        
        if skipped_files:
            logger.warning(f"Skipped {len(skipped_files)} corrupted files")
            log_path = self.root_dir.parent / "skipped_files.log"
            with open(log_path, "w") as f:
                f.writelines([f"{p[0]}: {p[1]}\n" for p in skipped_files])
        
        if len(self) == 0:
            raise RuntimeError(f"No images found in {root_dir}. Check dataset path and structure.")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]


def get_transforms(
    image_size: int = 128,
    is_training: bool = True,
    normalize_range: Tuple[float, float] = (-1, 1)
) -> transforms.Compose:
    """Create image transforms.
    
    Args:
        image_size: Target image size.
        is_training: Whether for training (with augmentation).
        normalize_range: Range for normalization.
    
    Returns:
        Composed transforms.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
    ]
    
    if is_training:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5],
            inplace=True
        ),
    ])
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    target_classes: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.
    
    Args:
        data_dir: Directory containing class folders.
        batch_size: Batch size for dataloaders.
        image_size: Target image size.
        num_workers: Number of worker processes.
        train_ratio: Ratio of training data.
        val_ratio: Ratio of validation data.
        seed: Random seed for splitting.
        target_classes: Classes to include.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    full_dataset = MedicalImageDataset(data_dir, train_transform, target_classes)
    
    generator = torch.Generator().manual_seed(seed)
    total_len = len(full_dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=generator
    )
    
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    splits = {
        "train": train_len,
        "val": val_len,
        "test": test_len,
        "seed": seed
    }
    splits_path = Path(data_dir).parent / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    class_counts = {}
    for i, c in enumerate(full_dataset.classes):
        class_counts[c.name] = sum(1 for label in full_dataset.labels if label == i)
    logger.info(f"Dataset class distribution: {class_counts}")
    
    if any(c < 50 for c in class_counts.values()):
        logger.warning(f"Some classes have fewer than 50 samples. Consider oversampling.")
    
    return train_loader, val_loader, test_loader


def grayscale_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert grayscale image to 3-channel RGB.
    
    Args:
        image: Grayscale tensor of shape (B, 1, H, W).
    
    Returns:
        RGB tensor of shape (B, 3, H, W).
    """
    return image.repeat(1, 3, 1, 1)