import torch
import torchvision.utils as vutils
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_image_grid(images: torch.Tensor, filepath: str, nrow: int = 8, normalize: bool = True, range: tuple = (-1, 1)):
    """Save a grid of images.
    
    Args:
        images: Tensor of shape (B, C, H, W).
        filepath: Path to save the grid.
        nrow: Number of images per row.
        normalize: Whether to normalize images.
        range: Pixel value range.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, range=range)
    vutils.save_image(grid, output_path)


def plot_loss_curves(losses: dict, save_path: str, title: str = "Training Losses"):
    """Plot and save loss curves.
    
    Args:
        losses: Dictionary with keys as loss names and values as lists of loss values.
        save_path: Path to save the plot.
        title: Plot title.
    """
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for name, values in losses.items():
        plt.plot(values, label=name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to displayable image.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W).
    
    Returns:
        Numpy array of shape (H, W, C) with values in [0, 255].
    """
    tensor = tensor.cpu().detach()
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
        img = (tensor + 1) / 2 * 255
    else:
        tensor = tensor.permute(1, 2, 0)
        img = (tensor + 1) / 2 * 255
    return img.clamp(0, 255).to(torch.uint8).numpy()