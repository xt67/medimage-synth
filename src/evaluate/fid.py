import torch
import numpy as np
from pathlib import Path
from pytorch_fid.fid_score import calculate_fid_given_paths


def compute_fid(real_images_path: str, fake_images_path: str, device: str = "cuda", batch_size: int = 50) -> float:
    """Compute FID score between real and generated images.
    
    Args:
        real_images_path: Path to real images folder.
        fake_images_path: Path to generated images folder.
        device: Device to run computation on.
        batch_size: Batch size for Inception v3.
    
    Returns:
        FID score.
    """
    paths = [real_images_path, fake_images_path]
    fid_score = calculate_fid_given_paths(paths, batch_size, device, dims=2048)
    return fid_score


def compute_fid_from_tensors(real_images: torch.Tensor, fake_images: torch.Tensor, device: str = "cuda") -> float:
    """Compute FID score from image tensors.
    
    Args:
        real_images: Tensor of real images (B, C, H, W).
        fake_images: Tensor of fake images (B, C, H, W).
        device: Device to run on.
    
    Returns:
        FID score.
    """
    import tempfile
    
    real_path = Path(tempfile.mkdtemp())
    fake_path = Path(tempfile.mkdtemp())
    
    from torchvision.utils import save_image
    
    for i, img in enumerate(real_images):
        save_image(img, real_path / f"real_{i}.png")
    for i, img in enumerate(fake_images):
        save_image(img, fake_path / f"fake_{i}.png")
    
    fid_score = compute_fid(str(real_path), str(fake_path), device)
    
    import shutil
    shutil.rmtree(real_path)
    shutil.rmtree(fake_path)
    
    return fid_score