import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import MedicalImageDataset, get_transforms, grayscale_to_rgb


@pytest.fixture
def sample_transform():
    return get_transforms(image_size=128, is_training=False)


def test_grayscale_to_rgb():
    """Test grayscale to RGB conversion."""
    x = torch.randn(4, 1, 128, 128)
    x_rgb = grayscale_to_rgb(x)
    assert x_rgb.shape == (4, 3, 128, 128)


def test_transform_output_range():
    """Test transformed images are in correct range."""
    from PIL import Image
    import numpy as np
    
    # Create a proper RGB numpy array
    img_array = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, "RGB")
    
    transform = get_transforms(image_size=128, is_training=False)
    tensor = transform(img)
    
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])