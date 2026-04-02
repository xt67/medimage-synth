import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gan.discriminator import Discriminator


def test_discriminator_output_shape():
    """Test discriminator output shape."""
    netD = Discriminator(ndf=64, nc=1)
    batch_size = 4
    images = torch.randn(batch_size, 1, 128, 128)
    
    with torch.no_grad():
        output = netD(images)
    
    assert output.shape == (batch_size,)


def test_discriminator_output_range():
    """Test discriminator output is in sigmoid range [0, 1]."""
    netD = Discriminator(ndf=64, nc=1)
    images = torch.randn(4, 1, 128, 128)
    
    with torch.no_grad():
        output = netD(images)
    
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_discriminator_no_nan():
    """Test discriminator produces no NaN values."""
    netD = Discriminator(ndf=64, nc=1)
    images = torch.randn(4, 1, 128, 128)
    
    with torch.no_grad():
        output = netD(images)
    
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])