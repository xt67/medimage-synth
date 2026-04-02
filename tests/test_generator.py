import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gan.generator import Generator


def test_generator_output_shape():
    """Test generator output shape."""
    netG = Generator(z_dim=100, ngf=64, nc=1)
    batch_size = 4
    noise = torch.randn(batch_size, 100, 1, 1)
    
    with torch.no_grad():
        output = netG(noise)
    
    assert output.shape == (batch_size, 1, 128, 128)


def test_generator_output_range():
    """Test generator output is in tanh range [-1, 1]."""
    netG = Generator(z_dim=100, ngf=64, nc=1)
    noise = torch.randn(4, 100, 1, 1)
    
    with torch.no_grad():
        output = netG(noise)
    
    assert output.min() >= -1.0 and output.max() <= 1.0


def test_generator_no_nan():
    """Test generator produces no NaN values."""
    netG = Generator(z_dim=100, ngf=64, nc=1)
    noise = torch.randn(4, 100, 1, 1)
    
    with torch.no_grad():
        output = netG(noise)
    
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])