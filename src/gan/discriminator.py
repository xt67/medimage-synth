import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """DCGAN Discriminator for medical image classification.
    
    Args:
        ndf: Number of discriminator features.
        nc: Number of input channels (1 for grayscale).
    """
    
    def __init__(self, ndf: int = 64, nc: int = 1):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            else:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        # Adaptive average pooling to get 1x1 output
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(-1, 1).squeeze(1)