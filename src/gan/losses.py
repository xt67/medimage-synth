import torch
import torch.nn as nn


class GANLoss:
    """GAN loss functions container.
    
    Args:
        loss_type: Type of loss ("bce", "wgan-gp").
    """
    
    def __init__(self, loss_type: str = "bce"):
        self.loss_type = loss_type
        if loss_type == "bce":
            self.criterion = nn.BCELoss()
        elif loss_type == "wgan":
            pass
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def generator_loss(self, pred_fake: torch.Tensor, target_real: bool = True) -> torch.Tensor:
        """Calculate generator loss.
        
        Args:
            pred_fake: Discriminator prediction on fake images.
            target_real: Whether generator wants discriminator to predict real.
        
        Returns:
            Generator loss value.
        """
        if self.loss_type == "bce":
            target = torch.ones_like(pred_fake) if target_real else torch.zeros_like(pred_fake)
            return self.criterion(pred_fake, target)
        elif self.loss_type == "wgan":
            return -pred_fake.mean()
        raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def discriminator_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
        label_smoothing: float = 0.9
    ) -> torch.Tensor:
        """Calculate discriminator loss.
        
        Args:
            pred_real: Discriminator prediction on real images.
            pred_fake: Discriminator prediction on fake images.
            label_smoothing: Smoothing factor for real labels.
        
        Returns:
            Discriminator loss value.
        """
        if self.loss_type == "bce":
            real_target = torch.ones_like(pred_real) * label_smoothing
            fake_target = torch.zeros_like(pred_fake)
            real_loss = self.criterion(pred_real, real_target)
            fake_loss = self.criterion(pred_fake, fake_target)
            return (real_loss + fake_loss) / 2
        elif self.loss_type == "wgan":
            return pred_fake.mean() - pred_real.mean()
        raise ValueError(f"Unknown loss type: {self.loss_type}")


def compute_gradient_penalty(discriminator: nn.Module, real_images: torch.Tensor, fake_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator model.
        real_images: Batch of real images.
        fake_images: Batch of fake images.
        device: Device to run on.
    
    Returns:
        Gradient penalty loss.
    """
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    interpolated_pred = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=interpolated_pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty