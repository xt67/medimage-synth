import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from .generator import Generator
from .discriminator import Discriminator
from .losses import GANLoss, compute_gradient_penalty
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.visualize import save_image_grid
from src.utils.error_guard import OOMGuard


def train_gan(
    config_path: str,
    data_loader,
    device: torch.device,
    output_dir: str = "outputs"
):
    """Train GAN model.
    
    Args:
        config_path: Path to config YAML.
        data_loader: Training data loader.
        device: Device to train on.
        output_dir: Output directory for checkpoints and logs.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    seed = config.get("seed", 42)
    set_seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("gan_train", str(output_path / "logs"))
    writer = SummaryWriter(str(output_path / "runs"))
    
    z_dim = config["z_dim"]
    lr_g = config["lr_g"]
    lr_d = config["lr_d"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    epochs = config["epochs"]
    d_steps = config.get("d_steps", 1)
    label_smoothing = config.get("label_smoothing", 0.9)
    use_amp = config.get("use_amp", True)
    loss_type = config.get("loss_type", "bce")
    
    netG = Generator(z_dim=z_dim, ngf=config.get("ngf", 64), nc=1).to(device)
    netD = Discriminator(ndf=config.get("ndf", 64), nc=1).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    
    criterion = GANLoss(loss_type=loss_type)
    
    oom_guard = OOMGuard(logger)
    
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
    
    g_losses = []
    d_losses = []
    
    logger.info(f"Starting GAN training for {epochs} epochs")
    logger.info(f"Device: {device}, Batch size: {data_loader.batch_size}")
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for i, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            
            # Initialize variables that might be unbound
            lossD = torch.tensor(0.0, device=device)
            output_real = torch.tensor(0.0, device=device)
            output_fake = torch.tensor(0.0, device=device)
            
            with oom_guard():
                for _ in range(d_steps):
                    netD.zero_grad()
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_real = netD(real_images)
                        
                        fake_images = netG(noise).detach()
                        output_fake = netD(fake_images)
                        
                        lossD = criterion.discriminator_loss(
                            output_real, output_fake, label_smoothing
                        )
                    
                    if loss_type == "wgan-gp":
                        gp = compute_gradient_penalty(netD, real_images, fake_images, device)
                        lossD = lossD + 10 * gp
                    
                    lossD.backward()
                    if loss_type == "bce":
                        torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
                    optimizerD.step()
            
            netG.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_images = netG(noise)
                output_fake = netD(fake_images)
                lossG = criterion.generator_loss(output_fake, target_real=True)
            
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
            optimizerG.step()
            
            # Skip if NaN detected
            if torch.isnan(lossG) or torch.isnan(lossD):
                logger.warning(f"NaN detected at epoch {epoch}, batch {i}. Skipping update.")
                continue
            
            epoch_g_loss += lossG.item()
            epoch_d_loss += lossD.item()
            
            if i % 50 == 0:
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(data_loader)}] "
                    f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f} "
                    f"D(x): {output_real.mean().item():.4f} D(G(z)): {output_fake.mean().item():.4f}"
                )
        
        avg_g_loss = epoch_g_loss / len(data_loader)
        avg_d_loss = epoch_d_loss / len(data_loader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = output_path / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "netG_state": netG.state_dict(),
                "netD_state": netD.state_dict(),
                "optimizerG_state": optimizerG.state_dict(),
                "optimizerD_state": optimizerD.state_dict(),
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
            
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_image_grid(
                    fake,
                    str(output_path / "samples" / f"epoch_{epoch+1}.png"),
                    nrow=8
                )
        
        if (epoch + 1) % 25 == 0:
            torch.save(netG.state_dict(), output_path / "models" / "generator_latest.pth")
        
        # Ensure models directory exists
        (output_path / "models").mkdir(exist_ok=True)
        torch.save(netG.state_dict(), output_path / "models" / "generator_best.pth")
    logger.info("Training complete! Best model saved.")
    
    np.save(output_path / "g_losses.npy", np.array(g_losses))
    np.save(output_path / "d_losses.npy", np.array(d_losses))
    
    return netG, netD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dcgan.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")