# Utils package init - conditional imports to avoid missing dependencies
from .seed import set_seed
from .logger import setup_logger
from .error_guard import OOMGuard, NaNGuard, validate_batch

# Visualize components - may fail if torchvision/matplotlib not available
try:
    from .visualize import save_image_grid, plot_loss_curves
except ImportError:
    save_image_grid = None
    plot_loss_curves = None

__all__ = [
    "set_seed",
    "setup_logger",
    "save_image_grid",
    "plot_loss_curves",
    "OOMGuard",
    "NaNGuard",
    "validate_batch",
]