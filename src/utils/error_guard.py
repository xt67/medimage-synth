import torch
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class OOMGuard:
    """Guard for handling CUDA out of memory errors.
    
    Usage:
        with OOMGuard() as guard:
            # training code
            if guard.oom:
                # reduce batch size and retry
    """
    
    def __init__(self, logger_ref=None):
        self.oom = False
        self.logger_ref = logger_ref or logger
    
    @contextmanager
    def __call__(self):
        try:
            yield self
        except torch.cuda.OutOfMemoryError as e:
            self.oom = True
            self.logger_ref.warning(f"CUDA OOM detected: {e}. Clearing cache and reducing batch size.")
            torch.cuda.empty_cache()
            raise


class NaNGuard:
    """Guard for detecting NaN in gradients.
    
    Usage:
        guard = NaNGuard()
        guard.check_after_step(optimizer)
    """
    
    def __init__(self, logger_ref=None):
        self.logger_ref = logger_ref or logger
        self.has_nan = False
    
    def check_after_step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check for NaN in model parameters after backward pass.
        
        Args:
            optimizer: The optimizer with model parameters.
        
        Returns:
            True if NaN detected, False otherwise.
        """
        self.has_nan = False
        for param in optimizer.param_groups[0]["params"]:
            if param.grad is not None and torch.isnan(param.grad).any():
                self.has_nan = True
                self.logger_ref.warning(f"NaN detected in gradients for parameter {param.shape}")
                break
        return self.has_nan


def validate_batch(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor") -> None:
    """Validate tensor shape and content.
    
    Args:
        tensor: Tensor to validate.
        expected_shape: Expected shape tuple.
        name: Name of tensor for error messages.
    
    Raises:
        RuntimeError: If validation fails.
    """
    if tensor.shape != expected_shape:
        raise RuntimeError(
            f"{name} shape {tensor.shape} does not match expected {expected_shape}"
        )
    
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf values")


def check_model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Validate model output.
    
    Args:
        model: PyTorch model.
        x: Input tensor.
    
    Returns:
        Model output tensor.
    """
    output = model(x)
    if not torch.isfinite(output).all():
        raise RuntimeError(f"Model output contains NaN or Inf values")
    return output