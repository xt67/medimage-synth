import torch
import torch.nn as nn

try:
    from torchvision import models as tv_models
except ImportError as e:  # pragma: no cover - environment dependent
    raise ImportError(
        "torchvision is required for the classifier. "
        "Install it with: pip install torchvision"
    ) from e


class MedicalImageClassifier(nn.Module):
    """ResNet-18 based classifier for medical images.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(MedicalImageClassifier, self).__init__()

        try:
            weights = "IMAGENET1K_V1" if pretrained else None
            self.model = tv_models.resnet18(weights=weights)
        except TypeError:
            # Older torchvision API
            self.model = tv_models.resnet18(pretrained=pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def create_classifier(num_classes: int = 2, pretrained: bool = True) -> MedicalImageClassifier:
    """Create and return classifier model."""
    return MedicalImageClassifier(num_classes=num_classes, pretrained=pretrained)


def get_class_weights(dataset, num_classes: int = 2) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency.

    Handles torch.utils.data.Subset and datasets exposing `.labels`.
    """
    from collections import Counter

    labels = []
    if hasattr(dataset, "labels") and dataset.labels is not None:
        labels = list(dataset.labels)
    elif hasattr(dataset, "indices") and hasattr(dataset.dataset, "labels"):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        # Fall back to iterating (may be slow but robust).
        for _, label in dataset:
            labels.append(label)

    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float)