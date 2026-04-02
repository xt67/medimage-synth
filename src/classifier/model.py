import torch
import torch.nn as nn
from torchvision import models


class MedicalImageClassifier(nn.Module):
    """ResNet-18 based classifier for medical images.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(MedicalImageClassifier, self).__init__()
        
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def create_classifier(num_classes: int = 2, pretrained: bool = True) -> MedicalImageClassifier:
    """Create and return classifier model.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
    
    Returns:
        Initialized classifier model.
    """
    return MedicalImageClassifier(num_classes=num_classes, pretrained=pretrained)


def get_class_weights(dataset, num_classes: int = 2) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency.
    
    Args:
        dataset: Dataset with labels.
        num_classes: Number of classes.
    
    Returns:
        Tensor of class weights.
    """
    from collections import Counter
    labels = [label for _, label in dataset]
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        weights.append(total / (num_classes * counts[i]))
    return torch.tensor(weights)