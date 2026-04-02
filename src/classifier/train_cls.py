import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .model import MedicalImageClassifier, get_class_weights
from ..data_loader import grayscale_to_rgb
from ..utils.seed import set_seed
from ..utils.logger import setup_logger


def train_classifier(
    config_path: str,
    train_loader,
    val_loader,
    device: torch.device,
    use_synthetic: bool = False,
    synthetic_loader=None,
    output_dir: str = "outputs"
):
    """Train classifier model.
    
    Args:
        config_path: Path to config YAML.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device to train on.
        use_synthetic: Whether to include synthetic images.
        synthetic_loader: Synthetic image loader.
        output_dir: Output directory for checkpoints and logs.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    seed = config.get("seed", 42)
    set_seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("cls_train", str(output_path / "logs"))
    
    num_classes = config.get("num_classes", 2)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    epochs = config.get("epochs", 20)
    early_stop_patience = config.get("early_stop_patience", 10)
    
    model = MedicalImageClassifier(num_classes=num_classes, pretrained=True).to(device)
    class_weights = None
    if hasattr(train_loader.dataset, 'labels') and len(train_loader.dataset.labels) > 0:
        class_weights = get_class_weights(train_loader.dataset, num_classes)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    logger.info(f"Training classifier for {epochs} epochs")
    logger.info(f"Use synthetic augment: {use_synthetic}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            if use_synthetic and synthetic_loader is not None:
                try:
                    syn_images, syn_labels = next(iter(synthetic_loader))
                    syn_images = syn_images.to(device)
                    syn_labels = syn_labels.to(device)
                    
                    images = torch.cat([images, syn_images], dim=0)
                    labels = torch.cat([labels, syn_labels], dim=0)
                except StopIteration:
                    pass  # Synthetic loader exhausted, continue with real data only
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                images = grayscale_to_rgb(images)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        
        logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}, Recall={val_recall:.4f}, Prec={val_precision:.4f}, F1={val_f1:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path / "models" / "classifier_best.pth")
            logger.info(f"New best model saved with val_loss={avg_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("Training complete!")
    return model


def evaluate_classifier(model: nn.Module, test_loader, device: torch.device) -> dict:
    """Evaluate classifier on test set.
    
    Args:
        model: Trained classifier model.
        test_loader: Test data loader.
        device: Device to evaluate on.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = grayscale_to_rgb(images).to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    # Add ROC-AUC if both classes present
    if len(np.unique(all_labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
        except ImportError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    
    return metrics