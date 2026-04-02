import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> dict:
    """Compute classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_probs: Predicted probabilities (optional).
    
    Returns:
        Dictionary containing metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    if y_probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics["roc_auc"] = None
    
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics.
    """
    print("=" * 50)
    print("Classification Metrics")
    print("=" * 50)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"\n{key}:")
            cm = np.array(value)
            print(cm)
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)