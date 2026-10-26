#!/usr/bin/env python3
"""Run real-only classifier baseline with multiple seeds.

This script runs the complete baseline experiment (real data only, no augmentation)
with the specified seeds and saves results.
"""
import yaml
import torch
import json
import os
from pathlib import Path

from src.data_loader import create_dataloaders
from src.classifier.train_cls import train_classifier, evaluate_classifier
from src.utils.seed import set_seed


def run_baseline(seed: int, epochs: int = 20, device_str: str = "cpu") -> dict:
    """Run a single baseline experiment with the given seed."""
    device = torch.device(device_str)
    set_seed(seed)

    # Load config
    cfg = yaml.safe_load(open("configs/classifier.yaml"))
    cfg["epochs"] = epochs
    cfg["pretrained"] = True
    cfg["batch_size"] = 16
    cfg["seed"] = seed

    # Create data loaders
    tr, vl, te = create_dataloaders(
        cfg["data_dir"],
        batch_size=cfg["batch_size"],
        image_size=128,
        num_workers=0
    )

    # Train
    output_dir = Path("outputs") / f"baseline_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = train_classifier(
        config_path="configs/classifier.yaml",  # We'll pass the modified config via args
        train_loader=tr,
        val_loader=vl,
        device=device,
        output_dir=str(output_dir)
    )

    # Evaluate
    metrics = evaluate_classifier(model, te, device)
    metrics["seed"] = seed
    metrics["epochs"] = epochs
    metrics["device"] = device_str

    return metrics


def run_with_custom_config(seed: int, epochs: int, device_str: str, config_overrides: dict) -> dict:
    """Run baseline with custom config overrides."""
    device = torch.device(device_str)
    set_seed(seed)

    cfg = yaml.safe_load(open("configs/classifier.yaml"))
    cfg.update(config_overrides)
    cfg["seed"] = seed

    # Write temp config
    tmp_cfg_path = Path("outputs") / f"tmp_classifier_seed{seed}.yaml"
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    tr, vl, te = create_dataloaders(
        cfg["data_dir"],
        batch_size=cfg["batch_size"],
        image_size=128,
        num_workers=0
    )

    output_dir = Path("outputs") / f"baseline_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = train_classifier(
        config_path=str(tmp_cfg_path),
        train_loader=tr,
        val_loader=vl,
        device=device,
        output_dir=str(output_dir)
    )

    metrics = evaluate_classifier(model, te, device)
    metrics["seed"] = seed
    metrics["epochs"] = epochs
    metrics["device"] = device_str

    return metrics


if __name__ == "__main__":
    seeds = [42, 123, 999]
    epochs = 20
    device = "cpu"

    print("=" * 60)
    print("AURORA - Real-Only Classifier Baseline")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 60)

    config_overrides = {
        "epochs": epochs,
        "pretrained": True,
        "batch_size": 16,
    }

    all_results = []
    for seed in seeds:
        print(f"\n--- Running seed {seed} ---")
        try:
            metrics = run_with_custom_config(seed, epochs, device, config_overrides)
            all_results.append(metrics)
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"  AUROC:     {metrics['roc_auc']:.4f}")
            print(f"  AUPRC:     {metrics['auprc']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Compute mean ± std
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY (mean ± std)")
        print("=" * 60)
        import numpy as np
        keys = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "auprc"]
        for k in keys:
            vals = [r[k] for r in all_results if r[k] is not None]
            if vals:
                print(f"  {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save results
    out_path = Path("outputs") / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")