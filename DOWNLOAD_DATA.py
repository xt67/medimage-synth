#!/usr/bin/env python3
"""Download script for MedImage Synth - Download medical imaging datasets"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
import gdown


def check_requirements():
    """Check and install required packages."""
    print("Checking requirements...")
    packages = ["gdown", "datasets", "huggingface_hub", "kaggle"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  {pkg}: OK")
        except ImportError:
            print(f"  {Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


def download_tb_xray(output_dir: str = "data/raw/TB_Chest_Radiography_Dataset"):
    """Download TB Chest X-ray Dataset.
    
    Get from: https://www.kaggle.com/datasets/tawsifuture/tuberculosis-tb-chest-xray-dataset
    Or use this Google Drive link (set to Anyone with link):
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Downloading TB X-ray Dataset ===")
    print("Options:")
    print("1. Kaggle: https://www.kaggle.com/datasets/tawsifuture/tuberculosis-tb-chest-xray-dataset")
    print("2. Google Drive - Set link to 'Anyone with link' and update the link below")
    print("3. Manual download and place in data/raw/")
    
    tb_zip = output_path.parent / "tb_dataset.zip"
    
    DRIVE_LINK = input("Enter Google Drive share link (or press Enter to skip): ").strip()
    
    if DRIVE_LINK:
        try:
            file_id = DRIVE_LINK.split("/uc?id=")[-1].split("&")[0]
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading from Drive ID: {file_id}")
            gdown.download(url, str(tb_zip), quiet=False)
            
            if tb_zip.exists():
                print("Extracting...")
                with zipfile.ZipFile(tb_zip, "r") as zip_ref:
                    zip_ref.extractall(output_path.parent)
                tb_zip.unlink()
                print(f"Extracted to: {output_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download manually from Kaggle or HuggingFace")


def download_chestxray14(output_dir: str = "data/raw/nih_chestxray"):
    """Download NIH Chest X-ray14 dataset."""
    print("\n=== Downloading NIH Chest X-ray14 ===")
    print("Source: https://www.kaggle.com/datasets/nih-chest-xrays/data")
    print("Or: https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("Downloading from HuggingFace (small subset)...")
        ds = load_dataset("Manas2703/chest-xray-14", split="train[:200]", trust_remote_code=False)
        
        for i, item in enumerate(ds):
            if i >= 200:
                break
            try:
                img = item.get("image")
                if img:
                    img.save(output_path / f"xray_{i:05d}.png")
            except:
                continue
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Please download manually")


def download_kaggle_dataset(dataset: str, output_dir: str):
    """Download from Kaggle."""
    print(f"\n=== Downloading {dataset} ===")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        api.dataset_download_files(dataset, path=output_path, unzip=True)
        print(f"Downloaded to: {output_path}")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Make sure kaggle.json is in ~/.kaggle/")


def create_sample_data(output_dir: str = "data/processed"):
    """Create sample data structure for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    classes = ["Normal", "Tuberculosis"]
    for cls in classes:
        cls_dir = output_path / cls
        cls_dir.mkdir(exist_ok=True)
        print(f"Created: {cls_dir}")
    
    print(f"\nPlace your dataset images in:")
    print(f"  data/processed/Normal/")
    print(f"  data/processed/Tuberculosis/")


def main():
    print("=" * 60)
    print("MedImage Synth - Dataset Downloader")
    print("=" * 60)
    
    check_requirements()
    
    print("\nSelect dataset to download:")
    print("1. Tuberculosis X-ray Dataset (Kaggle/Google Drive)")
    print("2. NIH Chest X-ray14 (HuggingFace/Kaggle)")
    print("3. Create folder structure only")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        download_tb_xray()
    elif choice == "2":
        download_chestxray14()
    elif choice == "3":
        create_sample_data()
    else:
        print("Invalid choice")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()