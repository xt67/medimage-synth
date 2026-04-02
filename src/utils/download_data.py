import gdown
import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional


def download_tb_dataset(output_dir: str = "data/raw") -> str:
    """Download TB X-ray dataset from Google Drive.
    
    Args:
        output_dir: Directory to save dataset.
    
    Returns:
        Path to extracted dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Tuberculosis X-ray Dataset...")
    
    url = "https://drive.google.com/uc?id=1a0KdxP2z1g6I91fB1gYvK_FYazxzBxgE"
    output_file = output_path / "tb_dataset.zip"
    
    gdown.download(url, str(output_file), quiet=False)
    
    print("Extracting...")
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall(output_path)
    
    output_file.unlink()
    
    dataset_path = output_path / "TB_Chest_Radiography_Dataset"
    return str(dataset_path)


def download_from_huggingface(dataset_name: str, output_dir: str = "data/raw") -> str:
    """Download dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset name.
        output_dir: Directory to save dataset.
    
    Returns:
        Path to downloaded dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset_name} from HuggingFace...")
    
    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    
    data_path = output_path / dataset_name.replace("/", "_")
    data_path.mkdir(parents=True, exist_ok=True)
    
    for i, example in enumerate(dataset):
        if "image" in example:
            example["image"].save(data_path / f"image_{i}.png")
        elif "image_name" in example:
            example["image"].save(data_path / example["image_name"])
    
    print(f"Dataset saved to {data_path}")
    return str(data_path)


def download_chestxray14(output_dir: str = "data/raw") -> str:
    """Download NIH Chest X-ray14 from HuggingFace.
    
    Args:
        output_dir: Directory to save dataset.
    
    Returns:
        Path to downloaded dataset.
    """
    return download_from_huggingface("Manas2703/chest-xray-14", output_dir)


def download_chest_pneumonia(output_dir: str = "data/raw") -> str:
    """Download chest X-ray pneumonia dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save dataset.
    
    Returns:
        Path to downloaded dataset.
    """
    return download_from_huggingface("hf-vision/chest-xray-pneumonia", output_dir)


def download_kaggle_dataset(dataset_url: str, output_dir: str = "data/raw") -> Optional[str]:
    """Download dataset from Kaggle.
    
    Args:
        dataset_url: Kaggle dataset URL or name.
        output_dir: Directory to save dataset.
    
    Returns:
        Path to downloaded dataset or None.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle library...")
        os.system("pip install kaggle")
        import kaggle
    
    print(f"Downloading {dataset_url} from Kaggle...")
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    api.dataset_download_files(dataset_url, path=output_path, unzip=True)
    
    return str(output_path)


def organize_dataset(source_dir: str, target_dir: str = "data/processed") -> None:
    """Organize dataset into class folders.
    
    Args:
        source_dir: Source directory with class subfolders.
        target_dir: Target directory for organized data.
    """
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    
    for class_dir in source.iterdir():
        if class_dir.is_dir():
            target_class = target / class_dir.name
            target_class.mkdir(exist_ok=True)
            
            for img in class_dir.iterdir():
                if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    shutil.copy2(img, target_class / img.name)
    
    print(f"Dataset organized at {target}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download medical imaging datasets")
    parser.add_argument("--dataset", type=str, default="tb", choices=["tb", "nih", "pneumonia"], help="Dataset to download")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    if args.dataset == "tb":
        path = download_tb_dataset(args.output)
        print(f"Dataset downloaded to: {path}")
    elif args.dataset == "nih":
        path = download_chestxray14(args.output)
        print(f"Dataset downloaded to: {path}")
    elif args.dataset == "pneumonia":
        path = download_chest_pneumonia(args.output)
        print(f"Dataset downloaded to: {path}")