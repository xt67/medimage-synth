"""API handlers for MedImage Synth - supports multiple data sources."""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HuggingFaceAPI:
    """HuggingFace API wrapper for dataset downloads."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("HF_TOKEN")
        self.base_url = "https://huggingface.co/api"
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset information from HuggingFace."""
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset
        
        info = {
            "name": dataset_name,
            "available": True,
            "source": "huggingface"
        }
        return info
    
    def download_dataset(self, dataset_name: str, output_dir: str = "data/raw") -> str:
        """Download dataset from HuggingFace."""
        from datasets import load_dataset
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {dataset_name} from HuggingFace...")
        
        try:
            dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
            
            data_path = output_path / dataset_name.replace("/", "_")
            data_path.mkdir(parents=True, exist_ok=True)
            
            for i, example in enumerate(dataset):
                if "image" in example:
                    example["image"].save(data_path / f"image_{i:05d}.png")
                elif "image_name" in example:
                    example["image"].save(data_path / example["image_name"])
            
            logger.info(f"Dataset saved to {data_path}")
            return str(data_path)
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return str(output_path)


class KaggleAPI:
    """Kaggle API wrapper for dataset downloads."""
    
    def __init__(self):
        self.api = None
        self._init_api()
    
    def _init_api(self):
        """Initialize Kaggle API."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Kaggle API initialized")
        except Exception as e:
            logger.warning(f"Kaggle API not available: {e}")
    
    def download_dataset(self, dataset_name: str, output_dir: str) -> Optional[str]:
        """Download dataset from Kaggle."""
        if not self.api:
            logger.error("Kaggle API not initialized")
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading {dataset_name} from Kaggle...")
            self.api.dataset_download_files(dataset_name, path=output_path, unzip=True)
            return str(output_path)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None


class GoogleDriveAPI:
    """Google Drive API for shared files."""
    
    def __init__(self):
        pass
    
    def download_file(self, file_id: str, output_path: str) -> str:
        """Download file from Google Drive."""
        import gdown
        
        url = f"https://drive.google.com/uc?id={file_id}"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading from Google Drive: {file_id}")
        gdown.download(url, str(output), quiet=False)
        
        if output.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(output, "r") as zip_ref:
                zip_ref.extractall(output.parent)
            output.unlink()
        
        return str(output.parent)


class OpenAIAPI:
    """Optional OpenAI API for enhanced image analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    def is_available(self) -> bool:
        """Check if API key is available."""
        return self.api_key is not None
    
    def is_enabled(self) -> bool:
        """Check if OpenAI integration is enabled."""
        return self.api_key is not None


def get_api_handler(api_type: str):
    """Factory function to get API handler."""
    handlers = {
        "huggingface": HuggingFaceAPI,
        "kaggle": KaggleAPI,
        "gdrive": GoogleDriveAPI,
        "openai": OpenAIAPI
    }
    return handlers.get(api_type, HuggingFaceAPI)()

