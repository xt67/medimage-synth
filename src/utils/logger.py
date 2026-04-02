import logging
import json
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "medimage", log_dir: str = "outputs/logs") -> logging.Logger:
    """Setup structured JSON logger.
    
    Args:
        name: Logger name.
        log_dir: Directory to save log files.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"train_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            }
            if hasattr(record, "epoch"):
                log_data["epoch"] = record.epoch
            if hasattr(record, "loss"):
                log_data["loss"] = record.loss
            return json.dumps(log_data)
    
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    
    return logger