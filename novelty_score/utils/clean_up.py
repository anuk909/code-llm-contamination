from pathlib import Path
import shutil

from .logger import logger


def clean_up(dir_path: Path) -> None:
    """Remove all files and directories within the given directory."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    logger.info(f"Cleaned up directory: {dir_path}")
