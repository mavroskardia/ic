"""
File utility functions for the Image Classifier application.

This module provides platform-independent file operations and path handling.
"""

import os
import platform
from pathlib import Path
from typing import List


def get_app_data_dir() -> Path:
    """
    Get the platform-specific application data directory.

    Returns:
        Path: Path to the application data directory
    """
    system = platform.system().lower()

    if system == "windows":
        # Windows: %APPDATA%\ImageClassifier
        app_data = os.environ.get("APPDATA", "")
        if not app_data:
            app_data = os.path.expanduser("~\\AppData\\Roaming")
        return Path(app_data) / "ImageClassifier"

    elif system == "linux":
        # Linux: ~/.config/ImageClassifier
        return Path.home() / ".config" / "ImageClassifier"

    elif system == "darwin":
        # macOS: ~/Library/Application Support/ImageClassifier
        app_support = Path.home() / "Library" / "Application Support"
        return app_support / "ImageClassifier"

    else:
        # Fallback: ~/.ImageClassifier
        return Path.home() / ".ImageClassifier"


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory
    """
    path.mkdir(parents=True, exist_ok=True)


def get_image_files(directory: Path, max_files: int = None) -> List[Path]:
    """
    Get all image files from a directory and its subdirectories.

    Uses optimized os.scandir() for better performance with large directories.

    Args:
        directory: Root directory to search
        max_files: Optional max files to return (for early termination)

    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []

    def _scan_directory(path: Path) -> None:
        """Recursively scan directory using os.scandir()."""
        if max_files and len(image_files) >= max_files:
            return

        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if max_files and len(image_files) >= max_files:
                        break

                    if entry.is_file():
                        # Check extension during traversal
                        if entry.name.lower().endswith(
                                tuple(image_extensions)):
                            image_files.append(Path(entry.path))
                    elif entry.is_dir():
                        # Recursively scan subdirectories
                        _scan_directory(Path(entry.path))
        except (PermissionError, OSError):
            # Skip directories we can't access
            pass

    _scan_directory(directory)
    return sorted(image_files)


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image file.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is an image, False otherwise
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    return file_path.suffix.lower() in image_extensions


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in MB
    """
    try:
        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def create_backup_path(original_path: Path) -> Path:
    """
    Create a backup path for a file.

    Args:
        original_path: Path to the original file

    Returns:
        Path for the backup file
    """
    stem = original_path.stem
    suffix = original_path.suffix
    backup_dir = original_path.parent / "backup"
    backup_dir.mkdir(exist_ok=True)

    counter = 1
    while True:
        backup_name = f"{stem}_backup_{counter}{suffix}"
        backup_path = backup_dir / backup_name
        if not backup_path.exists():
            return backup_path
        counter += 1
