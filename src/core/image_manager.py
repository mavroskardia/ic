"""
Image Manager for the Image Classifier application.

This module handles image loading, navigation, and caching.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
import shutil

from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QObject, pyqtSignal, QRect

from utils.file_utils import (
    get_app_data_dir, get_image_files, get_file_size_mb
)


class ImageManager(QObject):
    """
    Manages image loading, navigation, and caching.

    This class handles:
    - Recursive image discovery
    - Image navigation (next/previous)
    - Image caching for performance
    - Progress tracking
    """

    # Signals
    image_changed = pyqtSignal(Path, int, int)  # path, current, total
    progress_updated = pyqtSignal(int, int, float)  # current, total, %

    def __init__(self, root_directory: Path, cache_size: int = 10,
                 max_files: int = None):
        """
        Initialize the Image Manager.

        Args:
            root_directory: Root directory containing images
            cache_size: Maximum number of images to cache
            max_files: Optional max images to discover (for performance)
        """
        super().__init__()
        self.root_directory = root_directory
        self.cache_size = cache_size
        self.max_files = max_files

        # Image collection
        self.image_files: List[Path] = []
        self.current_index: int = -1

        # Cache
        self.image_cache: Dict[Path, QPixmap] = {}
        self.cache_order: List[Path] = []

        # Favorites, deleted, and cropped directories - use app data directory
        app_data_dir = get_app_data_dir()
        self.favorites_directory = app_data_dir / "favorites"
        self.deleted_directory = app_data_dir / "deleted"
        self.cropped_directory = app_data_dir / "cropped"
        self.favorites_counter = 0
        self.cropped_counter = 0

        # Initialize
        self._setup_logging()
        self._discover_images()
        self._initialize_favorites()
        self._initialize_cropped()

    def _setup_logging(self) -> None:
        """Setup logging for the image manager."""
        self.logger = logging.getLogger(__name__)

    def _initialize_favorites(self) -> None:
        """Initialize the favorites directory and counter."""
        try:
            # Create favorites directory if it doesn't exist
            self.favorites_directory.mkdir(parents=True, exist_ok=True)

            # Find the highest numbered file to set the counter
            existing_files = list(self.favorites_directory.glob("*"))
            if existing_files:
                # Extract numbers from filenames and find the maximum
                max_num = -1
                for file_path in existing_files:
                    try:
                        # Try to extract number from filename
                        # (assuming format like "0.jpg", "1.png", etc.)
                        stem = file_path.stem
                        if stem.isdigit():
                            max_num = max(max_num, int(stem))
                    except (ValueError, AttributeError):
                        continue
                self.favorites_counter = max_num + 1
            else:
                self.favorites_counter = 0

            self.logger.info(
                f"Favorites directory initialized with counter at "
                f"{self.favorites_counter}"
            )
        except Exception as e:
            self.logger.error(f"Error initializing favorites directory: {e}")
            self.favorites_counter = 0

    def _discover_images(self) -> None:
        """Discover all image files in the root directory."""
        try:
            self.logger.info(f"Discovering images in {self.root_directory}")
            if self.max_files:
                self.logger.info(
                    f"Limiting discovery to {self.max_files} files"
                )

            self.image_files = get_image_files(self.root_directory,
                                               self.max_files)
            self.logger.info(f"Discovered {len(self.image_files)} images")

            if self.image_files:
                self.current_index = 0
                self._emit_progress_update()
            else:
                self.logger.warning("No images found in directory")

        except Exception as e:
            self.logger.error(f"Error discovering images: {e}")
            self.image_files = []

    def _emit_progress_update(self) -> None:
        """Emit progress update signal."""
        if self.image_files:
            current = self.current_index + 1
            total = len(self.image_files)
            percentage = (current / total) * 100
            self.progress_updated.emit(current, total, percentage)

    def get_current_image(self) -> Optional[QPixmap]:
        """
        Get the current image.

        Returns:
            QPixmap of current image or None if no images
        """
        if not self.image_files or self.current_index < 0:
            return None

        current_path = self.image_files[self.current_index]
        return self._load_image(current_path)

    def get_current_path(self) -> Optional[Path]:
        """
        Get the path of the current image.

        Returns:
            Path to current image or None if no images
        """
        if not self.image_files or self.current_index < 0:
            return None
        return self.image_files[self.current_index]

    def next_image(self) -> bool:
        """
        Move to the next image.

        Returns:
            True if successful, False if at the end
        """
        if not self.image_files:
            return False

        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._emit_progress_update()
            self._emit_image_changed()
            return True
        return False

    def previous_image(self) -> bool:
        """
        Move to the previous image.

        Returns:
            True if successful, False if at the beginning
        """
        if not self.image_files:
            return False

        if self.current_index > 0:
            self.current_index -= 1
            self._emit_progress_update()
            self._emit_image_changed()
            return True
        return False

    def go_to_image(self, index: int) -> bool:
        """
        Go to a specific image by index.

        Args:
            index: Target image index (0-based)

        Returns:
            True if successful, False if index is invalid
        """
        if not self.image_files:
            return False

        if 0 <= index < len(self.image_files):
            self.current_index = index
            self._emit_progress_update()
            self._emit_image_changed()
            return True
        return False

    def _emit_image_changed(self) -> None:
        """Emit image changed signal."""
        if self.image_files and self.current_index >= 0:
            current_path = self.image_files[self.current_index]
            self.image_changed.emit(
                current_path,
                self.current_index + 1,
                len(self.image_files)
            )

    def _load_image(self, image_path: Path) -> Optional[QPixmap]:
        """
        Load an image from disk or cache.

        Args:
            image_path: Path to the image file

        Returns:
            QPixmap of the image or None if loading failed
        """
        # Check cache first
        if image_path in self.image_cache:
            self._update_cache_order(image_path)
            return self.image_cache[image_path]

        try:
            # Load image from disk
            pixmap = QPixmap(str(image_path))

            if pixmap.isNull():
                self.logger.error(f"Failed to load image: {image_path}")
                return None

            # Add to cache
            self._add_to_cache(image_path, pixmap)

            # Log file size for debugging
            size_mb = get_file_size_mb(image_path)
            self.logger.debug(f"Loaded image: {image_path} ({size_mb:.2f} MB)")

            return pixmap

        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None

    def _add_to_cache(self, image_path: Path, pixmap: QPixmap) -> None:
        """
        Add an image to the cache.

        Args:
            image_path: Path to the image
            pixmap: The image pixmap
        """
        # Remove oldest cached image if cache is full
        if len(self.image_cache) >= self.cache_size:
            oldest_path = self.cache_order.pop(0)
            del self.image_cache[oldest_path]

        # Add new image to cache
        self.image_cache[image_path] = pixmap
        self.cache_order.append(image_path)

    def _update_cache_order(self, image_path: Path) -> None:
        """
        Update cache order when an image is accessed.

        Args:
            image_path: Path to the accessed image
        """
        if image_path in self.cache_order:
            self.cache_order.remove(image_path)
        self.cache_order.append(image_path)

    def get_image_info(self) -> Tuple[int, int, float]:
        """
        Get current image information.

        Returns:
            Tuple of (current_index, total_count, percentage)
        """
        if not self.image_files:
            return 0, 0, 0.0

        current = self.current_index + 1
        total = len(self.image_files)
        percentage = (current / total) * 100
        return current, total, percentage

    def get_total_count(self) -> int:
        """
        Get total number of images.

        Returns:
            Total count of images
        """
        return len(self.image_files)

    def delete_current_image(self) -> bool:
        """
        Delete the current image by moving it to a 'deleted' subdirectory.

        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.image_files or self.current_index < 0:
            self.logger.warning("No current image to delete")
            return False

        current_path = self.image_files[self.current_index]

        try:
            # Create deleted directory if it doesn't exist
            self.deleted_directory.mkdir(parents=True, exist_ok=True)

            # Create destination path
            dest = self.deleted_directory / current_path.name

            # Handle filename conflicts by adding a number suffix
            counter = 1
            while dest.exists():
                stem = current_path.stem
                suffix = current_path.suffix
                dest = self.deleted_directory / f"{stem}_{counter}{suffix}"
                counter += 1

            # Move the file (use shutil.move for cross-drive compatibility)
            shutil.move(str(current_path), str(dest))

            # Remove from cache if present
            if current_path in self.image_cache:
                del self.image_cache[current_path]
                if current_path in self.cache_order:
                    self.cache_order.remove(current_path)

            # Remove from image files list
            self.image_files.pop(self.current_index)

            # Adjust current index if necessary
            if self.current_index >= len(self.image_files):
                self.current_index = max(0, len(self.image_files) - 1)

            # Emit progress update
            self._emit_progress_update()

            self.logger.info(f"Image moved to deleted directory: {dest}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to delete image {current_path}: {e}",
                exc_info=True
            )
            return False

    def refresh_collection(self) -> None:
        """Refresh the image collection from disk."""
        self._discover_images()
        self._clear_cache()

    def _clear_cache(self) -> None:
        """Clear the image cache."""
        self.image_cache.clear()
        self.cache_order.clear()

    def get_image_paths(self) -> List[Path]:
        """
        Get all image paths.

        Returns:
            List of all image paths
        """
        return self.image_files.copy()

    def add_to_favorites(self) -> bool:
        """
        Copy the current image to the favorites folder with incremental naming.

        Returns:
            True if successful, False otherwise
        """
        if not self.image_files or self.current_index < 0:
            self.logger.warning("No current image to add to favorites")
            return False

        current_path = self.image_files[self.current_index]

        try:
            # Create destination path with current counter
            dest_path = self.favorites_directory / (
                f"{self.favorites_counter}{current_path.suffix}"
            )

            # Copy the file
            shutil.copy2(current_path, dest_path)

            # Increment counter for next time
            self.favorites_counter += 1

            self.logger.info(f"Image added to favorites: {dest_path}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to add image to favorites {current_path}: {e}",
                exc_info=True
            )
            return False

    def _initialize_cropped(self) -> None:
        """Initialize the cropped directory and counter."""
        try:
            # Create cropped directory if it doesn't exist
            self.cropped_directory.mkdir(parents=True, exist_ok=True)

            # Find the highest numbered file to set the counter
            existing_files = list(self.cropped_directory.glob("*"))
            if existing_files:
                # Extract numbers from filenames and find the maximum
                max_num = -1
                for file_path in existing_files:
                    try:
                        # Extract number from filename (e.g., "123.jpg" -> 123)
                        filename = file_path.stem
                        if filename.isdigit():
                            num = int(filename)
                            max_num = max(max_num, num)
                    except (ValueError, AttributeError):
                        continue

                self.cropped_counter = max_num + 1
            else:
                self.cropped_counter = 0

            self.logger.info(
                f"Initialized cropped directory: {self.cropped_directory}"
            )
            self.logger.info(f"Cropped counter set to: {self.cropped_counter}")

        except Exception as e:
            self.logger.error(f"Failed to initialize cropped directory: {e}")

    def crop_current_image(self, crop_rect: QRect) -> bool:
        """
        Crop the current image and save it to the cropped directory.

        Args:
            crop_rect: Rectangle defining the crop area in image coordinates

        Returns:
            True if cropping was successful, False otherwise
        """
        if not self.image_files or self.current_index < 0:
            self.logger.warning("No current image to crop")
            return False

        current_path = self.image_files[self.current_index]

        try:
            # Load the original image
            original_pixmap = QPixmap(str(current_path))
            if original_pixmap.isNull():
                self.logger.error(f"Failed to load image: {current_path}")
                return False

            # Validate crop rectangle
            if (crop_rect.x() < 0 or crop_rect.y() < 0 or
                    crop_rect.right() > original_pixmap.width() or
                    crop_rect.bottom() > original_pixmap.height() or
                    crop_rect.width() <= 0 or crop_rect.height() <= 0):
                self.logger.error(f"Invalid crop rectangle: {crop_rect}")
                return False

            # Create cropped pixmap
            cropped_pixmap = original_pixmap.copy(crop_rect)
            if cropped_pixmap.isNull():
                self.logger.error("Failed to create cropped pixmap")
                return False

            # Create destination path with current counter
            dest_path = self.cropped_directory / (
                f"{self.cropped_counter}{current_path.suffix}"
            )

            # Save the cropped image
            if not cropped_pixmap.save(str(dest_path)):
                self.logger.error(f"Failed to save cropped image: {dest_path}")
                return False

            # Increment counter for next time
            self.cropped_counter += 1

            self.logger.info(f"Image cropped and saved: {dest_path}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to crop image {current_path}: {e}",
                exc_info=True
            )
            return False

    def discover_more_images(self, additional_limit: int = 1000) -> int:
        """
        Discover additional images beyond the initial limit.

        Args:
            additional_limit: Number of additional images to discover

        Returns:
            Number of new images discovered
        """
        if not self.max_files:
            self.logger.info(
                "No file limit set, all images already discovered"
            )
            return 0

        try:
            # Get current count
            current_count = len(self.image_files)

            # Discover more images with a higher limit
            new_limit = current_count + additional_limit
            self.logger.info(f"Discovering more images (limit: {new_limit})")

            # Temporarily update max_files and rediscover
            old_max_files = self.max_files
            self.max_files = new_limit
            self._discover_images()
            self.max_files = old_max_files

            new_count = len(self.image_files)
            discovered = new_count - current_count

            self.logger.info(f"Discovered {discovered} additional images")
            return discovered

        except Exception as e:
            self.logger.error(f"Error discovering more images: {e}")
            return 0
