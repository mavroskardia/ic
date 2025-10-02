"""
Viewing Window for the Image Classifier application.

This module provides a minimal interface for viewing images with basic
navigation, favorites, and delete functionality. No training or rating
capabilities.
"""

import logging
import random
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QMessageBox
)
from typing import Optional
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer
from PyQt6.QtGui import QKeySequence, QFont, QAction

from .image_viewer import ImageViewer
from core.image_manager import ImageManager


class ViewingWindow(QMainWindow):
    """
    Minimal viewing window for the Image Classifier.

    This window provides:
    - Image display with zoom and pan
    - Basic navigation (left/right arrows)
    - Favorites functionality (F key)
    - Delete functionality (Delete key)
    - Shuffle functionality (S key)
    - Escape to exit
    - Minimal interface with thin border
    """

    # Signals
    aboutToQuit = pyqtSignal()

    def __init__(self, image_manager: ImageManager,
                 fit_to_window: bool = True, fullscreen_mode: bool = False,
                 slideshow_interval: float = 5.0,
                 transition_duration: float = 0.8,
                 background_color: Optional[str] = None,
                 rotate_deg: float = 0.0):
        """
        Initialize the viewing window.

        Args:
            image_manager: Manager for image operations
            fit_to_window: Automatically fit images to window when loaded
            fullscreen_mode: Launch in fullscreen mode with no UI elements
            slideshow_interval: Interval in seconds for slideshow mode
        """
        super().__init__()

        # Store manager
        self.image_manager = image_manager
        self.fit_to_window = fit_to_window
        self.fullscreen_mode = fullscreen_mode
        self.slideshow_interval = slideshow_interval
        self.transition_duration = transition_duration
        self.background_color = background_color
        self.rotate_deg = rotate_deg

        # Shuffle state
        self.is_shuffled = False
        self.original_image_order = []

        # Slideshow state
        self.slideshow_active = False
        self.slideshow_timer = QTimer()
        self.slideshow_timer.timeout.connect(self._slideshow_next_image)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize UI
        self._setup_ui()
        self._setup_connections()

        if not self.fullscreen_mode:
            self._setup_menu()

        # Load initial image
        self._load_current_image()

        # Show in fullscreen if requested
        if self.fullscreen_mode:
            self.showFullScreen()

        self.logger.info("Viewing window initialized")

        # Internal: whether the next image load should use a smooth transition
        self._slideshow_transition = False

    def _setup_ui(self) -> None:
        """Setup the minimal user interface."""
        if self.fullscreen_mode:
            # Fullscreen mode - just the image viewer, no borders or UI
            self.image_viewer = ImageViewer(
                fit_to_window=self.fit_to_window, borderless=True
            )
            self.setCentralWidget(self.image_viewer)

            # Remove window frame and decorations
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

            # Set background: transparent by default, or user-specified color
            if self.background_color:
                style_bg = (
                    "QMainWindow {{ background-color: {}; }}"
                    .format(self.background_color)
                )
                self.setStyleSheet(style_bg)
            else:
                self.setStyleSheet(
                    "QMainWindow { background-color: transparent; }"
                )
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

            # Window properties for fullscreen
            self.setWindowTitle("Image Viewer - Fullscreen")
        else:
            # Regular minimal mode with thin border
            central_widget = QWidget()
            self.setCentralWidget(central_widget)

            # Set minimal styling with thin border
            central_widget.setStyleSheet("""
                QWidget {
                    border: 1px solid #cccccc;
                    background-color: #f8f8f8;
                }
            """)

            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins

            # Image viewer (takes most of the space)
            self.image_viewer = ImageViewer(fit_to_window=self.fit_to_window)
            main_layout.addWidget(self.image_viewer)

            # Minimal status bar at bottom
            status_layout = QHBoxLayout()

            # Image counter
            self.image_counter = QLabel("Image 0 of 0")
            self.image_counter.setFont(QFont("Arial", 9))
            status_layout.addWidget(self.image_counter)

            status_layout.addStretch()

            # Shuffle indicator
            self.shuffle_label = QLabel("")
            self.shuffle_label.setFont(QFont("Arial", 9))
            self.shuffle_label.setStyleSheet("color: #666666;")
            status_layout.addWidget(self.shuffle_label)

            main_layout.addLayout(status_layout)

            # Window properties
            self.setWindowTitle("Image Viewer - Minimal Mode")
            self.setMinimumSize(800, 600)
            self.resize(1000, 700)

        # Set focus policy for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _setup_connections(self) -> None:
        """Setup signal connections between components."""
        # Image manager signals
        self.image_manager.image_changed.connect(self._on_image_changed)
        self.image_manager.progress_updated.connect(self._on_progress_updated)

    def _setup_menu(self) -> None:
        """Setup a minimal menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Shuffle action
        shuffle_action = QAction("&Shuffle Images", self)
        shuffle_action.setShortcut(QKeySequence("S"))
        shuffle_action.triggered.connect(self._shuffle_images)
        view_menu.addAction(shuffle_action)

        # Unshuffle action
        unshuffle_action = QAction("&Unshuffle Images", self)
        unshuffle_action.setShortcut(QKeySequence("U"))
        unshuffle_action.triggered.connect(self._unshuffle_images)
        view_menu.addAction(unshuffle_action)

        view_menu.addSeparator()

        # Delete current image action
        delete_action = QAction("&Delete Current Image", self)
        delete_action.setShortcut(QKeySequence("Delete"))
        delete_action.triggered.connect(self._delete_current_image)
        view_menu.addAction(delete_action)

        # Add to favorites action
        favorite_action = QAction("&Add to Favorites", self)
        favorite_action.setShortcut(QKeySequence("F"))
        favorite_action.triggered.connect(self._add_to_favorites)
        view_menu.addAction(favorite_action)

        view_menu.addSeparator()

        # Toggle fullscreen action
        fullscreen_action = QAction("Toggle &Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Reset pan memory action
        reset_pan_action = QAction("Reset &Pan Memory", self)
        reset_pan_action.setShortcut(QKeySequence("C"))
        reset_pan_action.triggered.connect(self._reset_pan_memory)
        view_menu.addAction(reset_pan_action)

    def _load_current_image(self, smooth_transition: bool = False) -> None:
        """Load and display the current image."""
        current_image = self.image_manager.get_current_image()
        if current_image:
            self.image_viewer.set_image(current_image, smooth_transition)
        else:
            self.image_viewer.clear_image()

    @pyqtSlot(Path, int, int)
    def _on_image_changed(self, image_path: Path, current: int,
                          total: int) -> None:
        """Handle image change events."""
        # If a slideshow-initiated change just occurred, use smooth transition
        if getattr(self, '_slideshow_transition', False):
            self._slideshow_transition = False
            self._load_current_image(smooth_transition=True)
        else:
            self._load_current_image()

        if not self.fullscreen_mode:
            # Update image counter
            self.image_counter.setText(f"Image {current} of {total}")

            # Update window title
            self.setWindowTitle(f"Image Viewer - {image_path.name}")
        else:
            # In fullscreen mode, just update the title
            self.setWindowTitle(f"Image Viewer - {image_path.name}")

    @pyqtSlot(int, int, float)
    def _on_progress_updated(self, current: int, total: int,
                             percentage: float) -> None:
        """Handle progress update events."""
        if not self.fullscreen_mode:
            # Update image counter
            self.image_counter.setText(f"Image {current} of {total}")

    def _shuffle_images(self) -> None:
        """Shuffle the order of images."""
        if not self.image_manager.image_files:
            return

        # Store original order if not already shuffled
        if not self.is_shuffled:
            self.original_image_order = self.image_manager.image_files.copy()
            self.is_shuffled = True

        # Shuffle the image files list
        current_path = self.image_manager.get_current_path()

        # Shuffle the list
        random.shuffle(self.image_manager.image_files)

        # Find the new index of the current image
        try:
            new_index = self.image_manager.image_files.index(current_path)
            self.image_manager.current_index = new_index
        except ValueError:
            # If current image not found, go to first image
            self.image_manager.current_index = 0

        # Update display
        self._load_current_image()
        if not self.fullscreen_mode:
            self._update_shuffle_indicator()

        self.logger.info("Images shuffled")

    def _unshuffle_images(self) -> None:
        """Restore the original order of images."""
        if not self.is_shuffled or not self.original_image_order:
            return

        # Store current image path
        current_path = self.image_manager.get_current_path()

        # Restore original order
        self.image_manager.image_files = self.original_image_order.copy()

        # Find the new index of the current image
        try:
            new_index = self.image_manager.image_files.index(current_path)
            self.image_manager.current_index = new_index
        except ValueError:
            # If current image not found, go to first image
            self.image_manager.current_index = 0

        # Clear shuffle state
        self.is_shuffled = False
        self.original_image_order = []

        # Update display
        self._load_current_image()
        if not self.fullscreen_mode:
            self._update_shuffle_indicator()

        self.logger.info("Images unshuffled")

    def _update_shuffle_indicator(self) -> None:
        """Update the shuffle indicator in the status bar."""
        if self.is_shuffled:
            self.shuffle_label.setText("(Shuffled)")
            self.shuffle_label.setStyleSheet(
                "color: #ff6600; font-weight: bold;")
        else:
            self.shuffle_label.setText("")
            self.shuffle_label.setStyleSheet("color: #666666;")

    def _delete_current_image(self) -> None:
        """Delete the current image by moving it to a deleted directory."""
        current_path = self.image_manager.get_current_path()
        if not current_path:
            return

        # Confirm deletion with user
        reply = QMessageBox.question(
            self, "Delete Image",
            f"Are you sure you want to delete '{current_path.name}'?\n\n"
            "The image will be moved to a 'deleted' subdirectory.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Delete the image
            success = self.image_manager.delete_current_image()
            if success:
                # Load next image if available
                if not self.image_manager.next_image():
                    # If no next image, try previous
                    self.image_manager.previous_image()
                self._load_current_image()
            else:
                QMessageBox.warning(
                    self, "Delete Failed",
                    ("Failed to delete the image. Check the log file for "
                     "details.")
                )

    def _add_to_favorites(self) -> None:
        """Add the current image to favorites with visual feedback."""
        current_path = self.image_manager.get_current_path()
        if not current_path:
            return

        # Add to favorites
        success = self.image_manager.add_to_favorites()
        if success:
            # Provide visual feedback with a flash effect
            self._show_favorites_flash()
        else:
            # Get log file location for user
            from utils.file_utils import get_app_data_dir
            log_file = get_app_data_dir() / "logs" / "image_classifier.log"
            QMessageBox.warning(
                self, "Favorites Failed",
                f"Failed to add the image to favorites. "
                f"Check the log file for details:\n\n{log_file}"
            )

    def _show_favorites_flash(self) -> None:
        """Show a visual flash effect to indicate successful favoriting."""
        # Store the original stylesheet
        original_style = self.image_viewer.styleSheet()

        # Create a flash effect with a white background
        flash_style = """
            QWidget {
                background-color: rgba(255, 255, 255, 200);
                border: 3px solid #4CAF50;
            }
        """

        # Apply the flash style
        self.image_viewer.setStyleSheet(flash_style)

        # Use a timer to restore the original style after a short delay
        QTimer.singleShot(300, lambda: self.image_viewer.setStyleSheet(
            original_style
        ))

    def _toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and windowed mode."""
        if self.isFullScreen():
            # Currently fullscreen, switch to windowed
            self.showNormal()
            self.setWindowFlags(Qt.WindowType.Window)
            self.show()
        else:
            # Currently windowed, switch to fullscreen
            self.showFullScreen()

    def _reset_pan_memory(self) -> None:
        """Reset the pan position memory to center."""
        self.image_viewer.reset_pan_memory()
        # Reload current image with reset pan position
        self._load_current_image()

    def _start_slideshow(self) -> None:
        """Start the slideshow mode."""
        if not self.image_manager.image_files:
            return

        self.slideshow_active = True
        # Convert to milliseconds
        self.slideshow_timer.start(int(self.slideshow_interval * 1000))
        self.logger.info(
            f"Slideshow started with {self.slideshow_interval}s interval")

    def _stop_slideshow(self) -> None:
        """Stop the slideshow mode."""
        self.slideshow_active = False
        self.slideshow_timer.stop()
        self.logger.info("Slideshow stopped")

    def _toggle_slideshow(self) -> None:
        """Toggle slideshow mode on/off."""
        if self.slideshow_active:
            self._stop_slideshow()
        else:
            self._start_slideshow()

    def _slideshow_next_image(self) -> None:
        """Advance to the next image during slideshow."""
        if not self.slideshow_active:
            return

        # Try to go to next image
        # Mark that the next image load should use a smooth transition
        self._slideshow_transition = True
        if not self.image_manager.next_image():
            # If at the end, loop back to the beginning
            self.image_manager.go_to_image(0)
        # Do not call _load_current_image here; let the image_changed signal
        # trigger a single load using the smooth transition flag

    def _navigate_by_offset(self, offset: int) -> None:
        """Navigate by a specific offset from current position."""
        current_index = self.image_manager.current_index
        target_index = current_index + offset

        # Clamp to valid range
        max_index = len(self.image_manager.image_files) - 1
        target_index = max(0, min(target_index, max_index))

        if target_index != current_index:
            self.image_manager.go_to_image(target_index)

    def _go_to_image_prompt(self) -> None:
        """Prompt user to enter an image number to jump to."""
        from PyQt6.QtWidgets import QInputDialog

        total_images = len(self.image_manager.image_files)
        if total_images == 0:
            return

        current_image_num = self.image_manager.current_index + 1

        # Show input dialog
        number, ok = QInputDialog.getInt(
            self,
            "Go to Image",
            f"Enter image number (1-{total_images}):",
            current_image_num,
            1,
            total_images,
            1
        )

        if ok:
            # Convert to 0-based index
            target_index = number - 1
            self.image_manager.go_to_image(target_index)

    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Left:
            # Stop slideshow if active
            if self.slideshow_active:
                self._stop_slideshow()
            self.image_manager.previous_image()
        elif event.key() == Qt.Key.Key_Right:
            # Stop slideshow if active
            if self.slideshow_active:
                self._stop_slideshow()
            self.image_manager.next_image()
        elif event.key() == Qt.Key.Key_Space:
            # Stop slideshow if active
            if self.slideshow_active:
                self._stop_slideshow()
            # Toggle between next/previous
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.image_manager.previous_image()
            else:
                self.image_manager.next_image()
        elif event.key() == Qt.Key.Key_Delete:
            self._delete_current_image()
        elif event.key() == Qt.Key.Key_F:
            # F key - add to favorites
            self._add_to_favorites()
        elif event.key() == Qt.Key.Key_S:
            # S key - shuffle images
            self._shuffle_images()
        elif event.key() == Qt.Key.Key_U:
            # U key - unshuffle images
            self._unshuffle_images()
        elif event.key() == Qt.Key.Key_Escape:
            # Stop slideshow if active
            if self.slideshow_active:
                self._stop_slideshow()
            # Escape key - exit viewing mode
            self.close()
        elif event.key() == Qt.Key.Key_F11:
            # F11 key - toggle fullscreen
            self._toggle_fullscreen()
        elif event.key() == Qt.Key.Key_C:
            # C key - reset pan position memory
            self._reset_pan_memory()
        elif event.key() == Qt.Key.Key_R:
            # Rotate 90 degrees clockwise
            if self.slideshow_active:
                self._stop_slideshow()
            # Increment rotation and normalize to [0, 360)
            try:
                self.rotate_deg = (float(self.rotate_deg) + 90.0) % 360.0
            except Exception:
                self.rotate_deg = 90.0
            # Reload current image with new rotation
            self._load_current_image()
        elif event.key() == Qt.Key.Key_PageUp:
            # Stop slideshow if active
            if self.slideshow_active:
                self._stop_slideshow()
            # Page Up - go back 10 images
            self._navigate_by_offset(-10)
        elif event.key() == Qt.Key.Key_PageDown:
            # Stop slideshow if active
            if self.slideshow_active:
                self._stop_slideshow()
            # Page Down - go forward 10 images
            self._navigate_by_offset(10)
        elif event.key() == Qt.Key.Key_G:
            # G key - go to specific image number
            self._go_to_image_prompt()
        elif event.key() == Qt.Key.Key_A:
            # A key - toggle slideshow (only in fullscreen mode)
            if self.fullscreen_mode:
                self._toggle_slideshow()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Stop slideshow if active
        if self.slideshow_active:
            self._stop_slideshow()

        # Print current image index to console
        current_index = self.image_manager.current_index
        total_count = self.image_manager.get_total_count()
        if total_count > 0:
            print(f"Exiting at image {current_index + 1} of {total_count}")
        # Emit signal for cleanup
        self.aboutToQuit.emit()
        super().closeEvent(event)
