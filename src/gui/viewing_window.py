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
                 fit_to_window: bool = True, fullscreen_mode: bool = False):
        """
        Initialize the viewing window.

        Args:
            image_manager: Manager for image operations
            fit_to_window: Automatically fit images to window when loaded
            fullscreen_mode: Launch in fullscreen mode with no UI elements
        """
        super().__init__()

        # Store manager
        self.image_manager = image_manager
        self.fit_to_window = fit_to_window
        self.fullscreen_mode = fullscreen_mode

        # Shuffle state
        self.is_shuffled = False
        self.original_image_order = []

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

    def _setup_ui(self) -> None:
        """Setup the minimal user interface."""
        if self.fullscreen_mode:
            # Fullscreen mode - just the image viewer, no borders or UI
            self.image_viewer = ImageViewer(fit_to_window=self.fit_to_window, borderless=True)
            self.setCentralWidget(self.image_viewer)

            # Remove window frame and decorations
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

            # Set completely transparent background
            self.setStyleSheet("QMainWindow { background-color: transparent; }")
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

    def _load_current_image(self) -> None:
        """Load and display the current image."""
        current_image = self.image_manager.get_current_image()
        if current_image:
            self.image_viewer.set_image(current_image)
        else:
            self.image_viewer.clear_image()

    @pyqtSlot(Path, int, int)
    def _on_image_changed(self, image_path: Path, current: int,
                          total: int) -> None:
        """Handle image change events."""
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
            QMessageBox.warning(
                self, "Favorites Failed",
                ("Failed to add the image to favorites. Check the log file "
                 "for details.")
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
        self._load_current_image()  # Reload current image with reset pan position

    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Left:
            self.image_manager.previous_image()
        elif event.key() == Qt.Key.Key_Right:
            self.image_manager.next_image()
        elif event.key() == Qt.Key.Key_Space:
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
            # Escape key - exit viewing mode
            self.close()
        elif event.key() == Qt.Key.Key_F11:
            # F11 key - toggle fullscreen
            self._toggle_fullscreen()
        elif event.key() == Qt.Key.Key_C:
            # C key - reset pan position memory
            self._reset_pan_memory()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Emit signal for cleanup
        self.aboutToQuit.emit()
        super().closeEvent(event)
