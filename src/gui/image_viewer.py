"""
Image Viewer widget for the Image Classifier application.

This module provides a widget for displaying images with zoom and pan capabilities.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent, QKeyEvent


class ImageViewer(QWidget):
    """
    Widget for displaying images with zoom and pan capabilities.

    Features:
    - Image display with aspect ratio preservation
    - Zoom in/out with mouse wheel
    - Pan with mouse drag
    - Fit to window option
    """

    # Signals
    image_clicked = pyqtSignal(int, int)  # x, y coordinates

    def __init__(self, fit_to_window: bool = True, borderless: bool = False):
        """
        Initialize the image viewer.

        Args:
            fit_to_window: Automatically fit images to window when loaded
            borderless: Remove all borders, controls, and UI elements
        """
        super().__init__()

        # Image data
        self.original_pixmap: QPixmap = None
        self.display_pixmap: QPixmap = None
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # Settings
        self.fit_to_window = fit_to_window
        self.borderless = borderless

        # Mouse interaction state
        self.is_panning = False
        self.last_mouse_pos = None
        self.crop_mode_active = False

        # Pan position memory
        self.saved_pan_offset_x = 0
        self.saved_pan_offset_y = 0

        # Zoom level memory
        self.saved_zoom_factor = 1.0
        self.user_has_zoomed = False  # Track if user has manually zoomed

        # Setup UI
        self._setup_ui()

        # Set focus policy for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        if self.borderless:
            # Borderless mode - just the image label, no controls or borders
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    border: none;
                }
            """)
            self.image_label.setText("")

            # Set the image label as the main widget
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.image_label)

            # Enable mouse tracking for panning
            self.image_label.setMouseTracking(True)
            self.image_label.mousePressEvent = self._mouse_press_event
            self.image_label.mouseMoveEvent = self._mouse_move_event
            self.image_label.mouseReleaseEvent = self._mouse_release_event
            self.image_label.wheelEvent = self._wheel_event

            # No control buttons or zoom label in borderless mode
            self.zoom_in_button = None
            self.zoom_out_button = None
            self.fit_button = None
            self.reset_button = None
            self.zoom_label = None

        else:
            # Regular mode with controls and borders
            layout = QVBoxLayout(self)

            # Image display label
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setMinimumSize(400, 300)
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    border: 2px dashed #cccccc;
                    color: #666666;
                }
            """)
            self.image_label.setText("No image loaded")
            layout.addWidget(self.image_label)

            # Control buttons
            button_layout = QHBoxLayout()

            # Zoom controls
            self.zoom_in_button = QPushButton("Zoom In")
            self.zoom_in_button.setShortcut("Ctrl++")
            self.zoom_in_button.clicked.connect(self._zoom_in)
            button_layout.addWidget(self.zoom_in_button)

            self.zoom_out_button = QPushButton("Zoom Out")
            self.zoom_out_button.setShortcut("Ctrl+-")
            self.zoom_out_button.clicked.connect(self._zoom_out)
            button_layout.addWidget(self.zoom_out_button)

            self.fit_button = QPushButton("Fit to Window")
            self.fit_button.setShortcut("Ctrl+0")
            self.fit_button.clicked.connect(self._fit_to_window)
            button_layout.addWidget(self.fit_button)

            self.reset_button = QPushButton("Reset View")
            self.reset_button.setShortcut("Ctrl+R")
            self.reset_button.clicked.connect(self._reset_view)
            button_layout.addWidget(self.reset_button)

            button_layout.addStretch()

            # Zoom info
            self.zoom_label = QLabel("Zoom: 100%")
            button_layout.addWidget(self.zoom_label)

            layout.addLayout(button_layout)

            # Enable mouse tracking for panning
            self.image_label.setMouseTracking(True)
            self.image_label.mousePressEvent = self._mouse_press_event
            self.image_label.mouseMoveEvent = self._mouse_move_event
            self.image_label.mouseReleaseEvent = self._mouse_release_event
            self.image_label.wheelEvent = self._wheel_event

    def set_image(self, pixmap: QPixmap) -> None:
        """
        Set the image to display.

        Args:
            pixmap: The image to display
        """
        if pixmap is None:
            self.clear_image()
            return

        # Save current pan position and zoom level before changing images
        self._save_pan_position()
        self._save_zoom_level()

        self.original_pixmap = pixmap

        # Auto-fit to window if enabled, but only if user hasn't manually zoomed
        if self.fit_to_window and not self.user_has_zoomed:
            self._fit_to_window()
        else:
            # Restore zoom level if user has manually zoomed or
            # fit_to_window is disabled
            self._restore_zoom_level()

        # Restore saved pan position
        self._restore_pan_position()

        self._update_display()

    def clear_image(self) -> None:
        """Clear the displayed image."""
        self.original_pixmap = None
        self.display_pixmap = None
        if self.borderless:
            self.image_label.setText("")
        else:
            self.image_label.setText("No image loaded")
            if self.zoom_label:
                self.zoom_label.setText("Zoom: 100%")
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

    def _update_display(self) -> None:
        """Update the displayed image with current zoom and pan."""
        if self.original_pixmap is None:
            return

        # Calculate scaled size
        scaled_width = int(self.original_pixmap.width() * self.zoom_factor)
        scaled_height = int(self.original_pixmap.height() * self.zoom_factor)

        # Create scaled pixmap
        self.display_pixmap = self.original_pixmap.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Apply pan offset
        self._apply_pan_offset()

        # Update zoom label
        if self.zoom_label:
            self.zoom_label.setText(f"Zoom: {self.zoom_factor * 100:.0f}%")

        # Update button states
        if not self.borderless:
            self._update_button_states()

    def _apply_pan_offset(self) -> None:
        """Apply pan offset to the display pixmap."""
        if self.display_pixmap is None:
            return

        # Create a new pixmap with the same size as the label
        label_size = self.image_label.size()
        final_pixmap = QPixmap(label_size)
        final_pixmap.fill(Qt.GlobalColor.transparent)

        # Create painter
        painter = QPainter(final_pixmap)

        # Calculate position to center the image
        x = (label_size.width() - self.display_pixmap.width()) // 2
        y = (label_size.height() - self.display_pixmap.height()) // 2

        # Apply pan offset
        x += self.pan_offset_x
        y += self.pan_offset_y

        # Convert to integers for drawPixmap
        x = int(x)
        y = int(y)

        # Draw the image
        painter.drawPixmap(x, y, self.display_pixmap)
        painter.end()

        # Set the final pixmap
        self.image_label.setPixmap(final_pixmap)

    def _update_button_states(self) -> None:
        """Update the enabled state of control buttons."""
        if self.borderless:
            return  # No buttons in borderless mode

        has_image = self.original_pixmap is not None
        can_zoom_in = has_image and self.zoom_factor < 5.0
        can_zoom_out = has_image and self.zoom_factor > 0.1
        can_reset = has_image and (self.zoom_factor != 1.0 or
                                   self.pan_offset_x != 0 or
                                   self.pan_offset_y != 0)

        self.zoom_in_button.setEnabled(can_zoom_in)
        self.zoom_out_button.setEnabled(can_zoom_out)
        self.fit_button.setEnabled(has_image)
        self.reset_button.setEnabled(can_reset)

    def _zoom_in(self) -> None:
        """Zoom in by 25%."""
        if self.original_pixmap is None:
            return

        self.zoom_factor *= 1.25
        self.zoom_factor = min(self.zoom_factor, 5.0)
        self.user_has_zoomed = True  # Mark that user has manually zoomed
        self._update_display()
        self._save_pan_position()
        self._save_zoom_level()

    def _zoom_out(self) -> None:
        """Zoom out by 25%."""
        if self.original_pixmap is None:
            return

        self.zoom_factor /= 1.25
        self.zoom_factor = max(self.zoom_factor, 0.1)
        self.user_has_zoomed = True  # Mark that user has manually zoomed
        self._update_display()
        self._save_pan_position()
        self._save_zoom_level()

    def _fit_to_window(self) -> None:
        """Fit the image to the window size."""
        if self.original_pixmap is None:
            return

        # Calculate zoom factor to fit the image
        label_size = self.image_label.size()
        image_size = self.original_pixmap.size()

        scale_x = label_size.width() / image_size.width()
        scale_y = label_size.height() / image_size.height()

        # Use the smaller scale to ensure the image fits
        self.zoom_factor = min(scale_x, scale_y) * 0.9  # 90% to add some margin

        # Reset pan offset
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self._update_display()

    def _reset_view(self) -> None:
        """Reset zoom and pan to default values."""
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.user_has_zoomed = False  # Reset manual zoom flag
        self._update_display()
        self._save_zoom_level()

    def _mouse_press_event(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton and not self.crop_mode_active:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        # Emit click signal
        self.image_clicked.emit(event.position().x(), event.position().y())

    def _mouse_move_event(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        if (self.is_panning and self.last_mouse_pos is not None and
                not self.crop_mode_active):
            # Calculate pan offset
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()

            self.pan_offset_x += dx
            self.pan_offset_y += dy

            self.last_mouse_pos = event.pos()

            # Update display
            self._update_display()

            # Save pan position for next image
            self._save_pan_position()

    def _mouse_release_event(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton and not self.crop_mode_active:
            self.is_panning = False
            self.last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _wheel_event(self, event: QWheelEvent) -> None:
        """Handle mouse wheel events for zooming."""
        if self.original_pixmap is None:
            return

        # Get zoom factor from wheel delta
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self._zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self._zoom_out()
        elif event.key() == Qt.Key.Key_0:
            self._fit_to_window()
        elif event.key() == Qt.Key.Key_R:
            self._reset_view()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        if self.original_pixmap is not None:
            self._update_display()

    def get_current_zoom(self) -> float:
        """Get the current zoom factor."""
        return self.zoom_factor

    def set_fit_to_window(self, enabled: bool) -> None:
        """
        Set whether images should automatically fit to window.

        Args:
            enabled: True to enable auto-fit, False to disable
        """
        self.fit_to_window = enabled

    def get_image_info(self) -> dict:
        """Get information about the current image."""
        if self.original_pixmap is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "width": self.original_pixmap.width(),
            "height": self.original_pixmap.height(),
            "zoom_factor": self.zoom_factor,
            "pan_offset_x": self.pan_offset_x,
            "pan_offset_y": self.pan_offset_y,
            "fit_to_window": self.fit_to_window
        }

    def get_image_position(self) -> tuple:
        """
        Get the position and size of the displayed image within the widget.

        Returns:
            Tuple of (x, y, width, height) of the image in widget coordinates
        """
        if self.display_pixmap is None:
            return (0, 0, 0, 0)

        label_size = self.image_label.size()

        # Calculate position to center the image
        x = (label_size.width() - self.display_pixmap.width()) // 2
        y = (label_size.height() - self.display_pixmap.height()) // 2

        # Apply pan offset
        x += self.pan_offset_x
        y += self.pan_offset_y

        return (x, y, self.display_pixmap.width(), self.display_pixmap.height())

    def convert_widget_to_image_coords(self, widget_rect: QRect) -> QRect:
        """
        Convert a rectangle from widget coordinates to image coordinates.

        Args:
            widget_rect: Rectangle in widget coordinates

        Returns:
            Rectangle in image coordinates
        """
        if self.original_pixmap is None or self.display_pixmap is None:
            return QRect()

        # Get image position in widget
        img_x, img_y, img_width, img_height = self.get_image_position()

        # Calculate scale factors
        scale_x = self.original_pixmap.width() / img_width
        scale_y = self.original_pixmap.height() / img_height

        # Convert widget coordinates to image coordinates
        # First, adjust for image position
        adj_x = widget_rect.x() - img_x
        adj_y = widget_rect.y() - img_y

        # Then scale to image coordinates
        img_rect_x = int(adj_x * scale_x)
        img_rect_y = int(adj_y * scale_y)
        img_rect_width = int(widget_rect.width() * scale_x)
        img_rect_height = int(widget_rect.height() * scale_y)

        # Ensure coordinates are within image bounds
        img_rect_x = max(0, min(img_rect_x, self.original_pixmap.width()))
        img_rect_y = max(0, min(img_rect_y, self.original_pixmap.height()))
        img_rect_width = max(0, min(img_rect_width, self.original_pixmap.width() - img_rect_x))
        img_rect_height = max(0, min(img_rect_height, self.original_pixmap.height() - img_rect_y))

        return QRect(img_rect_x, img_rect_y, img_rect_width, img_rect_height)

    def set_crop_mode(self, active: bool) -> None:
        """
        Set the crop mode state.

        Args:
            active: True to enable crop mode (disable panning), False to disable
        """
        self.crop_mode_active = active

        # If crop mode is being disabled and we were panning, stop panning
        if not active and self.is_panning:
            self.is_panning = False
            self.last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _save_pan_position(self) -> None:
        """Save the current pan position for restoration on next image."""
        self.saved_pan_offset_x = self.pan_offset_x
        self.saved_pan_offset_y = self.pan_offset_y

    def _restore_pan_position(self) -> None:
        """Restore the saved pan position."""
        self.pan_offset_x = self.saved_pan_offset_x
        self.pan_offset_y = self.saved_pan_offset_y

    def _save_zoom_level(self) -> None:
        """Save the current zoom level for restoration on next image."""
        self.saved_zoom_factor = self.zoom_factor

    def _restore_zoom_level(self) -> None:
        """Restore the saved zoom level."""
        self.zoom_factor = self.saved_zoom_factor

    def reset_pan_memory(self) -> None:
        """Reset the pan position memory to center."""
        self.saved_pan_offset_x = 0
        self.saved_pan_offset_y = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        # Also reset zoom memory
        self.saved_zoom_factor = 1.0
        self.zoom_factor = 1.0
        self.user_has_zoomed = False  # Reset manual zoom flag
