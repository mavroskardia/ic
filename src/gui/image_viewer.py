"""
Image Viewer widget for the Image Classifier application.

This module provides a widget for displaying images with zoom and pan
capabilities.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QWheelEvent, QMouseEvent, QKeyEvent, QTransform
)


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

        # Transition properties
        self._transition_opacity = 1.0
        self.transition_animation = None
        self.next_pixmap = None  # Store next image for transition

        # Double buffering for smooth transitions
        self.visible_buffer = None  # Currently displayed pixmap
        self.hidden_buffer = None   # Buffer being prepared for transition
        self._buffer_opacity = 1.0  # Opacity of the hidden buffer

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

        # Setup overlay label for crossfade transitions
        self._setup_overlay_label()

    def _setup_overlay_label(self) -> None:
        """Create an overlay QLabel used for crossfade transitions."""
        self.overlay_label = QLabel(self)
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
            }
        """)
        self.overlay_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.overlay_label.hide()
        # Opacity effects for base and overlay
        self._base_opacity_effect = None
        self._overlay_opacity_effect = None
        self._fade_out_anim = None
        self._fade_in_anim = None

    def set_image(self, pixmap: QPixmap,
                  smooth_transition: bool = False) -> None:
        """
        Set the image to display.

        Args:
            pixmap: The image to display
            smooth_transition: Whether to use smooth transition animation
        """
        if pixmap is None:
            self.clear_image()
            return

        if (smooth_transition and
                self.original_pixmap is not None):
            # Start crossfade transition
            self._start_crossfade_transition(pixmap)
        else:
            # Direct image change
            self._set_image_direct(pixmap)

    def _compose_from_scaled(self, scaled_pixmap: QPixmap) -> QPixmap:
        """Compose final label-sized pixmap from a scaled image using pan."""
        label_size = self.image_label.size()
        final_pixmap = QPixmap(label_size)
        final_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(final_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        x = (label_size.width() - scaled_pixmap.width()) // 2
        y = (label_size.height() - scaled_pixmap.height()) // 2
        x += int(self.pan_offset_x)
        y += int(self.pan_offset_y)
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()

        return final_pixmap

    def _get_rotate_degrees(self) -> float:
        """Get rotation degrees from parent window if provided."""
        rotate_degrees = 0.0
        parent_win = self.parent()
        if hasattr(parent_win, 'window'):
            parent_win = parent_win.window()
        if parent_win and hasattr(parent_win, 'rotate_deg'):
            try:
                rotate_degrees = float(parent_win.rotate_deg)
            except Exception:
                rotate_degrees = 0.0
        return rotate_degrees

    def _apply_rotation(self, pixmap: QPixmap) -> QPixmap:
        """Return a rotated copy of the pixmap based on configured degrees."""
        deg = self._get_rotate_degrees()
        if pixmap is None or (deg % 360 == 0):
            return pixmap
        transform = QTransform()
        transform.rotate(deg)
        return pixmap.transformed(
            transform, Qt.TransformationMode.SmoothTransformation
        )

    def _ensure_effects(self) -> None:
        """Ensure opacity effects exist on base and overlay labels."""
        if self._base_opacity_effect is None:
            self._base_opacity_effect = QGraphicsOpacityEffect(
                self.image_label
            )
            self._base_opacity_effect.setOpacity(1.0)
            self.image_label.setGraphicsEffect(self._base_opacity_effect)
        if self._overlay_opacity_effect is None:
            self._overlay_opacity_effect = QGraphicsOpacityEffect(
                self.overlay_label
            )
            self._overlay_opacity_effect.setOpacity(0.0)
            self.overlay_label.setGraphicsEffect(self._overlay_opacity_effect)

    def _start_crossfade_transition(self, next_pixmap: QPixmap) -> None:
        """
        Crossfade old image to new using overlaid labels and opacity effects.
        """
        # Prepare scaled/composed pixmap for the next image
        self._set_image_for_transition(next_pixmap)
        next_composed = self._compose_from_scaled(self.display_pixmap)

        # Place next image on overlay
        self.overlay_label.setPixmap(next_composed)
        self.overlay_label.setGeometry(self.image_label.geometry())
        self.overlay_label.raise_()
        self.overlay_label.show()

        # Ensure effects
        self._ensure_effects()
        self._base_opacity_effect.setOpacity(1.0)
        self._overlay_opacity_effect.setOpacity(0.0)

        # Create animations
        if self._fade_out_anim is not None:
            self._fade_out_anim.stop()
        if self._fade_in_anim is not None:
            self._fade_in_anim.stop()

        self._fade_out_anim = QPropertyAnimation(
            self._base_opacity_effect, b"opacity"
        )
        # Duration is configured via parent (ViewingWindow) if present
        duration_ms = 800
        parent_win = self.parent()
        if hasattr(parent_win, 'window'):
            parent_win = parent_win.window()
        if parent_win and hasattr(parent_win, 'transition_duration'):
            try:
                duration_ms = int(float(parent_win.transition_duration) * 1000)
            except Exception:
                duration_ms = 800
        self._fade_out_anim.setDuration(duration_ms)
        self._fade_out_anim.setStartValue(1.0)
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._fade_in_anim = QPropertyAnimation(
            self._overlay_opacity_effect, b"opacity"
        )
        self._fade_in_anim.setDuration(duration_ms)
        self._fade_in_anim.setStartValue(0.0)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def on_finished():
            # Commit the new image to the base label and clean up overlay
            self.image_label.setPixmap(next_composed)
            self._base_opacity_effect.setOpacity(1.0)
            self.overlay_label.hide()
            self._overlay_opacity_effect.setOpacity(0.0)

        # Connect only once: use the overlay anim as the finisher
        self._fade_in_anim.finished.connect(on_finished)

        # Start animations
        self._fade_out_anim.start()
        self._fade_in_anim.start()

    def _set_image_direct(self, pixmap: QPixmap) -> None:
        """Set image directly without transition."""
        # Save current pan position and zoom level before changing images
        self._save_pan_position()
        self._save_zoom_level()

        self.original_pixmap = pixmap

        # Auto-fit to window if enabled, but only if user hasn't manually
        # zoomed
        if self.fit_to_window and not self.user_has_zoomed:
            self._fit_to_window()
        else:
            # Restore zoom level if user has manually zoomed or fit_to_window
            # is disabled
            self._restore_zoom_level()

        # Restore saved pan position
        self._restore_pan_position()

        self._update_display()

    def _set_image_for_transition(self, pixmap: QPixmap) -> None:
        """Set image for transition without calling _update_display()."""
        # Save current pan position and zoom level before changing images
        self._save_pan_position()
        self._save_zoom_level()

        self.original_pixmap = pixmap

        # Auto-fit to window if enabled, but only if user hasn't manually
        # zoomed
        if self.fit_to_window and not self.user_has_zoomed:
            # Calculate zoom factor to fit the image
            label_size = self.image_label.size()
            image_size = self.original_pixmap.size()

            scale_x = label_size.width() / image_size.width()
            scale_y = label_size.height() / image_size.height()

            # Use the smaller scale to ensure the image fits
            # 90% to add some margin
            self.zoom_factor = min(scale_x, scale_y) * 0.9

            # Reset pan offset
            self.pan_offset_x = 0
            self.pan_offset_y = 0
        else:
            # Restore zoom level if user has manually zoomed or fit_to_window
            # is disabled
            self.zoom_factor = self.saved_zoom_factor

        # Restore saved pan position
        self.pan_offset_x = self.saved_pan_offset_x
        self.pan_offset_y = self.saved_pan_offset_y

        # Apply rotation prior to scaling
        rotated_source = self._apply_rotation(self.original_pixmap)

        # If fit-to-window used original size, recompute target dimensions
        # from the rotated source dimensions for better fidelity.
        scaled_width = int(rotated_source.width() * self.zoom_factor)
        scaled_height = int(rotated_source.height() * self.zoom_factor)

        # Create scaled pixmap
        self.display_pixmap = rotated_source.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Don't call _update_display() - let the transition animation handle it

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

        # Create scaled pixmap (with optional rotation)
        source_pixmap = self._apply_rotation(self.original_pixmap)

        self.display_pixmap = source_pixmap.scaled(
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Apply transition opacity
        painter.setOpacity(self._transition_opacity)

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

    def _create_buffer_with_opacity(self, pixmap: QPixmap,
                                    opacity: float) -> QPixmap:
        """Create a buffer with the specified opacity."""
        if pixmap is None:
            return None

        # Create a new pixmap with the same size as the label
        label_size = self.image_label.size()
        buffer_pixmap = QPixmap(label_size)
        buffer_pixmap.fill(Qt.GlobalColor.transparent)

        # Create painter
        painter = QPainter(buffer_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Apply opacity
        painter.setOpacity(opacity)

        # Calculate position to center the image
        x = (label_size.width() - pixmap.width()) // 2
        y = (label_size.height() - pixmap.height()) // 2

        # Apply pan offset
        x += self.pan_offset_x
        y += self.pan_offset_y

        # Convert to integers for drawPixmap
        x = int(x)
        y = int(y)

        # Draw the image
        painter.drawPixmap(x, y, pixmap)
        painter.end()

        return buffer_pixmap

    def _update_button_states(self) -> None:
        """Update the enabled state of control buttons."""
        if self.borderless:
            return  # No buttons in borderless mode

        has_image = self.original_pixmap is not None
        can_zoom_in = has_image and self.zoom_factor < 5.0
        can_zoom_out = has_image and self.zoom_factor > 0.1
        can_reset = (has_image and
                     (self.zoom_factor != 1.0 or
                      self.pan_offset_x != 0 or
                      self.pan_offset_y != 0))

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
        # Mark that user has manually zoomed
        self.user_has_zoomed = True
        self._update_display()
        self._save_pan_position()
        self._save_zoom_level()

    def _zoom_out(self) -> None:
        """Zoom out by 25%."""
        if self.original_pixmap is None:
            return

        self.zoom_factor /= 1.25
        self.zoom_factor = max(self.zoom_factor, 0.1)
        # Mark that user has manually zoomed
        self.user_has_zoomed = True
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
        # 90% to add some margin
        self.zoom_factor = min(scale_x, scale_y) * 0.9

        # Reset pan offset
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        self._update_display()

    def _reset_view(self) -> None:
        """Reset zoom and pan to default values."""
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        # Reset manual zoom flag
        self.user_has_zoomed = False
        self._update_display()
        self._save_zoom_level()

    def _mouse_press_event(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        if (event.button() == Qt.MouseButton.LeftButton and
                not self.crop_mode_active):
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
        if (event.button() == Qt.MouseButton.LeftButton and
                not self.crop_mode_active):
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
        # Keep overlay aligned with the base label
        if hasattr(self, 'overlay_label') and self.overlay_label is not None:
            self.overlay_label.setGeometry(self.image_label.geometry())

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

        return (x, y, self.display_pixmap.width(),
                self.display_pixmap.height())

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
        img_rect_width = max(0, min(
            img_rect_width, self.original_pixmap.width() - img_rect_x))
        img_rect_height = max(0, min(
            img_rect_height, self.original_pixmap.height() - img_rect_y))

        return QRect(img_rect_x, img_rect_y, img_rect_width, img_rect_height)

    def set_crop_mode(self, active: bool) -> None:
        """
        Set the crop mode state.

        Args:
            active: True to enable crop mode (disable panning), False to
                disable
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
        # Reset manual zoom flag
        self.user_has_zoomed = False

    def _start_smooth_transition(self, next_pixmap: QPixmap) -> None:
        """Start a smooth transition using double buffering."""
        # Stop any existing transition
        if self.transition_animation:
            self.transition_animation.stop()

        # Store the next image
        self.next_pixmap = next_pixmap

        # Prepare the new image in the hidden buffer
        self._prepare_hidden_buffer(next_pixmap)

        # Create fade-out animation for visible buffer
        self.transition_animation = QPropertyAnimation(
            self, b"transition_opacity")
        self.transition_animation.setDuration(800)  # 800ms transition
        self.transition_animation.setStartValue(1.0)
        self.transition_animation.setEndValue(0.0)
        self.transition_animation.setEasingCurve(
            QEasingCurve.Type.InOutCubic)
        self.transition_animation.finished.connect(
            self._on_fade_out_finished)
        self.transition_animation.start()

    def _prepare_hidden_buffer(self, next_pixmap: QPixmap) -> None:
        """Prepare the hidden buffer with the next image."""
        # Set up the new image
        self._set_image_for_transition(next_pixmap)

        # Create the hidden buffer with 0 opacity (invisible)
        self.hidden_buffer = self._create_buffer_with_opacity(
            self.display_pixmap, 0.0)
        self._buffer_opacity = 0.0

    def _on_fade_out_finished(self) -> None:
        """Handle fade-out completion - swap buffers and start fade-in."""
        # Swap buffers: hidden becomes visible, visible becomes hidden
        self.visible_buffer = self.hidden_buffer
        self.hidden_buffer = None

        # Update the display with the new visible buffer
        self.image_label.setPixmap(self.visible_buffer)

        # Clean up
        self.next_pixmap = None

        # Start fade-in animation for the new visible buffer
        self.transition_animation = QPropertyAnimation(
            self, b"buffer_opacity")
        self.transition_animation.setDuration(800)  # 800ms transition
        self.transition_animation.setStartValue(0.0)
        self.transition_animation.setEndValue(1.0)
        self.transition_animation.setEasingCurve(
            QEasingCurve.Type.InOutCubic)
        self.transition_animation.finished.connect(
            self._on_transition_complete)
        self.transition_animation.start()

    def _on_transition_complete(self) -> None:
        """Handle transition completion."""
        self.transition_animation = None
        # Reset to normal display mode
        self._transition_opacity = 1.0
        self._buffer_opacity = 1.0

    @pyqtProperty(float)
    def transition_opacity(self) -> float:
        """Get the current transition opacity (for animation)."""
        return self._transition_opacity

    @transition_opacity.setter
    def transition_opacity(self, opacity: float) -> None:
        """Set the transition opacity (for animation)."""
        self._transition_opacity = opacity
        self._update_display()

    @pyqtProperty(float)
    def buffer_opacity(self) -> float:
        """Get the current buffer opacity (for animation)."""
        return self._buffer_opacity

    @buffer_opacity.setter
    def buffer_opacity(self, opacity: float) -> None:
        """Set the buffer opacity (for animation)."""
        self._buffer_opacity = opacity
        if self.visible_buffer is not None:
            # Update the visible buffer with new opacity
            updated_buffer = self._create_buffer_with_opacity(
                self.display_pixmap, opacity)
            self.image_label.setPixmap(updated_buffer)
