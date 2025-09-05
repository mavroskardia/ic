"""
Crop selection widget for the Image Classifier application.

This module provides a widget for selecting crop regions on images.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent, QKeyEvent, QRegion


class CropWidget(QWidget):
    """
    Widget for selecting crop regions on images.

    Features:
    - Visual rectangle selection with mouse
    - Real-time dimension display
    - Keyboard shortcuts for confirmation/cancellation
    - Semi-transparent overlay
    """

    # Signals
    crop_selected = pyqtSignal(QRect)  # Emitted when crop is confirmed
    crop_cancelled = pyqtSignal()      # Emitted when crop is cancelled
    crop_mode_changed = pyqtSignal(bool)  # Emitted when crop mode starts/stops

    def __init__(self, parent=None):
        """
        Initialize the crop widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Crop selection state
        self.is_selecting = False
        self.start_point = None
        self.end_point = None
        self.crop_rect = QRect()

        # Visual settings
        self.dim_opacity = 102  # 40% opacity for dimming (102/255 ≈ 0.4)
        self.border_width = 2  # White border width
        self.show_dimming = False  # Whether to show dimming overlay

        # Setup UI
        self._setup_ui()

        # Set focus policy for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Initially hidden
        self.hide()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Create a simple layout with controls positioned at the top
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Instructions
        self.instruction_label = QLabel("Draw a rectangle to select crop area")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.instruction_label)

        # Dimension display
        self.dimension_label = QLabel("Selection: 0 x 0 pixels")
        self.dimension_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dimension_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.dimension_label)

        # Control buttons
        button_layout = QHBoxLayout()

        self.confirm_button = QPushButton("Confirm Crop (Enter)")
        self.confirm_button.setShortcut("Return")
        self.confirm_button.clicked.connect(self._confirm_crop)
        self.confirm_button.setEnabled(False)
        button_layout.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel (Esc)")
        self.cancel_button.setShortcut("Escape")
        self.cancel_button.clicked.connect(self._cancel_crop)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Add stretch to push controls to top
        layout.addStretch()

        # Make the widget transparent to mouse events except for the controls
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Set transparent background for the drawing area
        self.setStyleSheet("""
            CropWidget {
                background-color: transparent;
            }
        """)

    def start_crop_selection(self) -> None:
        """Start the crop selection process."""
        self.is_selecting = True
        self.start_point = None
        self.end_point = None
        self.crop_rect = QRect()
        self.show_dimming = True  # Show dimming immediately
        self.confirm_button.setEnabled(False)
        self.dimension_label.setText("Selection: 0 x 0 pixels")

        # Position the widget to cover the entire parent (image viewer)
        if self.parent():
            self.setGeometry(self.parent().rect())

        self.show()
        self.setFocus()
        self.raise_()
        # Emit signal that crop mode has started
        self.crop_mode_changed.emit(True)

    def _confirm_crop(self) -> None:
        """Confirm the current crop selection."""
        if not self.crop_rect.isEmpty():
            self.crop_selected.emit(self.crop_rect)
            self._end_crop_selection()

    def _cancel_crop(self) -> None:
        """Cancel the crop selection."""
        self.crop_cancelled.emit()
        self._end_crop_selection()

    def _end_crop_selection(self) -> None:
        """End the crop selection process."""
        self.is_selecting = False
        self.start_point = None
        self.end_point = None
        self.crop_rect = QRect()
        self.show_dimming = False  # Hide dimming
        self.hide()
        # Emit signal that crop mode has ended
        self.crop_mode_changed.emit(False)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.crop_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        if self.is_selecting and self.start_point is not None:
            self.end_point = event.position().toPoint()
            self.crop_rect = QRect(self.start_point, self.end_point).normalized()
            self._update_dimension_display()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            if not self.crop_rect.isEmpty():
                self.confirm_button.setEnabled(True)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self._confirm_crop()
        elif event.key() == Qt.Key.Key_Escape:
            self._cancel_crop()
        else:
            super().keyPressEvent(event)

    def _update_dimension_display(self) -> None:
        """Update the dimension display label."""
        if not self.crop_rect.isEmpty():
            width = self.crop_rect.width()
            height = self.crop_rect.height()
            self.dimension_label.setText(f"Selection: {width} x {height} pixels")
        else:
            self.dimension_label.setText("Selection: 0 x 0 pixels")

    def paintEvent(self, event) -> None:
        """Handle paint events to draw the selection overlay."""
        if not self.is_selecting:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw dimming overlay if enabled
        if self.show_dimming:
            dim_color = QColor(0, 0, 0, self.dim_opacity)  # 40% opacity

            if self.crop_rect.isEmpty():
                # No selection yet, dim the entire area
                painter.fillRect(self.rect(), dim_color)
            else:
                # Dim the area outside the selection rectangle
                # Create a region that covers everything except the crop rectangle
                full_region = QRegion(self.rect())
                crop_region = QRegion(self.crop_rect)
                outside_region = full_region.subtracted(crop_region)

                # Fill the outside region with dimming
                painter.setClipRegion(outside_region)
                painter.fillRect(self.rect(), dim_color)
                painter.setClipping(False)

        # Draw white border around the selection if there is one
        if not self.crop_rect.isEmpty():
            pen = QPen(QColor(255, 255, 255, 255), self.border_width)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.crop_rect)

        painter.end()

    def get_crop_rect(self) -> QRect:
        """
        Get the current crop rectangle.

        Returns:
            The current crop rectangle
        """
        return self.crop_rect

    def set_crop_rect(self, rect: QRect) -> None:
        """
        Set the crop rectangle.

        Args:
            rect: The crop rectangle to set
        """
        self.crop_rect = rect
        self._update_dimension_display()
        self.update()

    def resizeEvent(self, event) -> None:
        """Handle resize events to keep the widget covering the parent."""
        super().resizeEvent(event)
        if self.parent() and self.isVisible():
            self.setGeometry(self.parent().rect())
