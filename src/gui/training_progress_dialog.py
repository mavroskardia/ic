"""
Training Progress Dialog for the Image Classifier application.

This module provides a dialog to show training progress with time estimation.
"""

import time
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QTextEdit, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class TrainingProgressDialog(QDialog):
    """
    Dialog to show training progress with time estimation.

    This dialog displays:
    - Current training phase
    - Progress bar with percentage
    - Estimated time remaining
    - Detailed status messages
    - Cancel button (if supported)
    """

    # Signals
    cancelled = pyqtSignal()

    def __init__(self, parent=None, cancellable: bool = True):
        """
        Initialize the training progress dialog.

        Args:
            parent: Parent widget
            cancellable: Whether to show cancel button
        """
        super().__init__(parent)
        self.cancellable = cancellable

        # Training timing
        self.start_time: Optional[float] = None
        self.phase_start_time: Optional[float] = None
        self.current_phase = ""
        self.total_phases = 0
        self.current_phase_index = 0

        # Setup UI
        self._setup_ui()
        self._setup_connections()

        # Set window properties
        self.setWindowTitle("Training Model")
        self.setModal(True)
        self.setFixedSize(500, 300)
        flags = self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        self.setWindowFlags(flags)

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("Training Machine Learning Model")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Phase label
        self.phase_label = QLabel("Initializing...")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        layout.addWidget(self.phase_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Time estimation
        time_frame = QFrame()
        time_layout = QHBoxLayout(time_frame)
        time_layout.setContentsMargins(0, 0, 0, 0)

        self.elapsed_label = QLabel("Elapsed: 0:00")
        self.remaining_label = QLabel("Remaining: --:--")

        time_layout.addWidget(self.elapsed_label)
        time_layout.addStretch()
        time_layout.addWidget(self.remaining_label)

        layout.addWidget(time_frame)

        # Status messages
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.status_text)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        if self.cancellable:
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._on_cancel_clicked)
            button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        # Timer for updating time display
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time_display)

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        pass

    def start_training(self, total_phases: int = 5) -> None:
        """
        Start the training process.

        Args:
            total_phases: Total number of training phases
        """
        self.start_time = time.time()
        self.total_phases = total_phases
        self.current_phase_index = 0

        # Start timer for time updates
        self.timer.start(1000)  # Update every second

        # Reset UI
        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.close_button.setEnabled(False)
        if self.cancellable:
            self.cancel_button.setEnabled(True)

        self._add_status_message("Training started...")

    def update_phase(self, phase_name: str, progress: int) -> None:
        """
        Update the current training phase.

        Args:
            phase_name: Name of the current phase
            progress: Progress percentage (0-100)
        """
        if phase_name != self.current_phase:
            self.current_phase = phase_name
            self.phase_start_time = time.time()
            self.current_phase_index += 1

            phase_text = (f"Phase {self.current_phase_index}/"
                         f"{self.total_phases}: {phase_name}")
            self.phase_label.setText(phase_text)
            self._add_status_message(f"Starting: {phase_name}")

        # Update progress bar
        self.progress_bar.setValue(progress)

        # Update status
        self._add_status_message(f"{phase_name}: {progress}% complete")

    def complete_training(self, success: bool,
                         accuracy: Optional[float] = None) -> None:
        """
        Complete the training process.

        Args:
            success: Whether training was successful
            accuracy: Final accuracy if successful
        """
        # Stop timer
        self.timer.stop()

        # Update UI
        if success:
            self.phase_label.setText("Training Completed Successfully")
            self.progress_bar.setValue(100)
            if accuracy is not None:
                self._add_status_message(f"Final accuracy: {accuracy:.1%}")
            self._add_status_message("Training completed successfully!")
        else:
            self.phase_label.setText("Training Failed")
            self._add_status_message(
                "Training failed. Check logs for details."
            )

        # Enable close button
        self.close_button.setEnabled(True)
        if self.cancellable:
            self.cancel_button.setEnabled(False)

    def _add_status_message(self, message: str) -> None:
        """Add a status message to the log."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.status_text.append(formatted_message)

        # Auto-scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_time_display(self) -> None:
        """Update the elapsed and remaining time display."""
        if self.start_time is None:
            return

        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        self.elapsed_label.setText(f"Elapsed: {elapsed_str}")

        # Calculate remaining time
        if self.progress_bar.value() > 0:
            # Estimate remaining time based on current progress
            progress_ratio = self.progress_bar.value() / 100.0
            if progress_ratio > 0:
                estimated_total = elapsed / progress_ratio
                remaining = estimated_total - elapsed
                remaining_str = self._format_time(max(0, remaining))
                self.remaining_label.setText(f"Remaining: {remaining_str}")
            else:
                self.remaining_label.setText("Remaining: --:--")
        else:
            self.remaining_label.setText("Remaining: --:--")

    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"

    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        self.cancelled.emit()
        self.accept()

    def closeEvent(self, event) -> None:
        """Handle dialog close event."""
        self.timer.stop()
        super().closeEvent(event)
