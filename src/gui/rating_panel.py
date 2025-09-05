"""
Rating Panel widget for the Image Classifier application.

This module provides a widget for rating images on a 0-9 scale.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class RatingPanel(QWidget):
    """
    Widget for rating images on a 0-9 scale.

    Features:
    - 0-9 rating buttons
    - Current rating display
    - Prediction display
    - Rating statistics
    """

    # Signals
    rating_submitted = pyqtSignal(int)  # rating value

    def __init__(self):
        """Initialize the rating panel."""
        super().__init__()

        # Current state
        self.current_rating = None
        self.predicted_rating = None
        self.prediction_confidence = 0.0

        # Setup UI
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("Rate This Image")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Current rating display
        self.current_rating_frame = QFrame()
        self.current_rating_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        current_layout = QVBoxLayout(self.current_rating_frame)

        current_label = QLabel("Current Rating:")
        current_label.setFont(QFont("Arial", 10))
        current_layout.addWidget(current_label)

        self.current_rating_label = QLabel("Not rated")
        self.current_rating_label.setFont(QFont("Arial", 12))
        self.current_rating_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_rating_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        current_layout.addWidget(self.current_rating_label)

        layout.addWidget(self.current_rating_frame)

        # Prediction display
        self.prediction_frame = QFrame()
        self.prediction_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        prediction_layout = QVBoxLayout(self.prediction_frame)

        prediction_label = QLabel("Predicted Rating:")
        prediction_label.setFont(QFont("Arial", 10))
        prediction_layout.addWidget(prediction_label)

        self.prediction_label = QLabel("No prediction")
        self.prediction_label.setFont(QFont("Arial", 12))
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_label.setStyleSheet("""
            QLabel {
                background-color: #e8f4fd;
                border: 1px solid #b3d9ff;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        prediction_layout.addWidget(self.prediction_label)

        # Confidence bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setVisible(False)
        prediction_layout.addWidget(self.confidence_bar)

        layout.addWidget(self.prediction_frame)

        # Rating buttons
        rating_label = QLabel("Select Rating:")
        rating_label.setFont(QFont("Arial", 10))
        layout.addWidget(rating_label)

        # Create rating buttons in a grid
        self.rating_buttons = []
        for row in range(2):
            button_row = QHBoxLayout()
            for col in range(5):
                rating = row * 5 + col
                button = self._create_rating_button(rating)
                self.rating_buttons.append(button)
                button_row.addWidget(button)
            layout.addLayout(button_row)

        # Quick rating shortcuts
        shortcuts_label = QLabel("Quick Rating:")
        shortcuts_label.setFont(QFont("Arial", 10))
        layout.addWidget(shortcuts_label)

        shortcuts_layout = QHBoxLayout()

        # Low rating
        low_button = QPushButton("Low (0-2)")
        low_button.setStyleSheet("""
            QPushButton {
                background-color: #ffcccc;
                border: 1px solid #ff9999;
                padding: 8px;
                border-radius: 4px;
                color: #800000;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffbbbb;
            }
        """)
        low_button.clicked.connect(lambda: self._quick_rate([0, 1, 2]))
        shortcuts_layout.addWidget(low_button)

        # Medium rating
        medium_button = QPushButton("Medium (3-6)")
        medium_button.setStyleSheet("""
            QPushButton {
                background-color: #ffffcc;
                border: 1px solid #ffff99;
                padding: 8px;
                border-radius: 4px;
                color: #806000;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffffbb;
            }
        """)
        medium_button.clicked.connect(lambda: self._quick_rate([3, 4, 5, 6]))
        shortcuts_layout.addWidget(medium_button)

        # High rating
        high_button = QPushButton("High (7-9)")
        high_button.setStyleSheet("""
            QPushButton {
                background-color: #ccffcc;
                border: 1px solid #99ff99;
                padding: 8px;
                border-radius: 4px;
                color: #006000;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #bbffbb;
            }
        """)
        high_button.clicked.connect(lambda: self._quick_rate([7, 8, 9]))
        shortcuts_layout.addWidget(high_button)

        layout.addLayout(shortcuts_layout)

        # Clear rating button
        clear_button = QPushButton("Clear Rating")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                padding: 8px;
                border-radius: 4px;
                color: #333333;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        clear_button.clicked.connect(self._clear_rating)
        layout.addWidget(clear_button)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Set panel properties
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)

    def _create_rating_button(self, rating: int) -> QPushButton:
        """Create a rating button for a specific rating value."""
        button = QPushButton(str(rating))
        button.setFixedSize(50, 50)
        button.setFont(QFont("Arial", 14, QFont.Weight.Bold))

        # Set button style based on rating
        if rating <= 2:
            # Low ratings - red tones
            red_value = max(0, min(255, 255 - rating * 30))
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #ff{red_value:02x}cc;
                    border: 2px solid #ff{red_value:02x}99;
                    border-radius: 25px;
                    color: #800000;
                }}
                QPushButton:hover {{
                    background-color: #ff{red_value:02x}bb;
                    border-color: #ff{red_value:02x}77;
                }}
                QPushButton:pressed {{
                    background-color: #ff{red_value:02x}99;
                }}
            """)
        elif rating <= 6:
            # Medium ratings - yellow tones
            yellow_value = max(0, min(255, 255 - rating * 30))
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #ffff{yellow_value:02x};
                    border: 2px solid #ffff{yellow_value:02x};
                    border-radius: 25px;
                    color: #806000;
                }}
                QPushButton:hover {{
                    background-color: #ffff{yellow_value:02x};
                    border-color: #ffff{yellow_value:02x};
                }}
                QPushButton:pressed {{
                    background-color: #ffff{yellow_value:02x};
                }}
            """)
        else:
            # High ratings - green tones
            green_value = max(0, min(255, 255 - rating * 30))
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #ccff{green_value:02x};
                    border: 2px solid #99ff{green_value:02x};
                    border-radius: 25px;
                    color: #006000;
                }}
                QPushButton:hover {{
                    background-color: #bbff{green_value:02x};
                    border-color: #77ff{green_value:02x};
                }}
                QPushButton:pressed {{
                    background-color: #99ff{green_value:02x};
                }}
            """)

        button.clicked.connect(lambda: self._submit_rating(rating))
        return button

    def _submit_rating(self, rating: int) -> None:
        """Submit a rating."""
        self.current_rating = rating
        self._update_display()
        self.rating_submitted.emit(rating)

    def _quick_rate(self, ratings: list) -> None:
        """Show a quick rating dialog for a range of ratings."""
        # For now, just use the middle rating
        # This could be enhanced with a proper dialog
        middle_rating = ratings[len(ratings) // 2]
        self._submit_rating(middle_rating)

    def _clear_rating(self) -> None:
        """Clear the current rating."""
        self.current_rating = None
        self._update_display()
        self.rating_submitted.emit(-1)  # -1 indicates no rating

    def _update_display(self) -> None:
        """Update the display to reflect current state."""
        # Update current rating display
        if self.current_rating is not None:
            self.current_rating_label.setText(f"{self.current_rating}")
            self.current_rating_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    padding: 5px;
                    border-radius: 3px;
                    color: #155724;
                }
            """)
        else:
            self.current_rating_label.setText("Not rated")
            self.current_rating_label.setStyleSheet("""
                QLabel {
                    background-color: #f0f0f0;
                    border: 1px solid #cccccc;
                    padding: 5px;
                    border-radius: 3px;
                }
            """)

        # Update prediction display
        if self.predicted_rating is not None:
            self.prediction_label.setText(f"{self.predicted_rating}")
            self.prediction_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f4fd;
                    border: 1px solid #b3d9ff;
                    padding: 5px;
                    border-radius: 3px;
                    color: #0c5460;
                }
            """)

            # Show confidence bar
            # Handle infinite or NaN confidence values
            is_infinite = (self.prediction_confidence == float('inf') or
                          self.prediction_confidence == float('-inf'))
            is_nan = self.prediction_confidence != self.prediction_confidence
            if not (is_infinite or is_nan):
                confidence_percent = min(100, max(0,
                                    int(self.prediction_confidence * 100)))
                self.confidence_bar.setValue(confidence_percent)
            else:
                # Set to 0 for invalid confidence values
                self.confidence_bar.setValue(0)
            self.confidence_bar.setVisible(True)
        else:
            self.prediction_label.setText("No prediction")
            self.prediction_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f4fd;
                    border: 1px solid #b3d9ff;
                    padding: 5px;
                    border-radius: 3px;
                }
            """)
            self.confidence_bar.setVisible(False)

        # Update button states
        self._update_button_states()

    def _update_button_states(self) -> None:
        """Update the enabled state of rating buttons."""
        for i, button in enumerate(self.rating_buttons):
            if self.current_rating == i:
                # Highlight current rating
                button.setStyleSheet(button.styleSheet() + """
                    QPushButton {
                        border-width: 4px;
                        font-weight: bold;
                    }
                """)
            else:
                # Reset to normal style
                self._create_rating_button(i)

    def set_current_rating(self, rating: int) -> None:
        """
        Set the current rating display.

        Args:
            rating: The rating value to display
        """
        self.current_rating = rating
        self._update_display()

    def show_prediction(self, rating: int, confidence: float) -> None:
        """
        Show a prediction for the current image.

        Args:
            rating: The predicted rating
            confidence: The confidence score (0.0 to 1.0)
        """
        self.predicted_rating = rating
        self.prediction_confidence = confidence
        self._update_display()

    def clear_rating(self) -> None:
        """Clear all rating information."""
        self.current_rating = None
        self.predicted_rating = None
        self.prediction_confidence = 0.0
        self._update_display()

    def get_current_rating(self) -> int:
        """Get the current rating value."""
        return self.current_rating if self.current_rating is not None else -1

    def get_prediction(self) -> tuple:
        """Get the current prediction information."""
        if self.predicted_rating is not None:
            return (self.predicted_rating, self.prediction_confidence)
        return (None, 0.0)
