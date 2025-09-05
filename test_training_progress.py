#!/usr/bin/env python3
"""
Simple test script to verify training progress functionality.
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.classifier import ImageClassifier
from core.training_worker import TrainingWorker
from gui.training_progress_dialog import TrainingProgressDialog


def test_training_progress():
    """Test the training progress dialog."""
    app = QApplication(sys.argv)

    # Create classifier
    classifier = ImageClassifier()

    # Create progress dialog
    dialog = TrainingProgressDialog()

    # Show dialog
    dialog.show()

    # Simulate training with worker thread
    def simulate_training():
        # Create dummy training data
        dummy_paths = [Path(f"test_image_{i}.jpg") for i in range(25)]
        dummy_ratings = [i % 10 for i in range(25)]  # Ratings 0-9

        # Create training worker
        worker = TrainingWorker(classifier, dummy_paths, dummy_ratings)

        # Connect worker signals to dialog
        worker.training_started.connect(dialog.start_training)
        worker.training_progress.connect(dialog.update_phase)
        worker.training_completed.connect(dialog.complete_training)
        worker.error_occurred.connect(
            lambda msg: dialog.complete_training(False, None)
        )

        # Start the worker
        worker.start()

    # Start simulation after a short delay
    QTimer.singleShot(1000, simulate_training)

    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    test_training_progress()
