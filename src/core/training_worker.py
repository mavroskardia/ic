"""
Training Worker for the Image Classifier application.

This module provides a QThread-based worker for model training to prevent UI blocking.
"""

import logging
from pathlib import Path
from typing import List

from PyQt6.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    """
    Worker thread for model training.

    This class runs model training in a separate thread to prevent UI blocking.
    It emits signals to communicate progress and results back to the main thread.
    """

    # Signals
    training_started = pyqtSignal(int)  # total_phases
    training_progress = pyqtSignal(str, int)  # phase_name, progress
    training_completed = pyqtSignal(bool, float)  # success, accuracy
    error_occurred = pyqtSignal(str)  # error_message

    def __init__(self, classifier, image_paths: List[Path],
                 ratings: List[int], incremental: bool = False):
        """
        Initialize the training worker.

        Args:
            classifier: The ImageClassifier instance to train
            image_paths: List of image paths for training
            ratings: Corresponding ratings for training
            incremental: Whether to use incremental training
        """
        super().__init__()
        self.classifier = classifier
        self.image_paths = image_paths
        self.ratings = ratings
        self.incremental = incremental
        self.logger = logging.getLogger(__name__)

        # Connect classifier signals to our signals
        self.classifier.training_started.connect(
            self.training_started.emit
        )
        self.classifier.training_progress.connect(
            self.training_progress.emit
        )
        self.classifier.training_completed.connect(
            self.training_completed.emit
        )

    def run(self) -> None:
        """
        Run the training process in the worker thread.
        """
        try:
            self.logger.info(f"Starting training worker thread (incremental: {self.incremental})")

            # Perform the training
            if self.incremental:
                success = self.classifier.train_model_incremental(self.image_paths, self.ratings)
            else:
                success = self.classifier.train_model(self.image_paths, self.ratings)

            if not success:
                self.error_occurred.emit(
                    "Training failed - check logs for details"
                )

        except Exception as e:
            self.logger.error(f"Training worker error: {e}")
            self.error_occurred.emit(f"Training error: {str(e)}")

    def stop(self) -> None:
        """
        Stop the training worker.
        Note: This is a placeholder for future cancellation support.
        """
        self.logger.info("Training worker stop requested")
        # For now, we can't easily cancel sklearn training
        # This would require more complex implementation
        self.quit()
        self.wait()
