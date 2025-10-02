"""
Main Window for the Image Classifier application.

This module provides the primary user interface for viewing and rating images.
"""

import logging
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QSpinBox,
    QStatusBar, QFileDialog,
    QMessageBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal, QRect
from PyQt6.QtGui import QKeySequence, QFont, QAction

from .image_viewer import ImageViewer
from .rating_panel import RatingPanel
from .training_progress_dialog import TrainingProgressDialog
from .crop_widget import CropWidget
from core.image_manager import ImageManager
from core.rating_manager import RatingManager
from core.classifier import ImageClassifier
from core.training_worker import TrainingWorker


class MainWindow(QMainWindow):
    """
    Main application window for the Image Classifier.

    This window provides:
    - Image display and navigation
    - Rating interface
    - Progress tracking
    - Menu system
    - Status information
    """

    # Signals
    aboutToQuit = pyqtSignal()

    def __init__(self, image_manager: ImageManager,
                 rating_manager: RatingManager,
                 classifier: ImageClassifier,
                 auto_sort: bool = False,
                 accuracy_threshold: float = 0.9,
                 verbose: bool = False,
                 fit_to_window: bool = True):
        """
        Initialize the main window.

        Args:
            image_manager: Manager for image operations
            rating_manager: Manager for rating operations
            classifier: ML classifier for predictions
            auto_sort: Enable automatic sorting mode
            accuracy_threshold: Accuracy threshold for auto-sorting
            verbose: Enable verbose logging
            fit_to_window: Automatically fit images to window when loaded
        """
        super().__init__()

        # Store managers
        self.image_manager = image_manager
        self.rating_manager = rating_manager
        self.classifier = classifier

        # Configuration
        self.auto_sort = auto_sort
        self.accuracy_threshold = accuracy_threshold
        self.verbose = verbose
        self.fit_to_window = fit_to_window

        # Settings file path
        self.settings_file = self._get_settings_file_path()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        # Initialize UI
        self._setup_ui()
        self._setup_connections()
        self._setup_menu()
        self._setup_status_bar()

        # Load saved window geometry
        self._load_window_geometry()

        # Load saved image position
        self._load_image_position()

        # Load initial image
        self._load_current_image()

        # Auto-sort timer
        if auto_sort:
            self.auto_sort_timer = QTimer()
            self.auto_sort_timer.timeout.connect(self._check_auto_sort)
            self.auto_sort_timer.start(30000)  # Check every 30 seconds

        # Training worker and progress dialog
        self.training_worker = None
        self.training_dialog = None

        self.logger.info("Main window initialized")

    def _setup_ui(self) -> None:
        """Setup the user interface components."""
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Create splitter for resizable layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: Image viewer with crop overlay
        self.image_viewer = ImageViewer(fit_to_window=self.fit_to_window)
        splitter.addWidget(self.image_viewer)

        # Create crop widget as overlay on image viewer
        self.crop_widget = CropWidget(self.image_viewer)

        # Store original resize event method
        self._original_image_viewer_resize = self.image_viewer.resizeEvent
        # Connect image viewer resize events to update crop widget position
        self.image_viewer.resizeEvent = self._on_image_viewer_resize

        # Right side: Rating panel
        self.rating_panel = RatingPanel()
        splitter.addWidget(self.rating_panel)

        # Set splitter proportions (70% image, 30% rating)
        splitter.setSizes([700, 300])

        # Progress bar
        progress_frame = QFrame()
        progress_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        progress_layout = QHBoxLayout(progress_frame)

        # Progress label
        self.progress_label = QLabel("Image 0 of 0 (0%)")
        self.progress_label.setFont(QFont("Arial", 10))
        progress_layout.addWidget(self.progress_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addWidget(progress_frame)

        # Navigation buttons
        nav_frame = QFrame()
        nav_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        nav_layout = QHBoxLayout(nav_frame)

        # Previous button
        self.prev_button = QPushButton("← Previous")
        self.prev_button.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self.prev_button.clicked.connect(self._previous_image)
        nav_layout.addWidget(self.prev_button)

        # Next button
        self.next_button = QPushButton("Next →")
        self.next_button.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self.next_button.clicked.connect(self._next_image)
        nav_layout.addWidget(self.next_button)

        # Jump to image
        nav_layout.addWidget(QLabel("Go to:"))
        self.jump_spinbox = QSpinBox()
        self.jump_spinbox.setMinimum(1)
        self.jump_spinbox.setMaximum(1)
        self.jump_spinbox.valueChanged.connect(self._jump_to_image)
        nav_layout.addWidget(self.jump_spinbox)

        # Auto-sort indicator
        if self.auto_sort:
            self.auto_sort_label = QLabel("Auto-sort: Enabled")
            self.auto_sort_label.setStyleSheet("color: green; font-weight: bold;")
            nav_layout.addWidget(self.auto_sort_label)

        nav_layout.addStretch()

        main_layout.addWidget(nav_frame)

        # Window properties
        self.setWindowTitle("Image Classifier")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

    def _setup_connections(self) -> None:
        """Setup signal connections between components."""
        # Image manager signals
        self.image_manager.image_changed.connect(self._on_image_changed)
        self.image_manager.progress_updated.connect(self._on_progress_updated)

        # Rating panel signals
        self.rating_panel.rating_submitted.connect(self._on_rating_submitted)

        # Crop widget signals
        self.crop_widget.crop_selected.connect(self._on_crop_selected)
        self.crop_widget.crop_cancelled.connect(self._on_crop_cancelled)
        self.crop_widget.crop_mode_changed.connect(self._on_crop_mode_changed)

        # Classifier signals
        self.classifier.model_trained.connect(self._on_model_trained)
        self.classifier.prediction_made.connect(self._on_prediction_made)

    def _setup_menu(self) -> None:
        """Setup the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Open directory action
        open_action = QAction("&Open Directory...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_directory)
        file_menu.addAction(open_action)

        # Export ratings action
        export_action = QAction("&Export Ratings...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_ratings)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Fit to window toggle action
        self.fit_to_window_action = QAction("&Fit to Window", self)
        self.fit_to_window_action.setCheckable(True)
        self.fit_to_window_action.setChecked(self.fit_to_window)
        self.fit_to_window_action.setShortcut(QKeySequence("Ctrl+F"))
        self.fit_to_window_action.triggered.connect(self._toggle_fit_to_window)
        view_menu.addAction(self.fit_to_window_action)

        view_menu.addSeparator()

        # Delete current image action
        delete_action = QAction("&Delete Current Image", self)
        delete_action.setShortcut(QKeySequence("Delete"))
        delete_action.triggered.connect(self._delete_current_image)
        view_menu.addAction(delete_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Train model action
        train_action = QAction("&Train Model", self)
        train_action.setShortcut(QKeySequence("Ctrl+T"))
        train_action.triggered.connect(self._train_model)
        tools_menu.addAction(train_action)

        # Save model action
        save_model_action = QAction("&Save Model...", self)
        save_model_action.setShortcut(QKeySequence("Ctrl+S"))
        save_model_action.triggered.connect(self._save_model)
        tools_menu.addAction(save_model_action)

        # Load model action
        load_model_action = QAction("&Load Model...", self)
        load_model_action.setShortcut(QKeySequence("Ctrl+L"))
        load_model_action.triggered.connect(self._load_model)
        tools_menu.addAction(load_model_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status labels
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        self.status_bar.addPermanentWidget(QLabel("|"))

        # Model info
        self.model_label = QLabel("Model: Not trained")
        self.status_bar.addPermanentWidget(self.model_label)

        self.status_bar.addPermanentWidget(QLabel("|"))

        # Database info
        self.db_label = QLabel("DB: 0 ratings")
        self.status_bar.addPermanentWidget(self.db_label)

        # Update status
        self._update_status()

    def _load_current_image(self) -> None:
        """Load and display the current image."""
        current_image = self.image_manager.get_current_image()
        if current_image:
            self.image_viewer.set_image(current_image)
            self._update_rating_display()
        else:
            self.image_viewer.clear_image()
            self.rating_panel.clear_rating()

    def _update_rating_display(self) -> None:
        """Update the rating display for the current image."""
        current_path = self.image_manager.get_current_path()
        if current_path:
            rating = self.rating_manager.get_rating(current_path)
            self.rating_panel.set_current_rating(rating)

            # Show prediction if model is trained
            if self.classifier.is_trained:
                prediction, confidence = self.classifier.predict_rating(current_path)
                if prediction is not None:
                    self.rating_panel.show_prediction(prediction, confidence)

    def _update_status(self) -> None:
        """Update the status bar information."""
        # Model status
        model_info = self.classifier.get_model_info()
        if model_info['is_trained']:
            self.model_label.setText(f"Model: {model_info['accuracy']:.1%}")
        else:
            self.model_label.setText("Model: Not trained")

        # Database status
        stats = self.rating_manager.get_rating_statistics()
        self.db_label.setText(f"DB: {stats['total']} ratings")

    @pyqtSlot(Path, int, int)
    def _on_image_changed(self, image_path: Path, current: int, total: int) -> None:
        """Handle image change events."""
        self._load_current_image()
        self._update_status()

        # Update jump spinbox
        self.jump_spinbox.setMaximum(total)
        self.jump_spinbox.setValue(current)

        # Update window title
        self.setWindowTitle(f"Image Classifier - {image_path.name}")

        # Save current position to settings
        self._save_image_position()

    @pyqtSlot(int, int, float)
    def _on_progress_updated(self, current: int, total: int, percentage: float) -> None:
        """Handle progress update events."""
        self.progress_label.setText(f"Image {current} of {total} ({percentage:.1f}%)")
        self.progress_bar.setValue(int(percentage))

    @pyqtSlot(int)
    def _on_rating_submitted(self, rating: int) -> None:
        """Handle rating submission."""
        self.logger.debug(f"Rating submitted: {rating}")
        current_path = self.image_manager.get_current_path()
        if current_path:
            self.logger.debug(f"Adding rating to database for: {current_path}")
            # Add rating to database
            success = self.rating_manager.add_rating(current_path, rating)
            if success:
                self.logger.debug("Rating saved successfully, updating status")
                self.status_label.setText(f"Rating {rating} saved for {current_path.name}")

                # Defer status update to avoid blocking UI
                QTimer.singleShot(100, self._update_status)

                # Check if we should retrain the model
                if self._should_retrain_model():
                    self.logger.debug("Should retrain model, starting background training")
                    self._start_background_training()

                # Auto-advance to next image after successful rating
                self._auto_advance_after_rating()
            else:
                self.logger.error("Failed to save rating")
                self.status_label.setText("Failed to save rating")
        else:
            self.logger.warning("No current path available for rating")

    def _auto_advance_after_rating(self) -> None:
        """Automatically advance to the next image after a successful rating."""
        # Try to advance to the next image
        if self.image_manager.next_image():
            self.logger.debug("Auto-advanced to next image after rating")
            self._load_current_image()
            self._update_status()
        else:
            # If no next image, try to go to previous (in case we're at the end)
            if self.image_manager.previous_image():
                self.logger.debug("No next image, staying at current position")
                self._load_current_image()
                self._update_status()
            else:
                self.logger.debug("No images available for navigation")

    @pyqtSlot(float)
    def _on_model_trained(self, accuracy: float) -> None:
        """Handle model training completion."""
        self.status_label.setText(f"Model trained with {accuracy:.1%} accuracy")
        self._update_status()

        # Check auto-sort threshold
        if self.auto_sort and accuracy >= self.accuracy_threshold:
            self._perform_auto_sort()

    @pyqtSlot(Path, int, float)
    def _on_prediction_made(self, image_path: Path, rating: int, confidence: float) -> None:
        """Handle prediction events."""
        if self.verbose:
            self.logger.info(f"Prediction: {image_path.name} -> {rating} (confidence: {confidence:.3f})")

    def _on_training_started(self, total_phases: int) -> None:
        """Handle training started event."""
        if self.training_dialog is None:
            self.training_dialog = TrainingProgressDialog(self, cancellable=False)
            self.training_dialog.cancelled.connect(self._on_training_cancelled)

        self.training_dialog.start_training(total_phases)
        self.training_dialog.show()

    def _on_training_progress(self, phase_name: str, progress: int) -> None:
        """Handle training progress updates."""
        if self.training_dialog:
            self.training_dialog.update_phase(phase_name, progress)

    def _on_training_completed(self, success: bool, accuracy: float) -> None:
        """Handle training completion."""
        if self.training_dialog:
            self.training_dialog.complete_training(success, accuracy if success else None)
            # Don't close the dialog automatically - let user close it

        # Mark ratings as used for training if training was successful
        if success and self.training_worker:
            # Get the image paths that were used for training
            if hasattr(self.training_worker, 'image_paths'):
                self.rating_manager.mark_ratings_as_trained(self.training_worker.image_paths)

    def _on_training_error(self, error_message: str) -> None:
        """Handle training error."""
        if self.training_dialog:
            self.training_dialog.complete_training(False, None)

        self.status_label.setText(f"Training error: {error_message}")
        QMessageBox.critical(self, "Training Error", error_message)

    def _on_training_cancelled(self) -> None:
        """Handle training cancellation."""
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker = None

    def _previous_image(self) -> None:
        """Navigate to the previous image."""
        if self.image_manager.previous_image():
            self.status_label.setText("Moved to previous image")

    def _next_image(self) -> None:
        """Navigate to the next image."""
        if self.image_manager.next_image():
            self.status_label.setText("Moved to next image")

    def _jump_to_image(self, index: int) -> None:
        """Jump to a specific image by index."""
        if self.image_manager.go_to_image(index - 1):  # Convert to 0-based
            self.status_label.setText(f"Jumped to image {index}")

    def _should_retrain_model(self) -> bool:
        """Check if the model should be retrained."""
        self.logger.debug("Checking if model should be retrained...")
        try:
            # Add timeout protection for database operations
            stats = self.rating_manager.get_rating_statistics()
            self.logger.debug(f"Rating statistics: {stats}")

            # Check if we have enough total ratings
            if stats['total'] < 20:
                self.logger.debug("Insufficient total ratings for training")
                return False

            # Check if we have enough samples per class (minimum 3 per class)
            rating_counts = [stats.get(str(i), 0) for i in range(10)]
            min_samples_per_class = min(rating_counts)

            if min_samples_per_class < 3:
                self.logger.debug(f"Insufficient samples per class: {min_samples_per_class}")
                return False

            should_retrain = not self.classifier.is_trained
            self.logger.debug(f"Should retrain: {should_retrain}")
            return should_retrain
        except Exception as e:
            self.logger.error(f"Error checking if model should be retrained: {e}")
            return False

    def _start_background_training(self) -> None:
        """Start model training in a background thread."""
        try:
            # Check if we should use incremental training
            use_incremental = self.classifier.is_trained

            if use_incremental:
                # Get only new ratings for incremental training
                ratings_data = self.rating_manager.get_new_ratings_for_training()
                if len(ratings_data) == 0:
                    self.logger.debug("No new ratings for incremental training")
                    return

                self.status_label.setText("Starting incremental model training...")
            else:
                # Get all rated images for full training
                ratings_data = self.rating_manager.get_all_ratings()
                if len(ratings_data) < 10:  # Reduced minimum requirement
                    return

                self.status_label.setText("Starting model training...")

            # Extract image paths and ratings
            image_paths = [data[0] for data in ratings_data]
            ratings = [data[1] for data in ratings_data]

            # Check class distribution before training
            from collections import Counter
            rating_counts = Counter(ratings)
            min_samples = min(rating_counts.values())

            if min_samples < 2:  # Reduced minimum requirement
                return

            # Create and start training worker
            self.training_worker = TrainingWorker(
                self.classifier, image_paths, ratings, incremental=use_incremental
            )

            # Connect worker signals
            self.training_worker.training_started.connect(
                self._on_training_started
            )
            self.training_worker.training_progress.connect(
                self._on_training_progress
            )
            self.training_worker.training_completed.connect(
                self._on_training_completed
            )
            self.training_worker.error_occurred.connect(
                self._on_training_error
            )

            # Start the worker thread
            self.training_worker.start()

        except Exception as e:
            self.logger.error(f"Error starting background training: {e}")

    def _check_auto_sort(self) -> None:
        """Check if auto-sorting should be performed."""
        if not self.auto_sort:
            return

        if self.classifier.is_trained and self.classifier.accuracy >= self.accuracy_threshold:
            self._perform_auto_sort()

    def _perform_auto_sort(self) -> None:
        """Perform automatic sorting of images."""
        # This would implement the folder sorting logic
        # For now, just show a message
        QMessageBox.information(
            self, "Auto-Sort Ready",
            f"Model accuracy ({self.classifier.accuracy:.1%}) meets threshold "
            f"({self.accuracy_threshold:.1%}). Auto-sorting can be performed."
        )

    def _open_directory(self) -> None:
        """Open a new directory of images."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Image Directory"
        )
        if directory:
            # This would need to be implemented to change the image directory
            QMessageBox.information(
                self, "Not Implemented",
                "Changing directories is not yet implemented."
            )

    def _export_ratings(self) -> None:
        """Export ratings to a CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Ratings", "", "CSV Files (*.csv)"
        )
        if file_path:
            success = self.rating_manager.export_ratings(Path(file_path))
            if success:
                self.status_label.setText(f"Ratings exported to {file_path}")
            else:
                self.status_label.setText("Failed to export ratings")

    def _train_model(self) -> None:
        """Train the machine learning model."""
        # Get all rated images
        ratings_data = self.rating_manager.get_all_ratings()
        if len(ratings_data) < 5:  # Reduced minimum requirement
            QMessageBox.warning(
                self, "Insufficient Data",
                "Need at least 5 rated images to train the model."
            )
            return

        # Extract image paths and ratings
        image_paths = [data[0] for data in ratings_data]
        ratings = [data[1] for data in ratings_data]

        # Check class distribution before training
        from collections import Counter
        rating_counts = Counter(ratings)
        min_samples = min(rating_counts.values())

        if min_samples < 1:  # More lenient requirement
            # Show detailed information about the rating distribution
            rating_info = "\n".join([
                f"Rating {rating}: {count} images"
                for rating, count in sorted(rating_counts.items())
            ])

            QMessageBox.warning(
                self, "Insufficient Samples Per Class",
                f"Need at least 1 image per rating class to train the model.\n\n"
                f"Current distribution:\n{rating_info}\n\n"
                f"Please rate more images to ensure each rating has at least 1 sample."
            )
            return

        # Check if we should use incremental training
        use_incremental = self.classifier.is_trained

        if use_incremental:
            self.status_label.setText("Starting incremental model training...")
        else:
            self.status_label.setText("Starting model training...")

        # Create and start training worker
        self.training_worker = TrainingWorker(
            self.classifier, image_paths, ratings, incremental=use_incremental
        )

        # Connect worker signals
        self.training_worker.training_started.connect(
            self._on_training_started
        )
        self.training_worker.training_progress.connect(
            self._on_training_progress
        )
        self.training_worker.training_completed.connect(
            self._on_training_completed
        )
        self.training_worker.error_occurred.connect(
            self._on_training_error
        )

        # Start the worker thread
        self.training_worker.start()

    def _save_model(self) -> None:
        """Save the trained model."""
        if not self.classifier.is_trained:
            QMessageBox.warning(
                self, "No Model",
                "No trained model to save."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Model Files (*.pkl)"
        )
        if file_path:
            success = self.classifier.save_model(Path(file_path))
            if success:
                self.status_label.setText(f"Model saved to {file_path}")
            else:
                self.status_label.setText("Failed to save model")

    def _load_model(self) -> None:
        """Load a trained model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.pkl)"
        )
        if file_path:
            success = self.classifier.load_model(Path(file_path))
            if success:
                self.status_label.setText(f"Model loaded from {file_path}")
                self._update_status()
            else:
                self.status_label.setText("Failed to load model")

    def _delete_current_image(self) -> None:
        """Delete the current image by moving it to a deleted directory."""
        current_path = self.image_manager.get_current_path()
        if not current_path:
            self.status_label.setText("No image to delete")
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
                self.status_label.setText("Image moved to deleted directory")
                # Load next image if available
                if not self.image_manager.next_image():
                    # If no next image, try previous
                    self.image_manager.previous_image()
                self._load_current_image()
                self._update_status()
            else:
                self.status_label.setText("Failed to delete image")
                # Get log file location for user
                from utils.file_utils import get_app_data_dir
                log_file = get_app_data_dir() / "logs" / "image_classifier.log"
                QMessageBox.warning(
                    self, "Delete Failed",
                    f"Failed to delete the image. Check the log file for details:\n\n{log_file}"
                )

    def _add_to_favorites(self) -> None:
        """Add the current image to favorites with visual feedback."""
        current_path = self.image_manager.get_current_path()
        if not current_path:
            self.status_label.setText("No image to add to favorites")
            return

        # Add to favorites
        success = self.image_manager.add_to_favorites()
        if success:
            # Get the favorites directory path for the status message
            favorites_dir = self.image_manager.favorites_directory
            self.status_label.setText(
                f"Added '{current_path.name}' to favorites at {favorites_dir}"
            )

            # Provide visual feedback with a flash effect
            self._show_favorites_flash()
        else:
            self.status_label.setText("Failed to add image to favorites")
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

    def _start_crop_selection(self) -> None:
        """Start the crop selection process."""
        current_path = self.image_manager.get_current_path()
        if not current_path:
            self.status_label.setText("No image to crop")
            return

        # Start crop selection
        self.crop_widget.start_crop_selection()
        self.status_label.setText("Crop mode: Draw a rectangle to select crop area")

    def _on_crop_selected(self, crop_rect: QRect) -> None:
        """Handle crop selection confirmation."""
        # Convert crop rectangle from widget coordinates to image coordinates
        # This is a simplified conversion - in a real implementation, you'd need
        # to account for zoom, pan, and image scaling
        image_rect = self._convert_widget_to_image_coords(crop_rect)

        if image_rect.isEmpty():
            self.status_label.setText("Invalid crop selection")
            return

        # Perform the crop
        success = self.image_manager.crop_current_image(image_rect)
        if success:
            cropped_dir = self.image_manager.cropped_directory
            self.status_label.setText(
                f"Image cropped and saved to {cropped_dir}"
            )
        else:
            self.status_label.setText("Failed to crop image")
            # Get log file location for user
            from utils.file_utils import get_app_data_dir
            log_file = get_app_data_dir() / "logs" / "image_classifier.log"
            QMessageBox.warning(
                self, "Crop Failed",
                f"Failed to crop the image. "
                f"Check the log file for details:\n\n{log_file}"
            )

    def _on_crop_cancelled(self) -> None:
        """Handle crop selection cancellation."""
        self.status_label.setText("Crop cancelled")

    def _on_crop_mode_changed(self, active: bool) -> None:
        """Handle crop mode state changes."""
        self.image_viewer.set_crop_mode(active)
        if active:
            self.logger.debug("Crop mode activated - panning disabled")
        else:
            self.logger.debug("Crop mode deactivated - panning enabled")

    def _convert_widget_to_image_coords(self, widget_rect: QRect) -> QRect:
        """
        Convert crop rectangle from widget coordinates to image coordinates.

        Args:
            widget_rect: Rectangle in widget coordinates

        Returns:
            Rectangle in image coordinates
        """
        return self.image_viewer.convert_widget_to_image_coords(widget_rect)

    def _on_image_viewer_resize(self, event) -> None:
        """Handle image viewer resize events to update crop widget position."""
        # Call the original resize event method
        self._original_image_viewer_resize(event)

        # Update crop widget position if it's visible
        if self.crop_widget.isVisible():
            self.crop_widget.setGeometry(self.image_viewer.rect())

    def _toggle_fit_to_window(self) -> None:
        """Toggle the fit-to-window setting."""
        self.fit_to_window = self.fit_to_window_action.isChecked()
        self.image_viewer.set_fit_to_window(self.fit_to_window)

        # If enabled and we have an image, apply fit to window
        if self.fit_to_window and self.image_manager.get_current_image():
            self.image_viewer._fit_to_window()

        # Save settings immediately
        self._save_window_geometry()

        status = 'enabled' if self.fit_to_window else 'disabled'
        self.logger.info("Fit to window: %s", status)

    def _get_settings_file_path(self) -> Path:
        """Get the path to the settings file."""
        from utils.file_utils import get_app_data_dir
        return get_app_data_dir() / "window_settings.json"

    def _load_window_geometry(self) -> None:
        """Load saved window geometry from settings file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)

                # Restore window geometry
                if 'geometry' in settings:
                    geometry = settings['geometry']
                    self.setGeometry(
                        geometry.get('x', 100),
                        geometry.get('y', 100),
                        geometry.get('width', 1400),
                        geometry.get('height', 900)
                    )

                # Restore window state (maximized, etc.)
                if 'state' in settings:
                    state = settings['state']
                    if state.get('maximized', False):
                        self.showMaximized()
                    elif state.get('minimized', False):
                        self.showMinimized()

                # Restore fit-to-window setting
                if 'fit_to_window' in settings:
                    saved_fit = settings['fit_to_window']
                    if saved_fit != self.fit_to_window:
                        self.fit_to_window = saved_fit
                        self.fit_to_window_action.setChecked(saved_fit)
                        self.image_viewer.set_fit_to_window(saved_fit)

                self.logger.info("Window geometry restored from settings")
            else:
                # Use default geometry
                self.resize(1400, 900)
                self.move(100, 100)
                self.logger.info("Using default window geometry")

        except Exception as e:
            self.logger.warning(f"Failed to load window geometry: {e}")
            # Use default geometry on error
            self.resize(1400, 900)
            self.move(100, 100)

    def _save_window_geometry(self) -> None:
        """Save current window geometry to settings file."""
        try:
            # Create settings directory if it doesn't exist
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing settings or create new
            settings = {}
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)

            # Get current geometry
            geometry = self.geometry()
            state = {
                'maximized': self.isMaximized(),
                'minimized': self.isMinimized()
            }

            # Update geometry settings
            settings.update({
                'geometry': {
                    'x': geometry.x(),
                    'y': geometry.y(),
                    'width': geometry.width(),
                    'height': geometry.height()
                },
                'state': state,
                'fit_to_window': self.fit_to_window
            })

            # Save to file
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

            self.logger.info("Window geometry saved to settings")

        except Exception as e:
            self.logger.error(f"Failed to save window geometry: {e}")

    def _save_image_position(self) -> None:
        """Save current image position to settings file."""
        try:
            # Create settings directory if it doesn't exist
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing settings or create new
            settings = {}
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)

            # Get current image position info
            current_path = self.image_manager.get_current_path()
            current_index = self.image_manager.current_index
            total_count = self.image_manager.get_total_count()
            root_directory = str(self.image_manager.root_directory)

            # Update image position settings
            settings['image_position'] = {
                'root_directory': root_directory,
                'current_index': current_index,
                'total_count': total_count,
                'current_path': str(current_path) if current_path else None
            }

            # Save to file
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

            self.logger.debug(
                f"Image position saved: {current_index + 1}/{total_count} "
                f"in {root_directory}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save image position: {e}")

    def _load_image_position(self) -> bool:
        """Load saved image position from settings file.

        Returns:
            True if position was restored, False otherwise
        """
        try:
            if not self.settings_file.exists():
                self.logger.info(
                    "No settings file found, starting from first image"
                )
                return False

            with open(self.settings_file, 'r') as f:
                settings = json.load(f)

            if 'image_position' not in settings:
                self.logger.info(
                    "No image position found in settings, starting from first image"
                )
                return False

            image_pos = settings['image_position']
            saved_root = image_pos.get('root_directory')
            saved_index = image_pos.get('current_index', 0)
            saved_total = image_pos.get('total_count', 0)

            # Check if we're in the same directory
            current_root = str(self.image_manager.root_directory)
            if saved_root != current_root:
                self.logger.info(
                    f"Directory changed from {saved_root} to {current_root}, "
                    f"starting from first image"
                )
                return False

            # Check if the saved position is still valid
            current_total = self.image_manager.get_total_count()
            if saved_total != current_total:
                self.logger.info(
                    f"Image count changed from {saved_total} to {current_total}, "
                    f"adjusting position"
                )
                # Adjust index if it's beyond the new total
                if saved_index >= current_total:
                    saved_index = max(0, current_total - 1)

            # Restore position
            if 0 <= saved_index < current_total:
                self.image_manager.current_index = saved_index
                self.logger.info(
                    f"Restored image position: {saved_index + 1}/{current_total}"
                )
                return True
            else:
                self.logger.warning(
                    f"Invalid saved index {saved_index}, starting from first image"
                )
                return False

        except Exception as e:
            self.logger.error(f"Failed to load image position: {e}")
            return False

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Print current image index to console
        current_index = self.image_manager.current_index
        total_count = self.image_manager.get_total_count()
        if total_count > 0:
            print(f"Exiting at image {current_index + 1} of {total_count}")
        # Stop training worker if running
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait(3000)  # Wait up to 3 seconds

        # Save window geometry before closing
        self._save_window_geometry()

        # Emit signal for cleanup
        self.aboutToQuit.emit()

        super().closeEvent(event)

    def _show_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(
            self, "About Image Classifier",
            "Image Classifier v0.1.0\n\n"
            "A desktop application for viewing, rating, and "
            "classifying images using machine learning.\n\n"
            "Built with PyQt6 and scikit-learn."
        )

    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Left:
            self._previous_image()
        elif event.key() == Qt.Key.Key_Right:
            self._next_image()
        elif event.key() == Qt.Key.Key_Space:
            # Toggle between next/previous
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._previous_image()
            else:
                self._next_image()
        elif event.key() == Qt.Key.Key_Delete:
            self._delete_current_image()
        elif event.key() == Qt.Key.Key_F:
            # F key - add to favorites
            self._add_to_favorites()
        elif event.key() == Qt.Key.Key_C:
            # C key - start crop selection
            self._start_crop_selection()
        else:
            super().keyPressEvent(event)
