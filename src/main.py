#!/usr/bin/env python3
"""
Main entry point for the Image Classifier application.

This module handles command line arguments and initializes the PyQt6
application.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.viewing_window import ViewingWindow
from core.image_manager import ImageManager
from core.rating_manager import RatingManager
from core.classifier import ImageClassifier
from utils.file_utils import get_app_data_dir
from utils.single_instance import check_single_instance


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image Classifier - View, rate, and classify images"
    )
    parser.add_argument(
        "image_path",
        help="Path to directory containing images to classify"
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Enable automatic sorting when accuracy threshold is met"
    )
    parser.add_argument(
        "--db",
        help="Custom database location for ratings"
    )
    parser.add_argument(
        "--model",
        help="Load pre-trained model from specified path"
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=0.9,
        help="Accuracy threshold for auto-sorting (default: 0.9)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output"
    )
    parser.add_argument(
        "--no-fit-window",
        action="store_true",
        help="Disable automatic fit-to-window for images"
    )
    parser.add_argument(
        "--viewing-mode",
        action="store_true",
        help=("Launch in minimal viewing-only mode (no training, just "
              "navigation, favorites, and delete)")
    )
    parser.add_argument(
        "--fullscreen-mode",
        action="store_true",
        help=("Launch in ultra-minimal fullscreen mode (no borders, menus, "
              "or UI elements - just the image)")
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of images to discover initially (for performance)"
    )

    return parser.parse_args()


def validate_image_path(path: str) -> Path:
    """Validate and return the image path."""
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    if not image_path.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {path}")

    return image_path


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    # Create logs directory
    log_dir = get_app_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path
    log_file = log_dir / "image_classifier.log"

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            (logging.StreamHandler(sys.stdout) if verbose
             else logging.NullHandler())
        ]
    )

    # Log the log file location
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")


def main() -> int:
    """Main application entry point."""
    try:
        # Check for single instance
        single_instance_manager = check_single_instance("ImageClassifier")

        args = parse_arguments()

        # Setup logging
        setup_logging(args.verbose)

        # Validate image path
        image_path = validate_image_path(args.image_path)

        # Get database path
        if args.db:
            db_path = Path(args.db)
        else:
            db_path = get_app_data_dir() / "ratings.db"

        # Create PyQt6 application
        app = QApplication(sys.argv)
        app.setApplicationName("Image Classifier")
        app.setApplicationVersion("0.1.0")
        app.setOrganizationName("ImageClassifier")

        # Suppress Qt CSS parsing warnings
        os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

        # PyQt6 handles high DPI scaling automatically

        # Initialize core components
        rating_manager = RatingManager(db_path)
        image_manager = ImageManager(image_path, max_files=args.max_files)
        classifier = ImageClassifier()

        # Print favorites storage location to console
        app_data_dir = get_app_data_dir()
        favorites_dir = app_data_dir / "favorites"
        print(f"Favorited files will be stored in: {favorites_dir}")
        print(f"Application data directory: {app_data_dir}")

        # Load pre-trained model if specified
        if args.model:
            model_path = Path(args.model)
            if model_path.exists():
                classifier.load_model(model_path)

        # Create and show main window
        if args.fullscreen_mode:
            main_window = ViewingWindow(
                image_manager=image_manager,
                fit_to_window=not args.no_fit_window,  # Default enabled
                fullscreen_mode=True
            )
        elif args.viewing_mode:
            main_window = ViewingWindow(
                image_manager=image_manager,
                fit_to_window=not args.no_fit_window  # Default enabled
            )
        else:
            main_window = MainWindow(
                image_manager=image_manager,
                rating_manager=rating_manager,
                classifier=classifier,
                auto_sort=args.sort,
                accuracy_threshold=args.accuracy,
                verbose=args.verbose,
                fit_to_window=not args.no_fit_window  # Default enabled
            )

        # Connect cleanup signal
        main_window.aboutToQuit.connect(
            lambda: single_instance_manager.cleanup()
            if single_instance_manager else None
        )

        main_window.show()

        # Start application event loop
        try:
            return app.exec()
        finally:
            # Clean up single instance manager
            if single_instance_manager:
                single_instance_manager.cleanup()

    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
