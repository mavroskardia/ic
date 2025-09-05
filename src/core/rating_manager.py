"""
Rating Manager for the Image Classifier application.

This module handles storing and retrieving image ratings in an SQLite database.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal


class RatingManager(QObject):
    """
    Manages image ratings in an SQLite database.

    This class handles:
    - Rating storage and retrieval
    - Rating history tracking
    - Database initialization and maintenance
    - Rating statistics
    """

    # Signals
    rating_updated = pyqtSignal(Path, int)  # image_path, rating
    rating_added = pyqtSignal(Path, int)    # image_path, rating

    def __init__(self, database_path: Path):
        """
        Initialize the Rating Manager.

        Args:
            database_path: Path to the SQLite database file
        """
        super().__init__()
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)

        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()

                # Create ratings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ratings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_path TEXT UNIQUE NOT NULL,
                        rating INTEGER NOT NULL
                            CHECK (rating >= 0 AND rating <= 9),
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        features TEXT,
                        confidence REAL DEFAULT 0.0,
                        used_for_training BOOLEAN DEFAULT FALSE,
                        training_timestamp DATETIME
                    )
                """)

                # Create rating history table for tracking changes
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rating_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_path TEXT NOT NULL,
                        old_rating INTEGER,
                        new_rating INTEGER NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        reason TEXT
                    )
                """)

                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_image_path
                    ON ratings(image_path)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rating
                    ON ratings(rating)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON ratings(timestamp)
                """)

                conn.commit()
                self.logger.info("Database initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def add_rating(self, image_path: Path, rating: int,
                   features: Optional[str] = None,
                   confidence: float = 0.0) -> bool:
        """
        Add or update a rating for an image.

        Args:
            image_path: Path to the image
            rating: Rating value (0-9)
            features: Optional feature string for ML training
            confidence: Confidence score for the rating

        Returns:
            True if successful, False otherwise
        """
        self.logger.debug(f"Adding rating {rating} for {image_path}")

        if not (0 <= rating <= 9):
            self.logger.error(f"Invalid rating: {rating}. Must be 0-9.")
            return False

        try:
            self.logger.debug("Connecting to database...")
            with sqlite3.connect(self.database_path, timeout=5.0) as conn:
                # Set busy timeout to prevent hanging
                conn.execute("PRAGMA busy_timeout = 5000")
                cursor = conn.cursor()

                # Check if rating already exists
                self.logger.debug("Checking if rating exists...")
                cursor.execute("""
                    SELECT rating FROM ratings WHERE image_path = ?
                """, (str(image_path),))

                result = cursor.fetchone()

                if result:
                    # Update existing rating
                    old_rating = result[0]
                    self.logger.debug(f"Updating existing rating from {old_rating} to {rating}")
                    cursor.execute("""
                        UPDATE ratings
                        SET rating = ?, timestamp = CURRENT_TIMESTAMP,
                            features = ?, confidence = ?
                        WHERE image_path = ?
                    """, (rating, features, confidence, str(image_path)))

                    # Record in history
                    cursor.execute("""
                        INSERT INTO rating_history
                        (image_path, old_rating, new_rating, reason)
                        VALUES (?, ?, ?, ?)
                    """, (str(image_path), old_rating, rating, "User update"))

                    self.rating_updated.emit(image_path, rating)
                    self.logger.info(f"Updated rating for {image_path}: {rating}")

                else:
                    # Add new rating
                    self.logger.debug("Adding new rating...")
                    cursor.execute("""
                        INSERT INTO ratings
                        (image_path, rating, features, confidence)
                        VALUES (?, ?, ?, ?)
                    """, (str(image_path), rating, features, confidence))

                    self.rating_added.emit(image_path, rating)
                    self.logger.info(f"Added rating for {image_path}: {rating}")

                self.logger.debug("Committing transaction...")
                conn.commit()
                self.logger.debug("Rating operation completed successfully")
                return True

        except Exception as e:
            self.logger.error(f"Error adding rating for {image_path}: {e}")
            return False

    def get_rating(self, image_path: Path) -> Optional[int]:
        """
        Get the rating for an image.

        Args:
            image_path: Path to the image

        Returns:
            Rating value (0-9) or None if not found
        """
        try:
            with sqlite3.connect(self.database_path, timeout=5.0) as conn:
                # Set busy timeout to prevent hanging
                conn.execute("PRAGMA busy_timeout = 5000")

                cursor = conn.cursor()
                cursor.execute("""
                    SELECT rating FROM ratings WHERE image_path = ?
                """, (str(image_path),))

                result = cursor.fetchone()
                return result[0] if result else None

        except Exception as e:
            self.logger.error(f"Error getting rating for {image_path}: {e}")
            return None

    def get_all_ratings(self) -> List[Tuple[Path, int, str, float]]:
        """
        Get all ratings with features and confidence.

        Returns:
            List of tuples: (image_path, rating, features, confidence)
        """
        try:
            with sqlite3.connect(self.database_path, timeout=5.0) as conn:
                # Set busy timeout to prevent hanging
                conn.execute("PRAGMA busy_timeout = 5000")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT image_path, rating, features, confidence
                    FROM ratings
                    ORDER BY timestamp DESC
                """)

                results = cursor.fetchall()
                return [(Path(path), rating, features or "", confidence)
                        for path, rating, features, confidence in results]

        except Exception as e:
            self.logger.error(f"Error getting all ratings: {e}")
            return []

    def get_new_ratings_for_training(self) -> List[Tuple[Path, int, str, float]]:
        """
        Get ratings that haven't been used for training yet.

        Returns:
            List of tuples: (image_path, rating, features, confidence)
        """
        try:
            with sqlite3.connect(self.database_path, timeout=5.0) as conn:
                # Set busy timeout to prevent hanging
                conn.execute("PRAGMA busy_timeout = 5000")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT image_path, rating, features, confidence
                    FROM ratings
                    WHERE used_for_training = FALSE
                    ORDER BY timestamp DESC
                """)

                results = cursor.fetchall()
                return [(Path(path), rating, features or "", confidence)
                        for path, rating, features, confidence in results]

        except Exception as e:
            self.logger.error(f"Error getting new ratings for training: {e}")
            return []

    def mark_ratings_as_trained(self, image_paths: List[Path]) -> bool:
        """
        Mark specific ratings as used for training.

        Args:
            image_paths: List of image paths to mark as trained

        Returns:
            True if successful, False otherwise
        """
        if not image_paths:
            return True

        try:
            with sqlite3.connect(self.database_path, timeout=5.0) as conn:
                # Set busy timeout to prevent hanging
                conn.execute("PRAGMA busy_timeout = 5000")
                cursor = conn.cursor()

                # Mark ratings as used for training
                placeholders = ','.join(['?' for _ in image_paths])
                cursor.execute(f"""
                    UPDATE ratings
                    SET used_for_training = TRUE, training_timestamp = CURRENT_TIMESTAMP
                    WHERE image_path IN ({placeholders})
                """, [str(path) for path in image_paths])

                conn.commit()
                self.logger.info(f"Marked {len(image_paths)} ratings as used for training")
                return True

        except Exception as e:
            self.logger.error(f"Error marking ratings as trained: {e}")
            return False

    def get_rating_statistics(self) -> Dict[str, int]:
        """
        Get statistics about ratings.

        Returns:
            Dictionary with rating counts and total
        """
        try:
            with sqlite3.connect(self.database_path, timeout=5.0) as conn:
                # Set busy timeout to prevent hanging
                conn.execute("PRAGMA busy_timeout = 5000")

                cursor = conn.cursor()

                # Get total count
                cursor.execute("SELECT COUNT(*) FROM ratings")
                total = cursor.fetchone()[0]

                # Get count by rating
                cursor.execute("""
                    SELECT rating, COUNT(*)
                    FROM ratings
                    GROUP BY rating
                    ORDER BY rating
                """)

                rating_counts = dict(cursor.fetchall())

                # Ensure all ratings 0-9 are represented
                for i in range(10):
                    if i not in rating_counts:
                        rating_counts[i] = 0

                stats = {
                    "total": total,
                    **rating_counts
                }

                return stats

        except Exception as e:
            self.logger.error(f"Error getting rating statistics: {e}")
            return {"total": 0}

    def get_rating_history(self, image_path: Path) -> List[Tuple[int, int, datetime, str]]:
        """
        Get rating history for an image.

        Args:
            image_path: Path to the image

        Returns:
            List of tuples: (old_rating, new_rating, timestamp, reason)
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT old_rating, new_rating, timestamp, reason
                    FROM rating_history
                    WHERE image_path = ?
                    ORDER BY timestamp DESC
                """, (str(image_path),))

                results = cursor.fetchall()
                return [(old_rating, new_rating,
                        datetime.fromisoformat(timestamp), reason)
                       for old_rating, new_rating, timestamp, reason in results]

        except Exception as e:
            self.logger.error(f"Error getting rating history: {image_path}: {e}")
            return []

    def delete_rating(self, image_path: Path) -> bool:
        """
        Delete a rating for an image.

        Args:
            image_path: Path to the image

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()

                # Get current rating before deletion
                cursor.execute("""
                    SELECT rating FROM ratings WHERE image_path = ?
                """, (str(image_path),))

                result = cursor.fetchone()
                if not result:
                    return False

                old_rating = result[0]

                # Delete from ratings table
                cursor.execute("""
                    DELETE FROM ratings WHERE image_path = ?
                """, (str(image_path),))

                # Record deletion in history
                cursor.execute("""
                    INSERT INTO rating_history
                    (image_path, old_rating, new_rating, reason)
                    VALUES (?, ?, ?, ?)
                """, (str(image_path), old_rating, None, "Deleted"))

                conn.commit()
                self.logger.info(f"Deleted rating for {image_path}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting rating for {image_path}: {e}")
            return False

    def export_ratings(self, export_path: Path) -> bool:
        """
        Export ratings to a CSV file.

        Args:
            export_path: Path to export file

        Returns:
            True if successful, False otherwise
        """
        try:
            import csv

            ratings = self.get_all_ratings()

            with open(export_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Image Path', 'Rating', 'Features', 'Confidence'])

                for image_path, rating, features, confidence in ratings:
                    writer.writerow([str(image_path), rating, features, confidence])

            self.logger.info(f"Exported {len(ratings)} ratings to {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting ratings: {e}")
            return False

    def get_database_info(self) -> Dict[str, any]:
        """
        Get database information.

        Returns:
            Dictionary with database statistics
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()

                # Get table sizes
                cursor.execute("SELECT COUNT(*) FROM ratings")
                ratings_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM rating_history")
                history_count = cursor.fetchone()[0]

                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                size_bytes = cursor.fetchone()[0]

                return {
                    "ratings_count": ratings_count,
                    "history_count": history_count,
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / (1024 * 1024),
                    "database_path": str(self.database_path)
                }

        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            return {}
