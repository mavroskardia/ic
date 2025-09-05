"""
Image Classifier for the Image Classifier application.

This module handles machine learning classification of images based on extracted features.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import cv2

from PyQt6.QtCore import QObject, pyqtSignal


class ImageClassifier(QObject):
    """
    Machine learning classifier for predicting image ratings.

    This class handles:
    - Feature extraction from images
    - Model training and prediction
    - Model persistence and loading
    - Performance evaluation
    """

    # Signals
    model_trained = pyqtSignal(float)  # accuracy score
    prediction_made = pyqtSignal(Path, int, float)  # image_path, rating, confidence
    training_progress = pyqtSignal(str, int)  # phase_name, progress
    training_started = pyqtSignal(int)  # total_phases
    training_completed = pyqtSignal(bool, float)  # success, accuracy

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the Image Classifier.

        Args:
            model_path: Optional path to pre-trained model
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # ML components
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

        # Feature extraction cache with size limit
        self.feature_cache: Dict[Path, np.ndarray] = {}
        self.max_cache_size = 1000  # Limit cache size to prevent memory issues

        # Model state
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_names: List[str] = []
        self.training_data_count = 0  # Track how many samples have been used for training

        # Load pre-trained model if provided
        if model_path and model_path.exists():
            self.load_model(model_path)

    def extract_features(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extract features from an image.

        Args:
            image_path: Path to the image

        Returns:
            Feature vector as numpy array or None if failed
        """
        # Check cache first
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]

        try:
            # Load image with OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract features
            features = self._extract_all_features(image_rgb)

            # Cache features with size management
            self._cache_features(image_path, features)

            return features

        except Exception as e:
            self.logger.error(f"Error extracting features from {image_path}: {e}")
            return None

    def _extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all features from an image array with optimizations.

        Args:
            image: RGB image as numpy array

        Returns:
            Concatenated feature vector
        """
        features = []

        # Resize image for faster processing if too large
        if image.shape[0] > 512 or image.shape[1] > 512:
            scale = min(512 / image.shape[0], 512 / image.shape[1])
            new_height = int(image.shape[0] * scale)
            new_width = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Color histogram features (RGB and HSV) - most important
        features.extend(self._extract_color_features(image))

        # Texture features - simplified for speed
        features.extend(self._extract_texture_features_fast(image))

        # Edge features - simplified
        features.extend(self._extract_edge_features_fast(image))

        # Skip expensive features for speed
        # Shape features
        # features.extend(self._extract_shape_features(image))

        # Face detection features (skip for speed)
        # features.extend(self._extract_face_features(image))

        return np.array(features, dtype=np.float32)

    def _extract_color_features(self, image: np.ndarray) -> List[float]:
        """Extract color histogram features."""
        features = []

        # RGB histograms
        for i in range(3):  # R, G, B channels
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        # HSV histograms
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i in range(3):  # H, S, V channels
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        return features

    def _extract_texture_features_fast(self, image: np.ndarray) -> List[float]:
        """Extract simplified texture features for speed."""
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple texture features (variance, mean)
        features.append(np.var(gray))
        features.append(np.mean(gray))
        features.append(np.std(gray))

        return features

    def _extract_edge_features_fast(self, image: np.ndarray) -> List[float]:
        """Extract simplified edge features for speed."""
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_ratio)

        return features

    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features using GLCM."""
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple texture features (variance, entropy)
        features.append(np.var(gray))
        features.append(self._calculate_entropy(gray))

        # Local Binary Pattern (simplified)
        lbp = self._calculate_lbp(gray)
        features.append(np.mean(lbp))
        features.append(np.std(lbp))

        return features

    def _extract_edge_features(self, image: np.ndarray) -> List[float]:
        """Extract edge detection features."""
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))

        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.append(np.mean(gradient_magnitude))

        return features

    def _extract_shape_features(self, image: np.ndarray) -> List[float]:
        """Extract basic shape features."""
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find contours
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Largest contour area
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            features.append(area / (image.shape[0] * image.shape[1]))

            # Perimeter
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(perimeter / (image.shape[0] + image.shape[1]))
        else:
            features.extend([0.0, 0.0])

        return features

    def _extract_face_features(self, image: np.ndarray) -> List[float]:
        """Extract face detection features."""
        features = []

        try:
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Number of faces
                features.append(len(faces))

                # Largest face area
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                face_area = largest_face[2] * largest_face[3]
                total_area = image.shape[0] * image.shape[1]
                features.append(face_area / total_area)
            else:
                features.extend([0.0, 0.0])

        except Exception as e:
            self.logger.warning(f"Face detection failed: {e}")
            features.extend([0.0, 0.0])

        return features

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern (simplified)."""
        # Simple 3x3 LBP
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                code = 0
                for k, (di, dj) in enumerate([
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1), (1, -1),
                    (1, 0), (1, 1)
                ]):
                    if image[i + di, j + dj] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        return lbp

    def _cache_features(self, image_path: Path, features: np.ndarray) -> None:
        """Cache features with size management."""
        # Remove oldest entries if cache is too large
        if len(self.feature_cache) >= self.max_cache_size:
            # Remove 20% of oldest entries
            items_to_remove = len(self.feature_cache) // 5
            keys_to_remove = list(self.feature_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.feature_cache[key]

        self.feature_cache[image_path] = features

    def train_model_incremental(self, image_paths: List[Path], ratings: List[int]) -> bool:
        """
        Train the model incrementally using only new data.

        Args:
            image_paths: List of new image paths
            ratings: Corresponding ratings

        Returns:
            True if training successful, False otherwise
        """
        if not image_paths or len(image_paths) == 0:
            self.logger.warning("No new data for incremental training")
            return True  # Not an error, just nothing to do

        if not self.is_trained:
            self.logger.info("No existing model found, performing full training")
            return self.train_model(image_paths, ratings)

        try:
            self.logger.info(f"Starting incremental training with {len(image_paths)} new samples")

            # Emit training started signal
            self.training_started.emit(4)  # 4 phases for incremental training

            # Phase 1: Extract features from new data
            self.training_progress.emit("Extracting features", 0)
            features_list = []
            valid_ratings = []
            valid_paths = []

            for i, (image_path, rating) in enumerate(zip(image_paths, ratings)):
                try:
                    features = self.extract_features(image_path)
                    if features is not None and not np.isnan(features).any():
                        features_list.append(features)
                        valid_ratings.append(rating)
                        valid_paths.append(image_path)
                    else:
                        self.logger.warning(f"Failed to extract features from {image_path}")
                except Exception as e:
                    self.logger.warning(f"Error extracting features from {image_path}: {e}")

                # Update progress
                progress = int((i + 1) / len(image_paths) * 25)
                self.training_progress.emit("Extracting features", progress)

            if len(features_list) == 0:
                self.logger.warning("No valid features extracted for incremental training")
                return False

            # Phase 2: Prepare new data
            self.training_progress.emit("Preparing data", 25)
            X_new = np.array(features_list, dtype=np.float32)
            y_new = np.array(valid_ratings, dtype=np.int32)

            # Check for NaN or infinite values
            if np.isnan(X_new).any() or np.isinf(X_new).any():
                self.logger.error("New features contain NaN or infinite values")
                return False

            # Phase 3: Scale new features using existing scaler
            self.training_progress.emit("Scaling features", 50)
            try:
                X_new_scaled = self.scaler.transform(X_new)
            except Exception as e:
                self.logger.warning(f"Feature scaling failed: {e}, using original features")
                X_new_scaled = X_new

            # Phase 4: Update model using partial_fit (if available) or retrain
            self.training_progress.emit("Updating model", 75)

            # For RandomForest, we need to retrain with combined data
            # In a real implementation, you might want to use an online learning algorithm
            # like SGDClassifier or PassiveAggressiveClassifier for true incremental learning

            # For now, we'll use a hybrid approach: retrain with recent data
            # This is still more efficient than retraining with all data
            self.logger.info("Updating RandomForest model with new data")

            # Use a smaller, faster model for incremental updates
            n_estimators = min(50, max(10, len(X_new) * 2))
            self.classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
                max_depth=min(8, max(3, int(np.log2(len(X_new))))),
                min_samples_split=max(2, len(X_new) // 10)
            )

            # Train on new data only (this is a simplified approach)
            # In practice, you might want to keep some old data for stability
            self.classifier.fit(X_new_scaled, y_new)

            # Update training data count
            self.training_data_count += len(valid_paths)

            # Simple accuracy calculation on new data
            y_pred = self.classifier.predict(X_new_scaled)
            new_accuracy = accuracy_score(y_new, y_pred)

            # Update overall accuracy (weighted average)
            if self.training_data_count > 0:
                old_weight = (self.training_data_count - len(valid_paths)) / self.training_data_count
                new_weight = len(valid_paths) / self.training_data_count
                self.accuracy = (self.accuracy * old_weight) + (new_accuracy * new_weight)

            self.logger.info(f"Incremental training completed. New accuracy: {new_accuracy:.3f}, Overall: {self.accuracy:.3f}")

            # Emit completion signal
            self.training_progress.emit("Training completed", 100)
            self.training_completed.emit(True, self.accuracy)
            self.model_trained.emit(self.accuracy)

            return True

        except Exception as e:
            self.logger.error(f"Error in incremental training: {e}")
            self.training_completed.emit(False, 0.0)
            return False

    def train_model(self, image_paths: List[Path], ratings: List[int]) -> bool:
        """
        Train the classification model with improved robustness and speed.

        Args:
            image_paths: List of image paths
            ratings: Corresponding ratings

        Returns:
            True if training successful, False otherwise
        """
        # Enhanced validation
        if len(image_paths) < 10:  # Reduced minimum requirement
            self.logger.warning("Need at least 10 samples for training")
            self.training_completed.emit(False, 0.0)
            return False

        if len(image_paths) != len(ratings):
            self.logger.error("Mismatch between image paths and ratings count")
            self.training_completed.emit(False, 0.0)
            return False

        try:
            # Emit training started signal
            self.training_started.emit(6)  # 6 phases total

            # Phase 1: Extract features with better error handling
            self.training_progress.emit("Extracting features", 0)
            self.logger.info("Extracting features for training...")
            features_list = []
            valid_ratings = []
            failed_extractions = 0

            total_images = len(image_paths)
            for i, (image_path, rating) in enumerate(zip(image_paths, ratings)):
                try:
                    features = self.extract_features(image_path)
                    if features is not None and not np.isnan(features).any():
                        features_list.append(features)
                        valid_ratings.append(rating)
                    else:
                        failed_extractions += 1
                        self.logger.warning(f"Failed to extract features from {image_path}")
                except Exception as e:
                    failed_extractions += 1
                    self.logger.warning(f"Error extracting features from {image_path}: {e}")

                # Update progress every 5 images for better responsiveness
                if (i + 1) % 5 == 0 or i == total_images - 1:
                    # 25% of total progress
                    progress = int((i + 1) / total_images * 25)
                    self.training_progress.emit("Extracting features", progress)

            self.logger.info(f"Feature extraction complete: {len(features_list)} successful, {failed_extractions} failed")

            # More lenient validation
            if len(features_list) < 5:
                self.logger.error(f"Not enough valid features for training: {len(features_list)}")
                self.training_completed.emit(False, 0.0)
                return False

            # Phase 2: Prepare data with better error handling
            self.training_progress.emit("Preparing data", 25)

            try:
                # Convert to numpy arrays with error handling
                X = np.array(features_list, dtype=np.float32)
                y = np.array(valid_ratings, dtype=np.int32)

                # Check for NaN or infinite values
                if np.isnan(X).any() or np.isinf(X).any():
                    self.logger.error("Features contain NaN or infinite values")
                    self.training_completed.emit(False, 0.0)
                    return False

                # Check class distribution with more lenient requirements
                unique_ratings, counts = np.unique(y, return_counts=True)
                min_samples_per_class = counts.min()
                total_classes = len(unique_ratings)

                self.logger.info(f"Class distribution: {dict(zip(unique_ratings, counts))}")

                # More flexible class requirements
                if min_samples_per_class < 2:
                    self.logger.error(
                        f"Insufficient samples per class. Minimum required: 2, "
                        f"found: {min_samples_per_class}. "
                        f"Class distribution: {dict(zip(unique_ratings, counts))}"
                    )
                    self.training_completed.emit(False, 0.0)
                    return False

                # If we have very few classes, use a simpler approach
                if total_classes < 3:
                    self.logger.warning(f"Only {total_classes} classes found, using simplified training")
                    return self._train_simple_model(X, y)

            except Exception as e:
                self.logger.error(f"Error preparing data: {e}")
                self.training_completed.emit(False, 0.0)
                return False

            # Phase 3: Split data with better error handling
            self.training_progress.emit("Splitting data", 40)

            try:
                # Use smaller test size for small datasets
                test_size = min(0.2, max(0.1, 5 / len(X)))

                # Split data (only use stratify if we have enough samples per class)
                if min_samples_per_class >= 2 and total_classes >= 2:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                    except ValueError as e:
                        self.logger.warning(f"Stratified split failed: {e}, using random split")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                # Phase 4: Scale features with error handling
                self.training_progress.emit("Scaling features", 50)
                try:
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                except Exception as e:
                    self.logger.warning(f"Feature scaling failed: {e}, using original features")
                    X_train_scaled = X_train
                    X_test_scaled = X_test

                # Phase 5: Train model with optimized parameters
                self.training_progress.emit("Training classifier", 60)
                self.logger.info("Training Random Forest classifier...")

                # Use smaller, faster model for small datasets
                n_estimators = min(100, max(10, len(X_train) // 2))
                self.classifier = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=min(10, max(3, int(np.log2(len(X_train))))),
                    min_samples_split=max(2, len(X_train) // 20)
                )

                self.classifier.fit(X_train_scaled, y_train)

                # Phase 6: Evaluate model
                self.training_progress.emit("Evaluating model", 80)

                # Evaluate model
                y_pred = self.classifier.predict(X_test_scaled)
                self.accuracy = accuracy_score(y_test, y_pred)

                # Simplified cross-validation for small datasets
                if len(X_train) >= 20:
                    try:
                        cv_folds = min(5, len(X_train) // 4)
                        cv_scores = cross_val_score(
                            self.classifier, X_train_scaled, y_train, cv=cv_folds
                        )
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std() * 2
                        self.logger.info(f"Cross-validation scores: {cv_mean:.3f} (+/- {cv_std:.3f})")
                    except Exception as e:
                        self.logger.warning(f"Cross-validation failed: {e}")

                self.logger.info(f"Training completed. Accuracy: {self.accuracy:.3f}")

                # Update state
                self.is_trained = True
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

                # Emit completion signal
                self.training_progress.emit("Training completed", 100)
                self.training_completed.emit(True, self.accuracy)
                self.model_trained.emit(self.accuracy)

                return True

            except Exception as e:
                self.logger.error(f"Error in training phases: {e}")
                self.training_completed.emit(False, 0.0)
                return False

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            self.training_completed.emit(False, 0.0)
            return False

    def _train_simple_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Train a simplified model for cases with very few classes or samples.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            True if training successful, False otherwise
        """
        try:
            self.logger.info("Training simplified model...")

            # Use a very simple classifier
            from sklearn.ensemble import RandomForestClassifier

            # Minimal parameters for speed
            self.classifier = RandomForestClassifier(
                n_estimators=10,
                max_depth=3,
                random_state=42,
                n_jobs=-1
            )

            # Train on all data (no split for very small datasets)
            self.classifier.fit(X, y)

            # Simple accuracy calculation
            y_pred = self.classifier.predict(X)
            self.accuracy = accuracy_score(y, y_pred)

            # Update state
            self.is_trained = True
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            self.logger.info(f"Simplified training completed. Accuracy: {self.accuracy:.3f}")

            # Emit completion signal
            self.training_progress.emit("Training completed", 100)
            self.training_completed.emit(True, self.accuracy)
            self.model_trained.emit(self.accuracy)

            return True

        except Exception as e:
            self.logger.error(f"Error in simplified training: {e}")
            self.training_completed.emit(False, 0.0)
            return False

    def predict_rating(self, image_path: Path) -> Tuple[Optional[int], float]:
        """
        Predict rating for an image.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (predicted_rating, confidence)
        """
        if not self.is_trained:
            self.logger.warning("Model not trained yet")
            return None, 0.0

        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return None, 0.0

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Make prediction
            prediction = self.classifier.predict(features_scaled)[0]

            # Get confidence (probability of predicted class)
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]

            # Validate confidence value
            if (confidence == float('inf') or confidence == float('-inf') or
                confidence != confidence):  # NaN check
                self.logger.warning(f"Invalid confidence value: {confidence}")
                confidence = 0.0

            # Emit signal
            self.prediction_made.emit(image_path, prediction, confidence)

            return prediction, confidence

        except Exception as e:
            self.logger.error(f"Error predicting rating for {image_path}: {e}")
            return None, 0.0

    def save_model(self, model_path: Path) -> bool:
        """
        Save the trained model to disk.

        Args:
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            self.logger.warning("No trained model to save")
            return False

        try:
            # Ensure directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model components
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'accuracy': self.accuracy,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'training_data_count': self.training_data_count
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Model saved to {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, model_path: Path) -> bool:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the model file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Restore model components
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.accuracy = model_data['accuracy']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.training_data_count = model_data.get('training_data_count', 0)

            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(f"Model accuracy: {self.accuracy:.3f}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        return {
            'is_trained': self.is_trained,
            'accuracy': self.accuracy,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'classifier_type': type(self.classifier).__name__,
            'feature_names': self.feature_names
        }

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self.feature_cache.clear()
        self.logger.info("Feature cache cleared")
