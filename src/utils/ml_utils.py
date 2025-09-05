"""
Machine learning utility functions for the Image Classifier application.

This module provides utilities for model evaluation, feature preprocessing,
and machine learning pipeline management.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)


def normalize_features(features: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.

    Args:
        features: Feature array to normalize
        scaler: Optional pre-fitted scaler

    Returns:
        Tuple of (normalized_features, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = scaler.transform(features)

    return normalized_features, scaler


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
    """
    Evaluate a machine learning model using cross-validation.

    Args:
        model: Trained model to evaluate
        X: Feature matrix
        y: Target labels
        cv_folds: Number of cross-validation folds

    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Cross-validation accuracy
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')

        # Split data for additional evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train on training set and predict on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)

        results = {
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores.tolist()
        }

        logger.info(f"Model evaluation completed: CV accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        return results

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {
            'cv_mean_accuracy': 0.0,
            'cv_std_accuracy': 0.0,
            'test_accuracy': 0.0,
            'cv_scores': [],
            'error': str(e)
        }


def save_model(model: Any, file_path: Path) -> bool:
    """
    Save a trained model to disk.

    Args:
        model: Model to save
        file_path: Path where to save the model

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Model saved successfully to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        return False


def load_model(file_path: Path) -> Optional[Any]:
    """
    Load a trained model from disk.

    Args:
        file_path: Path to the saved model

    Returns:
        Loaded model or None if loading fails
    """
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded successfully from {file_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        return None


def calculate_feature_importance(model: Any, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Optional list of feature names

    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}

        importances = model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances))

        # Sort by importance (descending)
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Feature importance calculated successfully")
        return dict(sorted_features)

    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {}


def validate_features(features: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """
    Validate feature array for common issues.

    Args:
        features: Feature array to validate
        expected_shape: Expected shape for validation

    Returns:
        True if features are valid, False otherwise
    """
    try:
        # Check if features is numpy array
        if not isinstance(features, np.ndarray):
            logger.error("Features must be a numpy array")
            return False

        # Check for NaN values
        if np.any(np.isnan(features)):
            logger.error("Features contain NaN values")
            return False

        # Check for infinite values
        if np.any(np.isinf(features)):
            logger.error("Features contain infinite values")
            return False

        # Check shape if expected_shape is provided
        if expected_shape is not None:
            if features.shape != expected_shape:
                logger.error(f"Features shape {features.shape} does not match expected {expected_shape}")
                return False

        # Check for empty features
        if features.size == 0:
            logger.error("Features array is empty")
            return False

        logger.debug("Features validation passed")
        return True

    except Exception as e:
        logger.error(f"Error validating features: {e}")
        return False


def balance_dataset(X: np.ndarray, y: np.ndarray, method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset to handle class imbalance.

    Args:
        X: Feature matrix
        y: Target labels
        method: Balancing method ('undersample' or 'oversample')

    Returns:
        Tuple of (balanced_X, balanced_y)
    """
    try:
        unique_labels, counts = np.unique(y, return_counts=True)

        if len(unique_labels) <= 1:
            logger.warning("Only one class found, cannot balance dataset")
            return X, y

        min_count = np.min(counts)
        max_count = np.max(counts)

        if min_count == max_count:
            logger.info("Dataset is already balanced")
            return X, y

        logger.info(f"Dataset imbalance: min class has {min_count} samples, max class has {max_count} samples")

        if method == 'undersample':
            return _undersample_dataset(X, y, min_count)
        elif method == 'oversample':
            return _oversample_dataset(X, y, max_count)
        else:
            logger.warning(f"Unknown balancing method: {method}. Using undersampling.")
            return _undersample_dataset(X, y, min_count)

    except Exception as e:
        logger.error(f"Error balancing dataset: {e}")
        return X, y


def _undersample_dataset(X: np.ndarray, y: np.ndarray, target_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample dataset to balance classes."""
    balanced_X = []
    balanced_y = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]

        if len(label_indices) > target_count:
            # Randomly select target_count samples
            selected_indices = np.random.choice(label_indices, target_count, replace=False)
        else:
            selected_indices = label_indices

        balanced_X.append(X[selected_indices])
        balanced_y.append(y[selected_indices])

    return np.vstack(balanced_X), np.hstack(balanced_y)


def _oversample_dataset(X: np.ndarray, y: np.ndarray, target_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample dataset to balance classes."""
    balanced_X = []
    balanced_y = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]

        if len(label_indices) < target_count:
            # Repeat samples to reach target_count
            repeat_count = target_count // len(label_indices)
            remainder = target_count % len(label_indices)

            # Repeat samples
            repeated_indices = np.tile(label_indices, repeat_count)

            # Add random samples for remainder
            if remainder > 0:
                additional_indices = np.random.choice(label_indices, remainder, replace=True)
                repeated_indices = np.concatenate([repeated_indices, additional_indices])
        else:
            repeated_indices = label_indices

        balanced_X.append(X[repeated_indices])
        balanced_y.append(y[repeated_indices])

    return np.vstack(balanced_X), np.hstack(balanced_y)


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get information about a trained model.

    Args:
        model: Trained model

    Returns:
        Dictionary containing model information
    """
    try:
        info = {
            'model_type': type(model).__name__,
            'model_module': type(model).__module__,
        }

        # Add model-specific attributes
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators

        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth

        if hasattr(model, 'random_state'):
            info['random_state'] = model.random_state

        if hasattr(model, 'n_features_in_'):
            info['n_features_in'] = model.n_features_in_

        if hasattr(model, 'classes_'):
            info['n_classes'] = len(model.classes_)
            info['classes'] = model.classes_.tolist()

        logger.debug(f"Model info retrieved: {info}")
        return info

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {'error': str(e)}
