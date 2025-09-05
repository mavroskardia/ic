"""
Image utility functions for the Image Classifier application.

This module provides image processing and feature extraction capabilities
optimized for human subject photography.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image from file using OpenCV.

    Args:
        image_path: Path to the image file

    Returns:
        Loaded image as numpy array, or None if loading fails
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: Input image as numpy array
        max_size: Maximum dimension size

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    if max(height, width) <= max_size:
        return image

    # Calculate new dimensions
    if height > width:
        new_height = max_size
        new_width = int(width * max_size / height)
    else:
        new_width = max_size
        new_height = int(height * max_size / width)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def extract_color_features(image: np.ndarray) -> np.ndarray:
    """
    Extract color histogram features from image.
    Optimized for human subject photography.

    Args:
        image: Input image as BGR numpy array

    Returns:
        Color feature vector
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract color histograms for each channel
        hist_r = cv2.calcHist([rgb_image], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([rgb_image], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([rgb_image], [2], None, [32], [0, 256]).flatten()

        # Convert BGR to HSV for additional color features
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv_image], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv_image], [2], None, [32], [0, 256]).flatten()

        # Normalize histograms
        hist_r = hist_r / (hist_r.sum() + 1e-8)
        hist_g = hist_g / (hist_g.sum() + 1e-8)
        hist_b = hist_b / (hist_b.sum() + 1e-8)
        hist_h = hist_h / (hist_h.sum() + 1e-8)
        hist_s = hist_s / (hist_s.sum() + 1e-8)
        hist_v = hist_v / (hist_v.sum() + 1e-8)

        # Combine all color features
        color_features = np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v])

        return color_features

    except Exception as e:
        logger.error(f"Error extracting color features: {e}")
        # Return zero features if extraction fails
        return np.zeros(192)  # 6 channels * 32 bins


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """
    Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
    Optimized for human subject patterns.

    Args:
        image: Input image as BGR numpy array

    Returns:
        Texture feature vector
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize for consistent feature extraction
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        # Calculate GLCM for different angles and distances
        angles = [0, 45, 90, 135]
        distances = [1, 2, 3]

        texture_features = []

        for distance in distances:
            for angle in angles:
                # Calculate GLCM
                glcm = _calculate_glcm(gray, distance, angle)

                # Extract texture properties
                contrast = _glcm_contrast(glcm)
                homogeneity = _glcm_homogeneity(glcm)
                energy = _glcm_energy(glcm)
                correlation = _glcm_correlation(glcm)

                texture_features.extend([contrast, homogeneity, energy, correlation])

        # Normalize features
        texture_features = np.array(texture_features)
        texture_features = (texture_features - texture_features.mean()) / (texture_features.std() + 1e-8)

        return texture_features

    except Exception as e:
        logger.error(f"Error extracting texture features: {e}")
        # Return zero features if extraction fails
        return np.zeros(48)  # 4 distances * 4 angles * 4 properties


def extract_edge_features(image: np.ndarray) -> np.ndarray:
    """
    Extract edge detection features optimized for human contours.

    Args:
        image: Input image as BGR numpy array

    Returns:
        Edge feature vector
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Calculate edge density in different regions
        height, width = edges.shape
        regions = 4  # 2x2 grid

        edge_features = []

        for i in range(regions):
            for j in range(regions):
                # Extract region
                y_start = i * height // regions
                y_end = (i + 1) * height // regions
                x_start = j * width // regions
                x_end = (j + 1) * width // regions

                region = edges[y_start:y_end, x_start:x_end]

                # Calculate edge density
                edge_density = np.sum(region > 0) / region.size
                edge_features.append(edge_density)

        # Add overall edge density
        total_edge_density = np.sum(edges > 0) / edges.size
        edge_features.append(total_edge_density)

        # Normalize features
        edge_features = np.array(edge_features)
        edge_features = (edge_features - edge_features.mean()) / (edge_features.std() + 1e-8)

        return edge_features

    except Exception as e:
        logger.error(f"Error extracting edge features: {e}")
        # Return zero features if extraction fails
        return np.zeros(17)  # 16 regions + 1 overall


def extract_shape_features(image: np.ndarray) -> np.ndarray:
    """
    Extract basic shape characteristics optimized for human proportions.

    Args:
        image: Input image as BGR numpy array

    Returns:
        Shape feature vector
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(8)

        # Get the largest contour (assumed to be the main subject)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / (h + 1e-8)

        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-8)

        # Extent (area / bounding rectangle area)
        extent = area / (w * h + 1e-8)

        # Solidity (area / convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)

        # Compactness
        compactness = (perimeter * perimeter) / (4 * np.pi * area + 1e-8)

        # Normalize features
        shape_features = np.array([
            area / (image.shape[0] * image.shape[1]),  # Normalized area
            perimeter / (image.shape[0] + image.shape[1]),  # Normalized perimeter
            aspect_ratio,
            circularity,
            extent,
            solidity,
            compactness,
            w * h / (image.shape[0] * image.shape[1])  # Normalized bounding box area
        ])

        # Clip extreme values
        shape_features = np.clip(shape_features, 0, 1)

        return shape_features

    except Exception as e:
        logger.error(f"Error extracting shape features: {e}")
        # Return zero features if extraction fails
        return np.zeros(8)


def extract_face_features(image: np.ndarray) -> np.ndarray:
    """
    Extract face detection features if available.

    Args:
        image: Input image as BGR numpy array

    Returns:
        Face feature vector
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if face_cascade.empty():
            logger.warning("Face cascade classifier not available")
            return np.zeros(6)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            # No faces detected
            return np.array([0, 0, 0, 0, 0, 0])

        # Calculate face features
        num_faces = len(faces)

        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face

        # Face size relative to image
        face_area = w * h
        image_area = image.shape[0] * image.shape[1]
        face_size_ratio = face_area / image_area

        # Face position (normalized)
        face_center_x = (x + w/2) / image.shape[1]
        face_center_y = (y + h/2) / image.shape[0]

        # Face aspect ratio
        face_aspect_ratio = w / h

        # Average face size
        avg_face_size = np.mean([f[2] * f[3] for f in faces]) / image_area

        face_features = np.array([
            num_faces,
            face_size_ratio,
            face_center_x,
            face_center_y,
            face_aspect_ratio,
            avg_face_size
        ])

        return face_features

    except Exception as e:
        logger.error(f"Error extracting face features: {e}")
        # Return zero features if extraction fails
        return np.zeros(6)


def extract_all_features(image: np.ndarray) -> np.ndarray:
    """
    Extract all image features for classification.

    Args:
        image: Input image as BGR numpy array

    Returns:
        Combined feature vector
    """
    try:
        # Resize image for consistent feature extraction
        resized_image = resize_image(image, max_size=800)

        # Extract all feature types
        color_features = extract_color_features(resized_image)
        texture_features = extract_texture_features(resized_image)
        edge_features = extract_edge_features(resized_image)
        shape_features = extract_shape_features(resized_image)
        face_features = extract_face_features(resized_image)

        # Combine all features
        all_features = np.concatenate([
            color_features,      # 192 features
            texture_features,    # 48 features
            edge_features,       # 17 features
            shape_features,      # 8 features
            face_features        # 6 features
        ])

        # Total: 271 features
        return all_features

    except Exception as e:
        logger.error(f"Error extracting all features: {e}")
        # Return zero features if extraction fails
        return np.zeros(271)


# Helper functions for GLCM calculations
def _calculate_glcm(gray_image: np.ndarray, distance: int, angle: int) -> np.ndarray:
    """Calculate Gray-Level Co-occurrence Matrix."""
    # This is a simplified GLCM calculation
    # In production, consider using scikit-image's graycomatrix

    levels = 8  # Reduce levels for efficiency
    gray_quantized = (gray_image // (256 // levels)).astype(np.uint8)

    height, width = gray_quantized.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    # Calculate co-occurrence matrix
    for i in range(height):
        for j in range(width):
            if angle == 0:  # Horizontal
                if j + distance < width:
                    glcm[gray_quantized[i, j], gray_quantized[i, j + distance]] += 1
            elif angle == 45:  # Diagonal
                if i - distance >= 0 and j + distance < width:
                    glcm[gray_quantized[i, j], gray_quantized[i - distance, j + distance]] += 1
            elif angle == 90:  # Vertical
                if i + distance < height:
                    glcm[gray_quantized[i, j], gray_quantized[i + distance, j]] += 1
            elif angle == 135:  # Diagonal
                if i + distance < height and j + distance < width:
                    glcm[gray_quantized[i, j], gray_quantized[i + distance, j + distance]] += 1

    # Normalize
    if glcm.sum() > 0:
        glcm = glcm / glcm.sum()

    return glcm


def _glcm_contrast(glcm: np.ndarray) -> float:
    """Calculate GLCM contrast."""
    contrast = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            contrast += glcm[i, j] * (i - j) ** 2
    return contrast


def _glcm_homogeneity(glcm: np.ndarray) -> float:
    """Calculate GLCM homogeneity."""
    homogeneity = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
    return homogeneity


def _glcm_energy(glcm: np.ndarray) -> float:
    """Calculate GLCM energy."""
    return np.sum(glcm ** 2)


def _glcm_correlation(glcm: np.ndarray) -> float:
    """Calculate GLCM correlation."""
    # Simplified correlation calculation
    mean_i = np.sum(np.sum(glcm, axis=1) * np.arange(glcm.shape[0]))
    mean_j = np.sum(np.sum(glcm, axis=0) * np.arange(glcm.shape[1]))

    correlation = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            correlation += glcm[i, j] * (i - mean_i) * (j - mean_j)

    return correlation
