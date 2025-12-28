#!/usr/bin/env python3
"""
Satellite Image Dataset - Machine Learning Features
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Machine learning utilities including:
- Feature extraction
- Classification helpers
- Model training utilities
- Prediction functions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.feature_extraction import image as skimage_feature
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2


class MLFeatureExtractor:
    """
    Machine learning feature extraction for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.scaler = StandardScaler()
        self.pca = None
    
    def extract_handcrafted_features(self, image: np.ndarray) -> Dict:
        """
        Extract handcrafted features from satellite image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = float(np.mean(image))
        features['std'] = float(np.std(image))
        features['min'] = float(np.min(image))
        features['max'] = float(np.max(image))
        features['median'] = float(np.median(image))
        
        # Color statistics (if multi-channel)
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                features[f'channel_{i}_mean'] = float(np.mean(image[:, :, i]))
                features[f'channel_{i}_std'] = float(np.std(image[:, :, i]))
        
        # Texture features
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = float(np.mean(gradient_magnitude))
        features['gradient_std'] = float(np.std(gradient_magnitude))
        features['gradient_max'] = float(np.max(gradient_magnitude))
        
        # Histogram features
        hist, _ = np.histogram(gray.flatten(), bins=32, range=(0, 1))
        hist = hist / hist.sum()  # Normalize
        features['histogram_entropy'] = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
        features['histogram_skewness'] = float(self._calculate_skewness(hist))
        features['histogram_kurtosis'] = float(self._calculate_kurtosis(hist))
        
        # Local binary pattern-like features
        features['local_variance'] = float(np.var(cv2.GaussianBlur(gray, (5, 5), 0)))
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4)) - 3.0
    
    def extract_patch_features(self, image: np.ndarray, patch_size: int = 32, 
                               stride: int = 16) -> np.ndarray:
        """
        Extract features from image patches.
        
        Args:
            image: Input image
            patch_size: Size of patches
            stride: Stride for patch extraction
            
        Returns:
            Array of patch features
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        patches = skimage_feature.extract_patches_2d(gray, (patch_size, patch_size))
        
        # Extract features for each patch
        patch_features = []
        for patch in patches:
            features = [
                np.mean(patch),
                np.std(patch),
                np.var(patch),
                np.min(patch),
                np.max(patch)
            ]
            patch_features.append(features)
        
        return np.array(patch_features)
    
    def extract_deep_features(self, image: np.ndarray, method: str = 'histogram') -> np.ndarray:
        """
        Extract deep learning-style features.
        
        Args:
            image: Input image
            method: Feature extraction method
            
        Returns:
            Feature vector
        """
        if method == 'histogram':
            # Multi-scale histogram features
            features = []
            for scale in [1, 2, 4]:
                scaled = cv2.resize(image, 
                                   (image.shape[1] // scale, image.shape[0] // scale))
                if len(scaled.shape) == 3:
                    for c in range(scaled.shape[2]):
                        hist, _ = np.histogram(scaled[:, :, c].flatten(), bins=16)
                        features.extend(hist / hist.sum())
                else:
                    hist, _ = np.histogram(scaled.flatten(), bins=16)
                    features.extend(hist / hist.sum())
            return np.array(features)
        
        elif method == 'pca':
            # PCA-based features
            if len(image.shape) == 3:
                data = image.reshape(-1, image.shape[2])
            else:
                data = image.reshape(-1, 1)
            
            if self.pca is None:
                self.pca = PCA(n_components=50)
                data_scaled = self.scaler.fit_transform(data)
                self.pca.fit(data_scaled)
            
            data_scaled = self.scaler.transform(data)
            features = self.pca.transform(data_scaled)
            return features.mean(axis=0)  # Average over spatial dimensions
        
        else:
            # Default: flatten and sample
            if len(image.shape) == 3:
                flat = image.reshape(-1, image.shape[2])
            else:
                flat = image.reshape(-1, 1)
            
            # Sample features
            indices = np.linspace(0, len(flat) - 1, 100, dtype=int)
            return flat[indices].flatten()
    
    def cluster_image(self, image: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster image pixels into regions.
        
        Args:
            image: Input image
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (clustered image, cluster centers)
        """
        if len(image.shape) == 3:
            data = image.reshape(-1, image.shape[2])
        else:
            data = image.reshape(-1, 1)
        
        # Normalize
        data_scaled = self.scaler.fit_transform(data)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        
        # Reshape labels
        clustered = labels.reshape(image.shape[:2])
        
        return clustered, kmeans.cluster_centers_
    
    def classify_land_cover(self, image: np.ndarray, 
                           feature_vector: Optional[np.ndarray] = None) -> Dict:
        """
        Simple land cover classification based on features.
        
        Args:
            image: Input image
            feature_vector: Optional pre-computed feature vector
            
        Returns:
            Classification results dictionary
        """
        if feature_vector is None:
            features = self.extract_handcrafted_features(image)
            feature_vector = np.array(list(features.values()))
        
        # Simple rule-based classification (can be replaced with trained model)
        if len(image.shape) == 3:
            mean_values = np.mean(image, axis=(0, 1))
        else:
            mean_values = np.array([np.mean(image)])
        
        # Classify based on color characteristics
        classifications = []
        confidences = []
        
        # Water (blue/dark)
        if mean_values[0] < 0.3 and len(mean_values) > 2:
            classifications.append("water")
            confidences.append(0.7)
        
        # Vegetation (green)
        if len(mean_values) > 1 and mean_values[1] > 0.4:
            classifications.append("vegetation")
            confidences.append(0.8)
        
        # Urban (bright, high variance)
        if np.mean(mean_values) > 0.5 and np.std(image) > 0.2:
            classifications.append("urban")
            confidences.append(0.75)
        
        # Barren (brown/yellow)
        if len(mean_values) > 2 and mean_values[0] > 0.4 and mean_values[1] < 0.4:
            classifications.append("barren")
            confidences.append(0.7)
        
        if not classifications:
            classifications.append("unknown")
            confidences.append(0.5)
        
        return {
            'classes': classifications,
            'confidences': confidences,
            'primary_class': classifications[0],
            'primary_confidence': confidences[0]
        }


class ModelTrainer:
    """
    Helper class for training ML models on satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    @staticmethod
    def prepare_training_data(images: List[np.ndarray], 
                             labels: List[str],
                             feature_extractor: MLFeatureExtractor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from images and labels.
        
        Args:
            images: List of image arrays
            labels: List of label strings
            feature_extractor: Feature extractor instance
            
        Returns:
            Tuple of (feature matrix, label array)
        """
        features_list = []
        labels_list = []
        
        for image, label in zip(images, labels):
            features = feature_extractor.extract_handcrafted_features(image)
            feature_vector = np.array(list(features.values()))
            features_list.append(feature_vector)
            labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        return X, y


def example_usage():
    """Example usage of ML features."""
    print("Machine Learning Features - Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    extractor = MLFeatureExtractor()
    
    # Create sample image
    sample_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Extract handcrafted features
    features = extractor.extract_handcrafted_features(sample_image)
    print(f"Extracted {len(features)} features")
    print(f"Sample features: mean={features['mean']:.3f}, std={features['std']:.3f}")
    
    # Classify land cover
    classification = extractor.classify_land_cover(sample_image)
    print(f"Classification: {classification['primary_class']} "
          f"(confidence: {classification['primary_confidence']:.2f})")
    
    # Cluster image
    clustered, centers = extractor.cluster_image(sample_image, n_clusters=5)
    print(f"Clustered image shape: {clustered.shape}")
    print(f"Number of cluster centers: {len(centers)}")
    
    print("\nML features example completed!")


if __name__ == "__main__":
    example_usage()

