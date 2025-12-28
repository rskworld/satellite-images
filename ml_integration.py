#!/usr/bin/env python3
"""
Satellite Image Dataset - Machine Learning Integration
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

ML features including:
- Land cover classification
- Building detection
- Change detection
- Feature extraction for ML models
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import cv2


class SatelliteMLProcessor:
    """
    Machine learning processor for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self):
        """Initialize the ML processor."""
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_extractor = None
    
    def extract_features_for_ml(self, image: np.ndarray, 
                                patch_size: int = 32,
                                stride: int = 16) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract features from image patches for ML training.
        
        Args:
            image: Input image
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            
        Returns:
            Tuple of (feature matrix, patch positions)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features = []
        positions = []
        
        h, w = gray.shape
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = gray[y:y+patch_size, x:x+patch_size]
                
                # Extract simple features
                patch_features = [
                    np.mean(patch),
                    np.std(patch),
                    np.min(patch),
                    np.max(patch),
                    np.median(patch),
                    # Texture features
                    np.mean(np.gradient(patch)[0]),
                    np.mean(np.gradient(patch)[1]),
                ]
                
                features.append(patch_features)
                positions.append((x, y))
        
        return np.array(features), positions
    
    def train_land_cover_classifier(self, images: List[np.ndarray],
                                   labels: List[np.ndarray],
                                   test_size: float = 0.2,
                                   n_estimators: int = 100) -> Dict:
        """
        Train a land cover classification model.
        
        Args:
            images: List of training images
            labels: List of label arrays (same size as images)
            test_size: Proportion of data for testing
            n_estimators: Number of trees in Random Forest
            
        Returns:
            Dictionary with training results and metrics
        """
        print("Training Land Cover Classifier")
        print("Created by: RSK World (https://rskworld.in)")
        print("-" * 50)
        
        # Extract features and labels
        X = []
        y = []
        
        for image, label in zip(images, labels):
            features, positions = self.extract_features_for_ml(image)
            X.append(features)
            
            # Get labels for each patch
            patch_labels = []
            for x, y_pos in positions:
                # Get label from center of patch
                center_x = x + 16
                center_y = y_pos + 16
                if center_y < label.shape[0] and center_x < label.shape[1]:
                    patch_labels.append(label[center_y, center_x])
                else:
                    patch_labels.append(0)
            
            y.extend(patch_labels)
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"Training on {len(X_train)} samples...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        y_pred = self.classifier.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'classification_report': report,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        return results
    
    def predict_land_cover(self, image: np.ndarray) -> np.ndarray:
        """
        Predict land cover for an image.
        
        Args:
            image: Input image
            
        Returns:
            Predicted land cover map
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_land_cover_classifier first.")
        
        features, positions = self.extract_features_for_ml(image)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.classifier.predict(features_scaled)
        
        # Create prediction map
        h, w = image.shape[:2]
        prediction_map = np.zeros((h, w), dtype=np.uint8)
        
        for (x, y), pred in zip(positions, predictions):
            patch_size = 32
            prediction_map[y:y+patch_size, x:x+patch_size] = pred
        
        return prediction_map
    
    def detect_buildings_simple(self, image: np.ndarray,
                               min_area: int = 100,
                               max_area: int = 10000) -> List[Dict]:
        """
        Simple building detection using image processing.
        
        Args:
            image: Input image
            min_area: Minimum building area in pixels
            max_area: Maximum building area in pixels
            
        Returns:
            List of building detections with bboxes
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buildings = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                confidence = min(0.95, 0.5 + (area / max_area) * 0.45)
                
                buildings.append({
                    'id': f'building_{i}',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': float(area),
                    'confidence': round(confidence, 2),
                    'aspect_ratio': round(aspect_ratio, 2)
                })
        
        return buildings
    
    def detect_changes(self, image1: np.ndarray, image2: np.ndarray,
                      threshold: float = 0.3) -> Tuple[np.ndarray, Dict]:
        """
        Detect changes between two images.
        
        Args:
            image1: First image (before)
            image2: Second image (after)
            threshold: Threshold for change detection
            
        Returns:
            Tuple of (change map, statistics)
        """
        # Ensure same size
        if image1.shape != image2.shape:
            h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
            image1 = image1[:h, :w]
            image2 = image2[:h, :w]
        
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = image1
            gray2 = image2
        
        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        diff_normalized = diff.astype(np.float32) / 255.0
        
        # Create change map
        change_map = (diff_normalized > threshold).astype(np.uint8) * 255
        
        # Calculate statistics
        change_pixels = np.sum(change_map > 0)
        total_pixels = change_map.size
        change_percentage = (change_pixels / total_pixels) * 100
        
        stats = {
            'change_pixels': int(change_pixels),
            'total_pixels': int(total_pixels),
            'change_percentage': round(change_percentage, 2),
            'mean_difference': float(np.mean(diff_normalized)),
            'max_difference': float(np.max(diff_normalized))
        }
        
        return change_map, stats
    
    def extract_ndvi(self, image: np.ndarray) -> np.ndarray:
        """
        Extract NDVI (Normalized Difference Vegetation Index) from image.
        Assumes image has NIR band or uses red/green approximation.
        
        Args:
            image: Input image (should have NIR band ideally)
            
        Returns:
            NDVI map
        """
        if len(image.shape) == 3:
            # Approximate NDVI using RGB (not ideal but works for demo)
            # In real scenario, you'd have NIR band
            red = image[:, :, 0].astype(np.float32)
            green = image[:, :, 1].astype(np.float32)
            blue = image[:, :, 2].astype(np.float32)
            
            # Use green as approximation for NIR
            nir = green
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-7)
            ndvi = np.clip(ndvi, -1, 1)
            
            # Normalize to 0-255
            ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
            
            return ndvi_normalized
        else:
            raise ValueError("Image must be RGB or multispectral")
    
    def create_training_dataset(self, images: List[np.ndarray],
                               labels: List[np.ndarray],
                               output_dir: str = "data/ml_training") -> Dict:
        """
        Create a training dataset from images and labels.
        
        Args:
            images: List of images
            labels: List of label arrays
            output_dir: Directory to save dataset
            
        Returns:
            Dictionary with dataset information
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'n_samples': len(images),
            'image_shape': images[0].shape if images else None,
            'files': []
        }
        
        for i, (image, label) in enumerate(zip(images, labels)):
            # Save image
            img_file = output_path / f"image_{i:04d}.png"
            cv2.imwrite(str(img_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save label
            label_file = output_path / f"label_{i:04d}.png"
            cv2.imwrite(str(label_file), label)
            
            dataset_info['files'].append({
                'image': str(img_file),
                'label': str(label_file)
            })
        
        # Save dataset info
        info_file = output_path / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        return dataset_info


def main():
    """
    Example usage of SatelliteMLProcessor.
    Created by: RSK World (https://rskworld.in)
    """
    print("Machine Learning Integration - Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    processor = SatelliteMLProcessor()
    
    # Create sample data
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sample_label = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    
    # Extract features
    features, positions = processor.extract_features_for_ml(sample_image)
    print(f"Extracted {len(features)} feature vectors")
    
    # Building detection
    buildings = processor.detect_buildings_simple(sample_image)
    print(f"Detected {len(buildings)} buildings")
    
    # NDVI extraction
    ndvi = processor.extract_ndvi(sample_image)
    print(f"NDVI map shape: {ndvi.shape}")
    
    print("\nML processing complete!")


if __name__ == "__main__":
    main()

