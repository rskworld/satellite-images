#!/usr/bin/env python3
"""
Satellite Image Dataset - Image Processing Script
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script provides utilities for processing satellite images including:
- Loading and preprocessing images
- Extracting geospatial metadata
- Processing land cover labels
- Building detection utilities
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt


class SatelliteImageProcessor:
    """
    Main class for processing satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the processor with data directory.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.metadata_dir = self.data_dir / "metadata"
        
    def load_image(self, image_path: str, format: str = "png") -> np.ndarray:
        """
        Load a satellite image from file.
        
        Args:
            image_path: Path to the image file
            format: Image format (png, tiff, geotiff)
            
        Returns:
            Image as numpy array
        """
        path = Path(image_path)
        
        if format.lower() == "geotiff" or path.suffix.lower() == ".tif":
            return self._load_geotiff(str(path))
        else:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not load image from {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _load_geotiff(self, path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load GeoTIFF file with metadata.
        
        Args:
            path: Path to GeoTIFF file
            
        Returns:
            Tuple of (image array, metadata dictionary)
        """
        with rasterio.open(path) as src:
            image = src.read()
            metadata = {
                'crs': str(src.crs),
                'transform': src.transform,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
                'count': src.count
            }
            # Convert to RGB if multi-band
            if image.shape[0] == 1:
                image = image[0]
            elif image.shape[0] >= 3:
                image = np.transpose(image[:3], (1, 2, 0))
            
        return image, metadata
    
    def load_labels(self, label_path: str) -> Dict:
        """
        Load land cover classification labels.
        
        Args:
            label_path: Path to label JSON file
            
        Returns:
            Dictionary containing label data
        """
        with open(label_path, 'r') as f:
            return json.load(f)
    
    def preprocess_image(self, image: np.ndarray, 
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess satellite image.
        
        Args:
            image: Input image array
            target_size: Target size (width, height) for resizing
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        if target_size:
            processed = cv2.resize(processed, target_size, 
                                 interpolation=cv2.INTER_LINEAR)
        
        if normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract basic features from satellite image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image))
        }
        
        if len(image.shape) == 3:
            features['channels'] = image.shape[2]
            for i in range(image.shape[2]):
                features[f'channel_{i}_mean'] = float(np.mean(image[:, :, i]))
        
        return features
    
    def visualize_image(self, image: np.ndarray, 
                       labels: Optional[Dict] = None,
                       save_path: Optional[str] = None):
        """
        Visualize satellite image with optional labels overlay.
        
        Args:
            image: Image array to visualize
            labels: Optional label data for overlay
            save_path: Optional path to save the visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title('Satellite Image', fontsize=16, fontweight='bold')
        
        if labels:
            # Overlay land cover regions if available
            if 'regions' in labels:
                for region in labels['regions']:
                    if 'polygon' in region:
                        polygon = np.array(region['polygon'])
                        ax.plot(polygon[:, 0], polygon[:, 1], 
                               'r-', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def main():
    """
    Example usage of the SatelliteImageProcessor.
    Created by: RSK World (https://rskworld.in)
    """
    processor = SatelliteImageProcessor()
    
    # Example: Process a sample image
    print("Satellite Image Dataset - Processing Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    # Check if sample data exists
    sample_image = processor.images_dir / "sample_001.png"
    if sample_image.exists():
        # Load image
        image = processor.load_image(str(sample_image))
        print(f"Loaded image: {image.shape}")
        
        # Extract features
        features = processor.extract_features(image)
        print(f"Features: {features}")
        
        # Preprocess
        processed = processor.preprocess_image(image, normalize=True)
        print(f"Processed image shape: {processed.shape}")
        
        # Visualize
        processor.visualize_image(image)
    else:
        print("Sample image not found. Please add images to data/images/ directory.")


if __name__ == "__main__":
    main()

