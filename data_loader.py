#!/usr/bin/env python3
"""
Satellite Image Dataset - Data Loader
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides data loading utilities for the satellite image dataset.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import rasterio


class SatelliteDatasetLoader:
    """
    Data loader for satellite image dataset.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self, root_dir: str = "data"):
        """
        Initialize the dataset loader.
        
        Args:
            root_dir: Root directory of the dataset
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        self.building_dir = self.root_dir / "building_detection"
        self.metadata_dir = self.root_dir / "metadata"
        
    def get_image_list(self) -> List[str]:
        """
        Get list of all available images.
        
        Returns:
            List of image filenames
        """
        if not self.images_dir.exists():
            return []
        
        image_extensions = ['.png', '.tiff', '.tif', '.jpg', '.jpeg']
        images = []
        for ext in image_extensions:
            images.extend(list(self.images_dir.glob(f"*{ext}")))
        
        return [img.name for img in images]
    
    def load_image_pair(self, image_id: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Load image and corresponding labels.
        
        Args:
            image_id: Image identifier (filename without extension)
            
        Returns:
            Tuple of (image array, labels dictionary)
        """
        # Find image file
        image_path = None
        for ext in ['.png', '.tiff', '.tif', '.jpg', '.jpeg']:
            potential_path = self.images_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path is None:
            return None, None
        
        # Load image
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            try:
                with rasterio.open(str(image_path)) as src:
                    image = src.read()
                    if image.shape[0] >= 3:
                        image = np.transpose(image[:3], (1, 2, 0))
                    else:
                        image = image[0]
            except:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.labels_dir / f"{image_id}.json"
        labels = None
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = json.load(f)
        
        return image, labels
    
    def load_building_detections(self, image_id: str) -> Optional[Dict]:
        """
        Load building detection annotations.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Building detection data dictionary
        """
        building_path = self.building_dir / f"{image_id}.json"
        if not building_path.exists():
            return None
        
        with open(building_path, 'r') as f:
            return json.load(f)
    
    def load_metadata(self, image_id: str) -> Optional[Dict]:
        """
        Load geospatial metadata.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Metadata dictionary
        """
        metadata_path = self.metadata_dir / f"{image_id}.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        images = self.get_image_list()
        
        info = {
            'total_images': len(images),
            'images': images,
            'has_labels': self.labels_dir.exists() and len(list(self.labels_dir.glob("*.json"))) > 0,
            'has_buildings': self.building_dir.exists() and len(list(self.building_dir.glob("*.json"))) > 0,
            'has_metadata': self.metadata_dir.exists() and len(list(self.metadata_dir.glob("*.json"))) > 0
        }
        
        return info


def example_usage():
    """
    Example usage of the dataset loader.
    Created by: RSK World (https://rskworld.in)
    """
    print("Satellite Image Dataset - Data Loader Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    loader = SatelliteDatasetLoader()
    
    # Get dataset info
    info = loader.get_dataset_info()
    print(f"Total images: {info['total_images']}")
    print(f"Has labels: {info['has_labels']}")
    print(f"Has building detections: {info['has_buildings']}")
    print(f"Has metadata: {info['has_metadata']}")
    
    # Load a sample image if available
    if info['total_images'] > 0:
        image_id = Path(info['images'][0]).stem
        image, labels = loader.load_image_pair(image_id)
        
        if image is not None:
            print(f"\nLoaded image: {image_id}")
            print(f"Image shape: {image.shape}")
            
            if labels:
                print(f"Labels: {len(labels.get('regions', []))} regions")
        
        # Load building detections
        buildings = loader.load_building_detections(image_id)
        if buildings:
            print(f"Buildings detected: {len(buildings.get('buildings', []))}")


if __name__ == "__main__":
    example_usage()

