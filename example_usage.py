#!/usr/bin/env python3
"""
Satellite Image Dataset - Example Usage
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates how to use the satellite image dataset.
"""

from data_loader import SatelliteDatasetLoader
from process_images import SatelliteImageProcessor
from visualize import visualize_land_cover, visualize_buildings
import numpy as np


def main():
    """
    Main example function.
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("Satellite Image Dataset - Example Usage")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    
    # Initialize components
    loader = SatelliteDatasetLoader()
    processor = SatelliteImageProcessor()
    
    # Get dataset information
    print("1. Dataset Information:")
    print("-" * 40)
    info = loader.get_dataset_info()
    print(f"   Total images: {info['total_images']}")
    print(f"   Has labels: {info['has_labels']}")
    print(f"   Has building detections: {info['has_buildings']}")
    print(f"   Has metadata: {info['has_metadata']}")
    print()
    
    # Example: Load an image (if available)
    if info['total_images'] > 0:
        image_id = "sample_001"
        print(f"2. Loading image: {image_id}")
        print("-" * 40)
        
        image, labels = loader.load_image_pair(image_id)
        
        if image is not None:
            print(f"   Image shape: {image.shape}")
            print(f"   Image dtype: {image.dtype}")
            
            # Extract features
            features = processor.extract_features(image)
            print(f"   Mean pixel value: {features['mean']:.2f}")
            print(f"   Standard deviation: {features['std']:.2f}")
            print()
            
            # Process labels if available
            if labels:
                print(f"3. Land Cover Labels:")
                print("-" * 40)
                print(f"   Classes: {labels.get('classes', [])}")
                print(f"   Number of regions: {len(labels.get('regions', []))}")
                print()
            
            # Load building detections if available
            buildings = loader.load_building_detections(image_id)
            if buildings:
                print(f"4. Building Detections:")
                print("-" * 40)
                print(f"   Number of buildings: {len(buildings.get('buildings', []))}")
                for i, building in enumerate(buildings.get('buildings', [])[:3]):
                    print(f"   Building {i+1}: bbox={building.get('bbox')}, "
                          f"confidence={building.get('confidence', 0):.2f}")
                print()
            
            # Load metadata if available
            metadata = loader.load_metadata(image_id)
            if metadata:
                print(f"5. Geospatial Metadata:")
                print("-" * 40)
                print(f"   CRS: {metadata.get('crs', 'N/A')}")
                print(f"   Resolution: {metadata.get('resolution', 'N/A')}")
                print(f"   Acquisition date: {metadata.get('acquisition_date', 'N/A')}")
                print()
        else:
            print(f"   Image not found. Please add images to {loader.images_dir}")
            print()
    else:
        print("2. No images found in dataset.")
        print(f"   Please add images to: {loader.images_dir}")
        print()
    
    # Example: Preprocessing
    print("6. Image Preprocessing Example:")
    print("-" * 40)
    print("   Creating sample image for demonstration...")
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    processed = processor.preprocess_image(sample_image, 
                                          target_size=(512, 512), 
                                          normalize=True)
    print(f"   Original shape: {sample_image.shape}")
    print(f"   Processed shape: {processed.shape}")
    print(f"   Processed dtype: {processed.dtype}")
    print(f"   Processed value range: [{processed.min():.2f}, {processed.max():.2f}]")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("For more information, visit: https://rskworld.in")
    print("=" * 60)


if __name__ == "__main__":
    main()

