#!/usr/bin/env python3
"""
Satellite Image Dataset - Advanced Features Example
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates all advanced features:
- Advanced image processing
- Machine learning integration
- Real image downloading
- Advanced visualization
- Batch processing
"""

import numpy as np
from pathlib import Path
import cv2
import json

# Import advanced modules
try:
    from advanced_processing import AdvancedImageProcessor
    from ml_integration import SatelliteMLProcessor
    from enhanced_real_image_downloader import EnhancedRealImageDownloader
    from advanced_visualization import AdvancedVisualizer
    from batch_processing import BatchProcessor, ImageAugmenter
    from data_loader import SatelliteDatasetLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the same directory.")
    exit(1)


def demonstrate_advanced_processing():
    """Demonstrate advanced image processing features."""
    print("\n" + "=" * 60)
    print("1. ADVANCED IMAGE PROCESSING")
    print("=" * 60)
    
    processor = AdvancedImageProcessor()
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print(f"Sample image shape: {sample_image.shape}")
    
    # Edge detection
    print("\n1.1 Edge Detection")
    edges = processor.detect_edges(sample_image, method='canny')
    print(f"   Edges detected: {edges.shape}")
    
    # Segmentation
    print("\n1.2 Image Segmentation")
    segmented, props = processor.segment_image(sample_image, method='slic', num_segments=10)
    print(f"   Segmented into {props['num_segments']} regions")
    
    # Feature extraction
    print("\n1.3 Feature Extraction")
    features = processor.extract_all_features(sample_image)
    print(f"   Extracted {len(features)} feature types:")
    for key in features.keys():
        print(f"     - {key}")
    
    # Image enhancement
    print("\n1.4 Image Enhancement")
    enhanced = processor.enhance_image(sample_image, method='clahe')
    print(f"   Enhanced image shape: {enhanced.shape}")
    
    # Noise reduction
    print("\n1.5 Noise Reduction")
    denoised = processor.reduce_noise(sample_image, method='bilateral')
    print(f"   Denoised image shape: {denoised.shape}")


def demonstrate_ml_integration():
    """Demonstrate ML integration features."""
    print("\n" + "=" * 60)
    print("2. MACHINE LEARNING INTEGRATION")
    print("=" * 60)
    
    ml_processor = SatelliteMLProcessor()
    
    # Create sample data
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sample_label = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    
    # Feature extraction for ML
    print("\n2.1 Feature Extraction for ML")
    features, positions = ml_processor.extract_features_for_ml(sample_image)
    print(f"   Extracted {len(features)} feature vectors")
    print(f"   Feature vector shape: {features.shape}")
    
    # Building detection
    print("\n2.2 Building Detection")
    buildings = ml_processor.detect_buildings_simple(sample_image)
    print(f"   Detected {len(buildings)} buildings")
    if buildings:
        print(f"   Example building: {buildings[0]}")
    
    # NDVI extraction
    print("\n2.3 NDVI Extraction")
    ndvi = ml_processor.extract_ndvi(sample_image)
    print(f"   NDVI map shape: {ndvi.shape}")
    print(f"   NDVI range: [{ndvi.min()}, {ndvi.max()}]")
    
    # Change detection
    print("\n2.4 Change Detection")
    image2 = sample_image.copy()
    image2[100:150, 100:150] = 255  # Simulate change
    change_map, stats = ml_processor.detect_changes(sample_image, image2)
    print(f"   Change percentage: {stats['change_percentage']:.2f}%")
    print(f"   Mean difference: {stats['mean_difference']:.4f}")


def demonstrate_real_image_download():
    """Demonstrate real image downloading."""
    print("\n" + "=" * 60)
    print("3. REAL IMAGE DOWNLOADING")
    print("=" * 60)
    
    downloader = EnhancedRealImageDownloader()
    
    print("\n3.1 Planetary Computer Download (No credentials needed)")
    print("   Note: This requires internet connection and packages:")
    print("   pip install pystac-client planetary-computer requests pillow")
    
    try:
        # Try to download (will fail if packages not installed, but shows how it works)
        files = downloader.download_sample_real_images(num_images=2)
        if files:
            print(f"   Successfully downloaded {len(files)} images!")
        else:
            print("   Download skipped (packages may not be installed)")
    except Exception as e:
        print(f"   Download test skipped: {e}")
    
    print("\n3.2 Other Sources Available:")
    print("   - USGS EarthExplorer (Landsat)")
    print("   - Copernicus Hub (Sentinel-2)")
    print("   See enhanced_real_image_downloader.py for details")


def demonstrate_advanced_visualization():
    """Demonstrate advanced visualization features."""
    print("\n" + "=" * 60)
    print("4. ADVANCED VISUALIZATION")
    print("=" * 60)
    
    visualizer = AdvancedVisualizer()
    
    # Create sample data
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Ensure visualizations directory exists
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    print("\n4.1 Statistical Visualization")
    visualizer.visualize_statistics(sample_image, 
                                    save_path=str(viz_dir / "stats_example.png"))
    print("   Saved statistics visualization")
    
    print("\n4.2 3D Surface Plot")
    visualizer.visualize_3d_surface(sample_image,
                                    save_path=str(viz_dir / "3d_example.png"))
    print("   Saved 3D visualization")
    
    print("\n4.3 Comparison View")
    image2 = cv2.flip(sample_image, 1)
    visualizer.create_comparison_view([sample_image, image2],
                                     titles=['Original', 'Flipped'],
                                     save_path=str(viz_dir / "comparison_example.png"))
    print("   Saved comparison view")
    
    print("\n4.4 Dashboard")
    features = {'mean': np.mean(sample_image), 'std': np.std(sample_image)}
    metadata = {'source': 'sample', 'size': '256x256'}
    visualizer.create_dashboard(sample_image, features, metadata,
                                save_path=str(viz_dir / "dashboard_example.png"))
    print("   Saved dashboard")


def demonstrate_batch_processing():
    """Demonstrate batch processing and augmentation."""
    print("\n" + "=" * 60)
    print("5. BATCH PROCESSING & AUGMENTATION")
    print("=" * 60)
    
    augmenter = ImageAugmenter()
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    print("\n5.1 Image Augmentation")
    augmented = augmenter.augment_image(sample_image, 
                                       ['flip_horizontal', 'brightness', 'rotate'])
    print(f"   Created {len(augmented)} augmented versions")
    
    print("\n5.2 Available Augmentations:")
    aug_methods = ['flip_horizontal', 'flip_vertical', 'rotate', 
                  'brightness', 'contrast', 'noise', 'crop', 'scale']
    for method in aug_methods:
        print(f"   - {method}")
    
    print("\n5.3 Batch Processing")
    print("   Use BatchProcessor for parallel processing of multiple images")
    print("   Supports ThreadPoolExecutor for efficient processing")


def demonstrate_integration():
    """Demonstrate integration of all features."""
    print("\n" + "=" * 60)
    print("6. INTEGRATED WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Load dataset
    loader = SatelliteDatasetLoader()
    info = loader.get_dataset_info()
    
    print(f"\n6.1 Dataset Information:")
    print(f"   Total images: {info['total_images']}")
    print(f"   Has labels: {info['has_labels']}")
    print(f"   Has buildings: {info['has_buildings']}")
    print(f"   Has metadata: {info['has_metadata']}")
    
    if info['total_images'] > 0:
        image_id = "sample_001"
        print(f"\n6.2 Processing {image_id}:")
        
        # Load image
        image, labels = loader.load_image_pair(image_id)
        if image is not None:
            print(f"   Loaded image: {image.shape}")
            
            # Advanced processing
            adv_processor = AdvancedImageProcessor()
            features = adv_processor.extract_all_features(image)
            print(f"   Extracted advanced features: {len(features)} types")
            
            # ML processing
            ml_processor = SatelliteMLProcessor()
            buildings = ml_processor.detect_buildings_simple(image)
            print(f"   Detected {len(buildings)} buildings")
            
            # Visualization
            if labels:
                visualizer = AdvancedVisualizer()
                viz_dir = Path("visualizations")
                viz_dir.mkdir(exist_ok=True)
                visualizer.visualize_statistics(image,
                                               save_path=str(viz_dir / f"{image_id}_stats.png"))
                print(f"   Created visualization")
        else:
            print(f"   Image not found, using sample data")
    else:
        print("\n   No images in dataset. Add images to data/images/ to see full workflow.")


def main():
    """Main function demonstrating all advanced features."""
    print("=" * 60)
    print("SATELLITE IMAGE DATASET - ADVANCED FEATURES DEMO")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    
    try:
        # 1. Advanced Processing
        demonstrate_advanced_processing()
        
        # 2. ML Integration
        demonstrate_ml_integration()
        
        # 3. Real Image Download
        demonstrate_real_image_download()
        
        # 4. Advanced Visualization
        demonstrate_advanced_visualization()
        
        # 5. Batch Processing
        demonstrate_batch_processing()
        
        # 6. Integration
        demonstrate_integration()
        
        print("\n" + "=" * 60)
        print("ADVANCED FEATURES DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("\nAll advanced features are now available:")
        print("  ✓ Advanced image processing (edge detection, segmentation, features)")
        print("  ✓ Machine learning integration (classification, detection)")
        print("  ✓ Real image downloading (multiple sources)")
        print("  ✓ Advanced visualization (3D, statistics, dashboards)")
        print("  ✓ Batch processing and augmentation")
        print("\nFor more information, visit: https://rskworld.in")
        print("=" * 60)
    
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

