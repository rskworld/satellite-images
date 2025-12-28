# Satellite Image Dataset

<!--
    Satellite Image Dataset Project
    Created by: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Description: High-resolution satellite imagery dataset with land cover classification
-->

## Overview

This dataset includes high-resolution satellite images with land cover classifications, building detection, and environmental monitoring labels. Perfect for remote sensing, urban planning, agriculture monitoring, and geospatial analysis.

## Project Details

- **ID**: 21
- **Title**: Satellite Image Dataset
- **Category**: Image Data
- **Difficulty**: Advanced
- **Technologies**: PNG, TIFF, GeoTIFF, NumPy, OpenCV

## Features

### Core Features
- ✅ High-resolution images
- ✅ Land cover labels
- ✅ Building detection
- ✅ Multiple regions
- ✅ Geospatial metadata

### Advanced Features (NEW!)
- ✅ **Advanced Image Processing**: Edge detection, segmentation, feature extraction (HOG, LBP, GLCM)
- ✅ **Machine Learning Integration**: Classification, object detection, change detection, NDVI extraction
- ✅ **Real Image Downloading**: Support for multiple sources (Planetary Computer, USGS, Copernicus)
- ✅ **Advanced Visualization**: 3D plots, statistical analysis, interactive dashboards
- ✅ **Batch Processing**: Parallel processing, data augmentation, export utilities
- ✅ **Image Enhancement**: CLAHE, histogram equalization, noise reduction

## Dataset Structure

```
satellite-images/
├── images/              # Satellite image files (PNG, TIFF, GeoTIFF)
├── labels/              # Land cover classification labels
├── metadata/            # Geospatial metadata files
├── building_detection/  # Building detection annotations
└── samples/            # Sample data for testing
```

## Installation

### Requirements

- Python 3.7+
- NumPy
- OpenCV
- Rasterio (for GeoTIFF support)
- Matplotlib (for visualization)

### Setup

```bash
# Clone or download the dataset
# Install required packages
pip install -r requirements.txt
```

## Quick Start

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# For real image downloading (optional)
pip install pystac-client planetary-computer requests
```

### Run Advanced Features Demo

```bash
# Run comprehensive demo of all advanced features
python advanced_example.py
```

## Usage Examples

### Basic Usage

### Loading Satellite Images

```python
import cv2
import numpy as np
from pathlib import Path

# Load a PNG satellite image
image_path = Path('data/images/sample_001.png')
image = cv2.imread(str(image_path))
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
```

### Working with GeoTIFF Files

```python
import rasterio
from rasterio.plot import show

# Load GeoTIFF with geospatial metadata
with rasterio.open('data/images/sample_001.tif') as src:
    image = src.read()
    transform = src.transform
    crs = src.crs
    print(f"CRS: {crs}")
    print(f"Transform: {transform}")
    show(src)
```

### Processing Land Cover Labels

```python
import numpy as np
import json

# Load land cover classification labels
with open('data/labels/sample_001.json', 'r') as f:
    labels = json.load(f)
    
print(f"Land cover classes: {labels['classes']}")
print(f"Regions: {len(labels['regions'])}")
```

### Building Detection

```python
import json

# Load building detection annotations
with open('data/building_detection/sample_001.json', 'r') as f:
    buildings = json.load(f)
    
for building in buildings['buildings']:
    bbox = building['bbox']
    confidence = building['confidence']
    print(f"Building at {bbox} with confidence {confidence}")
```

## Data Format

### Image Files
- **PNG**: Standard PNG format for visualization
- **TIFF**: High-quality TIFF format
- **GeoTIFF**: TIFF with embedded geospatial metadata

### Label Format (JSON)
```json
{
    "image_id": "sample_001",
    "classes": ["water", "forest", "urban", "agriculture", "barren"],
    "regions": [
        {
            "class": "urban",
            "polygon": [[x1, y1], [x2, y2], ...],
            "area": 12345.67
        }
    ]
}
```

### Building Detection Format (JSON)
```json
{
    "image_id": "sample_001",
    "buildings": [
        {
            "bbox": [x, y, width, height],
            "confidence": 0.95,
            "area": 1234.56
        }
    ]
}
```

## Advanced Features

### 1. Advanced Image Processing

```python
from advanced_processing import AdvancedImageProcessor

processor = AdvancedImageProcessor()

# Edge detection
edges = processor.detect_edges(image, method='canny')

# Image segmentation
segmented, props = processor.segment_image(image, method='slic', num_segments=10)

# Feature extraction
features = processor.extract_all_features(image)  # HOG, LBP, GLCM features

# Image enhancement
enhanced = processor.enhance_image(image, method='clahe')

# Noise reduction
denoised = processor.reduce_noise(image, method='bilateral')
```

### 2. Machine Learning Integration

```python
from ml_integration import SatelliteMLProcessor

ml_processor = SatelliteMLProcessor()

# Extract features for ML
features, positions = ml_processor.extract_features_for_ml(image)

# Building detection
buildings = ml_processor.detect_buildings_simple(image)

# NDVI extraction
ndvi = ml_processor.extract_ndvi(image)

# Change detection
change_map, stats = ml_processor.detect_changes(image1, image2)
```

### 3. Real Image Downloading

```python
from enhanced_real_image_downloader import EnhancedRealImageDownloader

downloader = EnhancedRealImageDownloader()

# Download from Planetary Computer (no credentials needed)
files = downloader.download_sample_real_images(num_images=10)

# Download from USGS (requires credentials)
files = downloader.download_from_usgs(username, password, num_images=5)

# Download from Copernicus (requires credentials)
files = downloader.download_from_copernicus(username, password, bbox, num_images=5)
```

### 4. Advanced Visualization

```python
from advanced_visualization import AdvancedVisualizer

visualizer = AdvancedVisualizer()

# Statistical analysis
visualizer.visualize_statistics(image, save_path='stats.png')

# 3D surface plot
visualizer.visualize_3d_surface(image, save_path='3d.png')

# Comparison view
visualizer.create_comparison_view([img1, img2], titles=['Before', 'After'])

# Comprehensive dashboard
visualizer.create_dashboard(image, features, metadata, save_path='dashboard.png')
```

### 5. Batch Processing & Augmentation

```python
from batch_processing import BatchProcessor, ImageAugmenter

# Image augmentation
augmenter = ImageAugmenter()
augmented = augmenter.augment_image(image, ['flip_horizontal', 'brightness', 'rotate'])

# Batch processing
processor = BatchProcessor(num_workers=4)
results = processor.process_batch(image_paths, processor_func, output_dir)
```

## Applications

1. **Remote Sensing**: Analyze land use and land cover changes
2. **Urban Planning**: Monitor urban growth and development
3. **Agriculture Monitoring**: Track crop health and yield estimation
4. **Environmental Monitoring**: Detect deforestation, water bodies, etc.
5. **Disaster Management**: Assess damage and plan recovery
6. **Machine Learning Research**: Train models for classification and detection
7. **Geospatial Analysis**: Advanced feature extraction and analysis

## Citation

If you use this dataset in your research or projects, please cite:

```
Satellite Image Dataset. RSK World. https://rskworld.in
```

## License

Please refer to the license file included with the dataset.

## Contact

**RSK World**
- Website: [https://rskworld.in](https://rskworld.in)
- Email: help@rskworld.in
- Phone: +91 93305 39277

## Acknowledgments

This dataset is created and maintained by RSK World. For more free programming resources and source code, visit [rskworld.in](https://rskworld.in).

---

*Last updated: 2024*

