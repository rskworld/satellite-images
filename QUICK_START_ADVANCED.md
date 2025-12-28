# Quick Start Guide - Advanced Features

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Optional: For real image downloading
pip install pystac-client planetary-computer requests
```

## Quick Examples

### 1. Advanced Image Processing

```python
from advanced_processing import AdvancedImageProcessor
import numpy as np

processor = AdvancedImageProcessor()
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# Edge detection
edges = processor.detect_edges(image, method='canny')

# Segmentation
segmented, props = processor.segment_image(image, method='slic', num_segments=10)

# Extract features
features = processor.extract_all_features(image)
```

### 2. Machine Learning

```python
from ml_integration import SatelliteMLProcessor

ml = SatelliteMLProcessor()

# Building detection
buildings = ml.detect_buildings_simple(image)
print(f"Detected {len(buildings)} buildings")

# NDVI
ndvi = ml.extract_ndvi(image)
```

### 3. Download Real Images

```python
from enhanced_real_image_downloader import EnhancedRealImageDownloader

downloader = EnhancedRealImageDownloader()
files = downloader.download_sample_real_images(num_images=5)
```

### 4. Advanced Visualization

```python
from advanced_visualization import AdvancedVisualizer

viz = AdvancedVisualizer()
viz.visualize_statistics(image, save_path='visualizations/stats.png')
viz.visualize_3d_surface(image, save_path='visualizations/3d.png')
```

### 5. Batch Processing

```python
from batch_processing import ImageAugmenter

augmenter = ImageAugmenter()
augmented = augmenter.augment_image(image, ['flip_horizontal', 'brightness'])
```

## Run Complete Demo

```bash
python advanced_example.py
```

This will demonstrate all features with sample data.

## Next Steps

1. Read `ADVANCED_FEATURES.md` for detailed documentation
2. Check `README.md` for full project documentation
3. Explore individual modules for specific features

---

*Created by RSK World - https://rskworld.in*

