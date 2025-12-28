# Data Directory

<!--
    Satellite Image Dataset Project
    Created by: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
-->

This directory contains the satellite image dataset files.

## Directory Structure

```
data/
├── images/              # Satellite image files (PNG, TIFF, GeoTIFF)
├── labels/              # Land cover classification labels (JSON)
├── metadata/            # Geospatial metadata files (JSON)
├── building_detection/  # Building detection annotations (JSON)
└── samples/            # Sample data for testing
```

## File Formats

### Images
- **PNG**: Standard PNG format for visualization
- **TIFF**: High-quality TIFF format
- **GeoTIFF**: TIFF with embedded geospatial metadata

### Labels (JSON)
Land cover classification labels in JSON format:
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

### Building Detection (JSON)
Building detection annotations:
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

### Metadata (JSON)
Geospatial metadata:
```json
{
    "image_id": "sample_001",
    "crs": "EPSG:4326",
    "bounds": {
        "min_lon": -122.5,
        "min_lat": 37.5,
        "max_lon": -122.0,
        "max_lat": 38.0
    },
    "resolution": 0.5,
    "acquisition_date": "2024-01-15"
}
```

## Usage

Place your satellite images and corresponding annotation files in the appropriate subdirectories. The dataset loader will automatically discover and load them.

For more information, see the main [README.md](../README.md).

---

**Created by: RSK World**  
Website: https://rskworld.in  
Email: help@rskworld.in  
Phone: +91 93305 39277

