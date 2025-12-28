# Satellite Image Dataset - Data Summary

<!--
    Satellite Image Dataset Project
    Created by: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
-->

## Generated Sample Data

This document summarizes the sample data files created for the Satellite Image Dataset project.

## Data Files Created

### Land Cover Labels (`data/labels/`)
Contains JSON files with land cover classification annotations:
- `sample_001.json` through `sample_010.json` (10 files)
- Each file includes:
  - Image ID
  - Land cover classes (water, forest, urban, agriculture, barren, grassland)
  - Polygon coordinates for each region
  - Area calculations
  - Confidence scores
  - Annotation metadata

### Building Detection (`data/building_detection/`)
Contains JSON files with building detection annotations:
- `sample_001.json` through `sample_010.json` (10 files)
- Each file includes:
  - Image ID
  - Building bounding boxes
  - Confidence scores
  - Building types (residential, commercial, industrial)
  - Height estimates
  - Area calculations

### Geospatial Metadata (`data/metadata/`)
Contains JSON files with geospatial metadata:
- `sample_001.json` through `sample_010.json` (10 files)
- Each file includes:
  - Image ID
  - Coordinate Reference System (CRS)
  - Geographic bounds (latitude/longitude)
  - Resolution
  - Acquisition date
  - Sensor information (Landsat 8/9, Sentinel-2, MODIS)
  - Spectral bands
  - Cloud cover percentage
  - Image dimensions
  - Provider information

## Statistics

- **Total Sample Images**: 10
- **Total Data Files**: 30 (10 labels + 10 buildings + 10 metadata)
- **Land Cover Classes**: 6 (water, forest, urban, agriculture, barren, grassland)
- **Building Types**: 3 (residential, commercial, industrial)
- **Sensors**: Landsat 8, Landsat 9, Sentinel-2, MODIS

## Data Format

All data files are in JSON format with the following structure:

### Label Format
```json
{
    "image_id": "sample_001",
    "classes": ["water", "forest", "urban"],
    "regions": [
        {
            "class": "urban",
            "polygon": [[x1, y1], [x2, y2], ...],
            "area": 10000.0,
            "confidence": 0.95
        }
    ]
}
```

### Building Detection Format
```json
{
    "image_id": "sample_001",
    "buildings": [
        {
            "id": "bld_001",
            "bbox": [x, y, width, height],
            "confidence": 0.95,
            "area": 2500.0,
            "type": "residential",
            "height_estimate": 15.5
        }
    ]
}
```

### Metadata Format
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
    "acquisition_date": "2024-01-15",
    "sensor": "Landsat 8"
}
```

## Usage

To use this data:

1. **Load labels:**
   ```python
   from data_loader import SatelliteDatasetLoader
   loader = SatelliteDatasetLoader()
   image, labels = loader.load_image_pair("sample_001")
   ```

2. **Load building detections:**
   ```python
   buildings = loader.load_building_detections("sample_001")
   ```

3. **Load metadata:**
   ```python
   metadata = loader.load_metadata("sample_001")
   ```

## Generating More Data

To generate additional sample data files, run:

```bash
python generate_sample_data.py
```

You can modify the `num_samples` parameter in the script to generate more or fewer samples.

## Notes

- All sample data is synthetic and generated for demonstration purposes
- Geographic coordinates are based on the San Francisco Bay Area region
- Building and land cover annotations are randomly generated but follow realistic patterns
- All files include RSK World attribution

---

**Created by: RSK World**  
Website: https://rskworld.in  
Email: help@rskworld.in  
Phone: +91 93305 39277

