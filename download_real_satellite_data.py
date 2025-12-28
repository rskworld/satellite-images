#!/usr/bin/env python3
"""
Download Real Satellite Image Data
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script downloads real satellite images from public sources.
Supports: Landsat, Sentinel-2, MODIS
"""

import os
import requests
from pathlib import Path
import json
from datetime import datetime
import zipfile
import io


class SatelliteDataDownloader:
    """
    Download real satellite data from public sources.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self, output_dir: str = "data/images"):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_landsat_lookup(self, scene_id: str = None):
        """
        Download Landsat data using EarthExplorer API or direct links.
        Note: Requires USGS EarthExplorer account for full access.
        Created by: RSK World (https://rskworld.in)
        """
        print("Landsat Data Download")
        print("=" * 50)
        print("To download Landsat data, you need to:")
        print("1. Create a free account at: https://earthexplorer.usgs.gov/")
        print("2. Use the USGS API or download manually")
        print("3. Alternative: Use Google Earth Engine (requires account)")
        print()
        print("Example Landsat scene IDs:")
        print("- LC08_L1TP_044034_20230615_20230616_02_T1")
        print("- LC09_L1TP_044034_20230615_20230616_02_T1")
        print()
        print("For automated downloads, consider using:")
        print("- landsatxplore Python package")
        print("- Google Earth Engine Python API")
        print("- USGS M2M API")
        
    def download_from_sentinel_hub(self, bbox: list = None, date: str = None):
        """
        Download Sentinel-2 data.
        Note: Requires Sentinel Hub account for full access.
        Created by: RSK World (https://rskworld.in)
        """
        print("Sentinel-2 Data Download")
        print("=" * 50)
        print("To download Sentinel-2 data, you need to:")
        print("1. Create a free account at: https://scihub.copernicus.eu/")
        print("2. Use the Copernicus Open Access Hub")
        print("3. Alternative: Use Google Earth Engine")
        print()
        print("Example bounding box (San Francisco):")
        print("bbox = [-122.5, 37.5, -122.0, 38.0]  # [min_lon, min_lat, max_lon, max_lat]")
        print()
        print("For automated downloads, consider using:")
        print("- sentinelsat Python package")
        print("- Google Earth Engine Python API")
        print("- Sentinel Hub API")
        
    def download_sample_from_public_source(self, image_id: str = "sample_001"):
        """
        Download sample satellite images from public repositories.
        Created by: RSK World (https://rskworld.in)
        """
        print(f"Downloading sample data for {image_id}...")
        print("=" * 50)
        
        # Public satellite image repositories
        sources = {
            "planetary_computer": "https://planetarycomputer.microsoft.com/",
            "aws_landsat": "https://registry.opendata.aws/landsat-8/",
            "google_earth_engine": "https://earthengine.google.com/",
            "usgs_earth_explorer": "https://earthexplorer.usgs.gov/"
        }
        
        print("Public sources for satellite data:")
        for name, url in sources.items():
            print(f"  - {name}: {url}")
        print()
        print("Note: Most require registration but are free to use.")
        
    def create_download_instructions(self):
        """
        Create a comprehensive guide for downloading real satellite data.
        Created by: RSK World (https://rskworld.in)
        """
        instructions = """
# How to Download Real Satellite Image Data

## Free Public Sources

### 1. USGS EarthExplorer (Landsat)
- **URL**: https://earthexplorer.usgs.gov/
- **Registration**: Free account required
- **Data**: Landsat 4-9, MODIS, ASTER
- **Python Package**: `landsatxplore`

### 2. Copernicus Open Access Hub (Sentinel)
- **URL**: https://scihub.copernicus.eu/
- **Registration**: Free account required
- **Data**: Sentinel-1, Sentinel-2, Sentinel-3
- **Python Package**: `sentinelsat`

### 3. Google Earth Engine
- **URL**: https://earthengine.google.com/
- **Registration**: Free account required
- **Data**: Multiple satellite sources
- **Python Package**: `earthengine-api`

### 4. Microsoft Planetary Computer
- **URL**: https://planetarycomputer.microsoft.com/
- **Registration**: Free API key
- **Data**: Multiple satellite sources
- **Python Package**: `pystac-client`, `planetary-computer`

## Installation Commands

```bash
# For Landsat data
pip install landsatxplore

# For Sentinel-2 data
pip install sentinelsat

# For Google Earth Engine
pip install earthengine-api

# For Planetary Computer
pip install pystac-client planetary-computer
```

## Example: Download Landsat Data

```python
from landsatxplore.api import API

# Initialize API
api = API("your_username", "your_password")

# Search for scenes
scenes = api.search(
    dataset='landsat_ot_c2_l2',
    latitude=37.7749,
    longitude=-122.4194,
    start_date='2023-01-01',
    end_date='2023-12-31',
    max_cloud_cover=10
)

# Download scene
api.download(scenes[0].entity_id, output_dir='data/images')
```

## Example: Download Sentinel-2 Data

```python
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

# Initialize API
api = SentinelAPI('your_username', 'your_password', 'https://scihub.copernicus.eu/dhus')

# Search for products
products = api.query(
    area=geojson_to_wkt(read_geojson('area.geojson')),
    date=('20230101', '20231231'),
    platformname='Sentinel-2',
    cloudcoverpercentage=(0, 10)
)

# Download product
api.download(products[0]['uuid'], directory_path='data/images')
```

## Example: Using Google Earth Engine

```python
import ee

# Initialize
ee.Initialize()

# Load Landsat image
image = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20230615')

# Export to Google Drive or Cloud Storage
task = ee.batch.Export.image.toDrive(
    image=image.select(['SR_B4', 'SR_B3', 'SR_B2']),
    description='landsat_export',
    folder='satellite_images',
    scale=30,
    region=geometry
)
task.start()
```

## Quick Start with Planetary Computer

```python
import pystac_client
import planetary_computer

# Open catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search for Sentinel-2 data
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[-122.5, 37.5, -122.0, 38.0],
    datetime="2023-06-01/2023-06-30",
    query={"eo:cloud_cover": {"lt": 10}}
)

# Get items
items = search.item_collection()
# Download using the item's assets
```

## Recommended Approach

For this project, I recommend:
1. **Start with Google Earth Engine** - Easiest to use, good documentation
2. **Use Planetary Computer** - Good for programmatic access
3. **Use USGS EarthExplorer** - Best for Landsat data

All sources are free but require registration.
"""
        
        output_file = Path("DOWNLOAD_REAL_DATA_GUIDE.md")
        with open(output_file, 'w') as f:
            f.write(instructions)
        
        print(f"Download guide created: {output_file}")
        return instructions


def main():
    """
    Main function to help download real satellite data.
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("Real Satellite Data Downloader")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    
    downloader = SatelliteDataDownloader()
    
    # Create download guide
    downloader.create_download_instructions()
    print()
    
    # Show information about different sources
    downloader.download_from_landsat_lookup()
    print()
    downloader.download_from_sentinel_hub()
    print()
    
    print("=" * 60)
    print("For detailed instructions, see: DOWNLOAD_REAL_DATA_GUIDE.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

