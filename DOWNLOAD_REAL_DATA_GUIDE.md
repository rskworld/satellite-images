
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
