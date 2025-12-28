#!/usr/bin/env python3
"""
Get Real Satellite Image Data
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script helps download real satellite images from public sources.
Supports multiple methods: Google Earth Engine, Planetary Computer, direct downloads.
"""

import os
import sys
from pathlib import Path
import json


def method_google_earth_engine():
    """
    Method 1: Download using Google Earth Engine (Recommended)
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("METHOD 1: Google Earth Engine")
    print("=" * 60)
    print()
    print("Steps:")
    print("1. Sign up for free: https://earthengine.google.com/")
    print("2. Install: pip install earthengine-api")
    print("3. Authenticate: earthengine authenticate")
    print("4. Run the script below")
    print()
    
    script = '''
import ee
from pathlib import Path
import requests

# Initialize
ee.Initialize()

# Define area (San Francisco Bay Area - matching your metadata)
geometry = ee.Geometry.Rectangle([-122.5, 37.5, -122.0, 38.0])

# Load Landsat 8 image
image = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20230615')

# Select RGB bands
rgb = image.select(['SR_B4', 'SR_B3', 'SR_B2']).multiply(0.0000275).add(-0.2).clamp(0, 1)

# Export to Google Drive
task = ee.batch.Export.image.toDrive(
    image=rgb.visualize(min=0, max=0.3, bands=['SR_B4', 'SR_B3', 'SR_B2']),
    description='satellite_sample_001',
    folder='satellite_images',
    scale=30,
    region=geometry,
    fileFormat='GeoTIFF'
)
task.start()
print("Task submitted. Check Google Drive for the image.")
'''
    
    print("Python Script:")
    print("-" * 60)
    print(script)
    print("-" * 60)
    print()


def method_planetary_computer():
    """
    Method 2: Download using Microsoft Planetary Computer
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("METHOD 2: Microsoft Planetary Computer")
    print("=" * 60)
    print()
    print("Steps:")
    print("1. Install: pip install pystac-client planetary-computer")
    print("2. No account needed for basic access")
    print("3. Run the script below")
    print()
    
    script = '''
import pystac_client
import planetary_computer
from pathlib import Path
import requests
from PIL import Image
import io

# Open catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search for Sentinel-2 data
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[-122.5, 37.5, -122.0, 38.0],  # San Francisco area
    datetime="2023-06-01/2023-06-30",
    query={"eo:cloud_cover": {"lt": 10}}
)

# Get first item
items = search.item_collection()
if items:
    item = items[0]
    
    # Get RGB asset
    asset = item.assets.get("visual")
    if asset:
        # Sign the URL
        href = planetary_computer.sign(asset.href)
        
        # Download image
        response = requests.get(href)
        img = Image.open(io.BytesIO(response.content))
        
        # Save
        output_dir = Path("data/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(output_dir / "sample_001.png")
        print(f"Downloaded: {output_dir / 'sample_001.png'}")
'''
    
    print("Python Script:")
    print("-" * 60)
    print(script)
    print("-" * 60)
    print()


def method_direct_download():
    """
    Method 3: Direct download from public repositories
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("METHOD 3: Direct Download (Sample Images)")
    print("=" * 60)
    print()
    print("Downloading sample real satellite images from public sources...")
    print()
    
    # Sample real satellite image URLs (public domain)
    sample_urls = {
        "sample_001": "https://raw.githubusercontent.com/developmentseed/geolambda/master/tests/fixtures/rgb.tif",
        # Add more public domain satellite image URLs here
    }
    
    print("Note: For real production data, use Method 1 or 2 above.")
    print("These methods provide access to full Landsat and Sentinel-2 archives.")
    print()


def create_download_script():
    """
    Create an automated download script.
    Created by: RSK World (https://rskworld.in)
    """
    script_content = '''#!/usr/bin/env python3
"""
Automated Real Satellite Data Download
Created by: RSK World (https://rskworld.in)
"""

import pystac_client
import planetary_computer
from pathlib import Path
import requests
from PIL import Image
import io
from datetime import datetime, timedelta

def download_satellite_images(num_images=10, output_dir="data/images"):
    """
    Download real satellite images from Planetary Computer.
    Created by: RSK World (https://rskworld.in)
    """
    print("Downloading Real Satellite Images")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Different locations for variety
    locations = [
        {"bbox": [-122.5, 37.5, -122.0, 38.0], "name": "San Francisco"},
        {"bbox": [-74.0, 40.6, -73.9, 40.7], "name": "New York"},
        {"bbox": [2.2, 48.8, 2.4, 48.9], "name": "Paris"},
        {"bbox": [139.6, 35.6, 139.8, 35.7], "name": "Tokyo"},
        {"bbox": [-0.2, 51.4, 0.0, 51.5], "name": "London"},
    ]
    
    downloaded = 0
    date_start = datetime(2023, 6, 1)
    
    for i in range(num_images):
        location = locations[i % len(locations)]
        date = date_start + timedelta(days=i*7)
        date_str = date.strftime("%Y-%m-%d")
        
        try:
            # Search for Sentinel-2 data
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=location["bbox"],
                datetime=f"{date_str}/{date_str}",
                query={"eo:cloud_cover": {"lt": 20}},
                limit=1
            )
            
            items = search.item_collection()
            if items:
                item = items[0]
                asset = item.assets.get("visual")
                
                if asset:
                    # Sign the URL
                    href = planetary_computer.sign(asset.href)
                    
                    # Download image
                    print(f"[{i+1}/{num_images}] Downloading from {location['name']}...")
                    response = requests.get(href, timeout=30)
                    
                    if response.status_code == 200:
                        img = Image.open(io.BytesIO(response.content))
                        
                        # Resize to 512x512 if needed
                        if img.size != (512, 512):
                            img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        
                        # Save as PNG
                        output_file = output_path / f"sample_{i+1:03d}.png"
                        img.save(output_file, 'PNG')
                        print(f"  [OK] Saved: {output_file}")
                        downloaded += 1
                    else:
                        print(f"  [ERROR] Download failed: {response.status_code}")
            else:
                print(f"  [SKIP] No images found for {location['name']} on {date_str}")
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
    
    print()
    print("=" * 60)
    print(f"Download complete! {downloaded}/{num_images} images downloaded.")
    print(f"Images saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    # Install required packages first:
    # pip install pystac-client planetary-computer requests pillow
    
    try:
        download_satellite_images(num_images=10)
    except ImportError as e:
        print("ERROR: Missing required package.")
        print("Please install: pip install pystac-client planetary-computer requests pillow")
        print(f"Details: {e}")
'''
    
    output_file = Path("download_real_images.py")
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print(f"Created automated download script: {output_file}")
    return output_file


def main():
    """
    Main function to help get real satellite data.
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("Get Real Satellite Image Data")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    
    # Show all methods
    method_google_earth_engine()
    print()
    method_planetary_computer()
    print()
    
    # Create automated script
    script_file = create_download_script()
    print()
    print("=" * 60)
    print("QUICK START:")
    print("=" * 60)
    print("1. Install packages:")
    print("   pip install pystac-client planetary-computer requests pillow")
    print()
    print("2. Run the automated script:")
    print(f"   python {script_file}")
    print()
    print("This will download 10 real satellite images from Microsoft Planetary Computer")
    print("(No account required for basic access)")
    print("=" * 60)


if __name__ == "__main__":
    main()

