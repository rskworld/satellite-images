#!/usr/bin/env python3
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
    date_end = datetime(2023, 12, 31)
    
    for i in range(num_images):
        location = locations[i % len(locations)]
        
        try:
            # Search for Sentinel-2 data with wider date range
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=location["bbox"],
                datetime=f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}",
                query={"eo:cloud_cover": {"lt": 30}},
                limit=10  # Get more results to choose from
            )
            
            items = list(search.item_collection())
            if items:
                # Try to get a unique item (skip if we already downloaded from this location)
                item = items[i % len(items)] if len(items) > i else items[0]
                
                # Try visual asset first, then fallback to other assets
                asset = item.assets.get("visual") or item.assets.get("rendered_preview")
                
                if asset:
                    # Sign the URL
                    href = planetary_computer.sign(asset.href)
                    
                    # Download image
                    print(f"[{i+1}/{num_images}] Downloading from {location['name']}...")
                    print(f"  Date: {item.properties.get('datetime', 'N/A')}")
                    print(f"  Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")
                    
                    try:
                        response = requests.get(href, timeout=60, stream=True)
                        
                        if response.status_code == 200:
                            # Read image data
                            img_data = response.content
                            img = Image.open(io.BytesIO(img_data))
                            
                            # Convert to RGB if needed
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Resize to 512x512 if needed
                            if img.size != (512, 512):
                                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                            
                            # Save as PNG
                            output_file = output_path / f"sample_{i+1:03d}.png"
                            img.save(output_file, 'PNG')
                            print(f"  [OK] Saved: {output_file}")
                            downloaded += 1
                        else:
                            print(f"  [ERROR] Download failed: HTTP {response.status_code}")
                    except Exception as e:
                        print(f"  [ERROR] Download error: {e}")
                else:
                    print(f"  [SKIP] No visual asset available for this item")
            else:
                print(f"  [SKIP] No images found for {location['name']}")
                
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
