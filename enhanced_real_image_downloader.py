#!/usr/bin/env python3
"""
Enhanced Real Satellite Image Downloader
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Supports multiple data sources:
- Microsoft Planetary Computer (Sentinel-2, Landsat)
- USGS EarthExplorer (Landsat)
- Copernicus Hub (Sentinel-2)
- Google Earth Engine (multiple sources)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import requests
from PIL import Image
import io
import numpy as np
import cv2


class EnhancedRealImageDownloader:
    """
    Enhanced downloader for real satellite images from multiple sources.
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
        self.downloaded_count = 0
    
    def download_from_planetary_computer(self, 
                                         locations: List[Dict],
                                         num_images: int = 10,
                                         date_start: Optional[datetime] = None,
                                         date_end: Optional[datetime] = None,
                                         max_cloud_cover: int = 30) -> List[str]:
        """
        Download images from Microsoft Planetary Computer.
        
        Args:
            locations: List of location dicts with 'bbox' and 'name'
            num_images: Number of images to download
            date_start: Start date for search
            date_end: End date for search
            max_cloud_cover: Maximum cloud cover percentage
            
        Returns:
            List of downloaded image paths
        """
        try:
            import pystac_client
            import planetary_computer
        except ImportError:
            print("ERROR: pystac-client and planetary-computer packages required.")
            print("Install with: pip install pystac-client planetary-computer")
            return []
        
        print("=" * 60)
        print("Downloading from Microsoft Planetary Computer")
        print("Created by: RSK World (https://rskworld.in)")
        print("=" * 60)
        
        if date_start is None:
            date_start = datetime(2023, 6, 1)
        if date_end is None:
            date_end = datetime(2023, 12, 31)
        
        downloaded_files = []
        
        try:
            # Open catalog
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            
            for i in range(num_images):
                location = locations[i % len(locations)]
                
                try:
                    # Search for Sentinel-2 data
                    search = catalog.search(
                        collections=["sentinel-2-l2a"],
                        bbox=location["bbox"],
                        datetime=f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}",
                        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                        limit=10
                    )
                    
                    items = list(search.item_collection())
                    if items:
                        item = items[i % len(items)] if len(items) > i else items[0]
                        
                        # Try visual asset first
                        asset = item.assets.get("visual") or item.assets.get("rendered_preview")
                        
                        if asset:
                            href = planetary_computer.sign(asset.href)
                            
                            print(f"\n[{i+1}/{num_images}] Downloading from {location['name']}...")
                            print(f"  Date: {item.properties.get('datetime', 'N/A')}")
                            print(f"  Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")
                            
                            try:
                                response = requests.get(href, timeout=60, stream=True)
                                
                                if response.status_code == 200:
                                    img_data = response.content
                                    img = Image.open(io.BytesIO(img_data))
                                    
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    
                                    # Resize to standard size
                                    if img.size != (512, 512):
                                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                                    
                                    # Save image
                                    output_file = self.output_dir / f"real_pc_{i+1:03d}.png"
                                    img.save(output_file, 'PNG')
                                    
                                    # Save metadata
                                    metadata = {
                                        'source': 'planetary_computer',
                                        'location': location['name'],
                                        'bbox': location['bbox'],
                                        'date': item.properties.get('datetime', ''),
                                        'cloud_cover': item.properties.get('eo:cloud_cover', 0),
                                        'download_date': datetime.now().isoformat()
                                    }
                                    
                                    metadata_file = self.output_dir / f"real_pc_{i+1:03d}_metadata.json"
                                    with open(metadata_file, 'w') as f:
                                        json.dump(metadata, f, indent=4)
                                    
                                    downloaded_files.append(str(output_file))
                                    self.downloaded_count += 1
                                    print(f"  [OK] Saved: {output_file}")
                                else:
                                    print(f"  [ERROR] HTTP {response.status_code}")
                            except Exception as e:
                                print(f"  [ERROR] {e}")
                        else:
                            print(f"  [SKIP] No visual asset available")
                    else:
                        print(f"  [SKIP] No images found for {location['name']}")
                
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    continue
            
            print(f"\nDownloaded {len(downloaded_files)} images from Planetary Computer")
            return downloaded_files
        
        except Exception as e:
            print(f"ERROR: {e}")
            return []
    
    def download_from_usgs(self, username: str, password: str,
                          latitude: float = 37.7749,
                          longitude: float = -122.4194,
                          start_date: str = '2023-01-01',
                          end_date: str = '2023-12-31',
                          max_cloud_cover: int = 10,
                          num_images: int = 5) -> List[str]:
        """
        Download images from USGS EarthExplorer (Landsat).
        
        Args:
            username: USGS username
            password: USGS password
            latitude: Latitude
            longitude: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Max cloud cover percentage
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        try:
            from landsatxplore.api import API
        except ImportError:
            print("ERROR: landsatxplore package required.")
            print("Install with: pip install landsatxplore")
            return []
        
        print("=" * 60)
        print("Downloading from USGS EarthExplorer (Landsat)")
        print("Created by: RSK World (https://rskworld.in)")
        print("=" * 60)
        
        downloaded_files = []
        
        try:
            api = API(username, password)
            
            print(f"Searching for Landsat scenes near ({latitude}, {longitude})...")
            
            scenes = api.search(
                dataset='landsat_ot_c2_l2',
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=max_cloud_cover,
                max_results=num_images
            )
            
            print(f"Found {len(scenes)} scenes")
            
            if not scenes:
                print("No scenes found.")
                return []
            
            for i, scene in enumerate(scenes[:num_images], 1):
                print(f"\n[{i}/{min(num_images, len(scenes))}] Downloading: {scene['display_id']}")
                print(f"  Date: {scene['acquisition_date']}")
                print(f"  Cloud cover: {scene['cloud_cover']}%")
                
                try:
                    # Download to temp directory first
                    temp_dir = self.output_dir / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    
                    api.download(scene['entity_id'], output_dir=str(temp_dir))
                    
                    # Process downloaded files (Landsat downloads are in tar.gz format)
                    # This is a simplified version - actual processing would extract and convert
                    print(f"  [OK] Downloaded (requires extraction)")
                    downloaded_files.append(f"temp/{scene['entity_id']}")
                
                except Exception as e:
                    print(f"  [ERROR] {e}")
            
            api.logout()
            print(f"\nDownloaded {len(downloaded_files)} scenes from USGS")
            return downloaded_files
        
        except Exception as e:
            print(f"ERROR: {e}")
            return []
    
    def download_from_copernicus(self, username: str, password: str,
                                 bbox: List[float],
                                 start_date: str = '20230601',
                                 end_date: str = '20230630',
                                 max_cloud_cover: int = 10,
                                 num_images: int = 5) -> List[str]:
        """
        Download images from Copernicus Hub (Sentinel-2).
        
        Args:
            username: Copernicus username
            password: Copernicus password
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            max_cloud_cover: Max cloud cover percentage
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        try:
            from sentinelsat import SentinelAPI, geojson_to_wkt
            import json
        except ImportError:
            print("ERROR: sentinelsat package required.")
            print("Install with: pip install sentinelsat")
            return []
        
        print("=" * 60)
        print("Downloading from Copernicus Hub (Sentinel-2)")
        print("Created by: RSK World (https://rskworld.in)")
        print("=" * 60)
        
        downloaded_files = []
        
        try:
            api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
            
            # Create area WKT from bbox
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]],
                            [bbox[0], bbox[1]]
                        ]]
                    }
                }]
            }
            area_wkt = geojson_to_wkt(geojson)
            
            print(f"Searching for Sentinel-2 products...")
            
            products = api.query(
                area=area_wkt,
                date=(start_date, end_date),
                platformname='Sentinel-2',
                processinglevel='Level-2A',
                cloudcoverpercentage=(0, max_cloud_cover)
            )
            
            print(f"Found {len(products)} products")
            
            if not products:
                print("No products found.")
                return []
            
            for i, (product_id, product_info) in enumerate(list(products.items())[:num_images], 1):
                print(f"\n[{i}/{min(num_images, len(products))}] Downloading: {product_info['title']}")
                print(f"  Date: {product_info['beginposition']}")
                print(f"  Cloud cover: {product_info['cloudcoverpercentage']}%")
                
                try:
                    api.download(product_id, directory_path=str(self.output_dir))
                    print(f"  [OK] Downloaded")
                    downloaded_files.append(product_id)
                
                except Exception as e:
                    print(f"  [ERROR] {e}")
            
            print(f"\nDownloaded {len(downloaded_files)} products from Copernicus")
            return downloaded_files
        
        except Exception as e:
            print(f"ERROR: {e}")
            return []
    
    def download_sample_real_images(self, num_images: int = 10) -> List[str]:
        """
        Download sample real images using Planetary Computer (no credentials needed).
        
        Args:
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        # Predefined locations for variety
        locations = [
            {"bbox": [-122.5, 37.5, -122.0, 38.0], "name": "San Francisco"},
            {"bbox": [-74.0, 40.6, -73.9, 40.7], "name": "New York"},
            {"bbox": [2.2, 48.8, 2.4, 48.9], "name": "Paris"},
            {"bbox": [139.6, 35.6, 139.8, 35.7], "name": "Tokyo"},
            {"bbox": [-0.2, 51.4, 0.0, 51.5], "name": "London"},
            {"bbox": [151.2, -33.8, 151.3, -33.7], "name": "Sydney"},
            {"bbox": [-46.6, -23.5, -46.5, -23.4], "name": "Sao Paulo"},
            {"bbox": [103.8, 1.3, 103.9, 1.4], "name": "Singapore"},
        ]
        
        return self.download_from_planetary_computer(
            locations=locations,
            num_images=num_images
        )
    
    def get_download_stats(self) -> Dict:
        """
        Get download statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_downloaded': self.downloaded_count,
            'output_directory': str(self.output_dir),
            'files_in_directory': len(list(self.output_dir.glob("*.png")))
        }


def main():
    """
    Main function for downloading real images.
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("Enhanced Real Satellite Image Downloader")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    
    downloader = EnhancedRealImageDownloader()
    
    # Option 1: Download from Planetary Computer (no credentials needed)
    print("Option 1: Download from Planetary Computer (Recommended - No credentials)")
    print("-" * 60)
    try:
        files = downloader.download_sample_real_images(num_images=10)
        print(f"\nSuccessfully downloaded {len(files)} images!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to install: pip install pystac-client planetary-computer requests pillow")
    
    # Option 2: Download from USGS (requires credentials)
    print("\n" + "=" * 60)
    print("Option 2: Download from USGS (Requires credentials)")
    print("-" * 60)
    username = os.getenv('USGS_USERNAME')
    password = os.getenv('USGS_PASSWORD')
    
    if username and password:
        try:
            files = downloader.download_from_usgs(
                username=username,
                password=password,
                num_images=3
            )
            print(f"\nSuccessfully downloaded {len(files)} images from USGS!")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("USGS_USERNAME and USGS_PASSWORD environment variables not set.")
        print("Skipping USGS download.")
    
    # Print statistics
    print("\n" + "=" * 60)
    stats = downloader.get_download_stats()
    print("Download Statistics:")
    print(f"  Total downloaded: {stats['total_downloaded']}")
    print(f"  Files in directory: {stats['files_in_directory']}")
    print(f"  Output directory: {stats['output_directory']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

