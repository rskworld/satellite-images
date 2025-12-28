#!/usr/bin/env python3
"""
Satellite Image Dataset - Enhanced Real Image Downloader
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Enhanced real satellite image downloader with:
- Multiple data sources
- Automatic processing
- Quality validation
- Metadata extraction
- Batch downloading
"""

import os
import sys
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import io
import time

try:
    import pystac_client
    import planetary_computer
    PLANETARY_COMPUTER_AVAILABLE = True
except ImportError:
    PLANETARY_COMPUTER_AVAILABLE = False

try:
    from landsatxplore.api import API as LandsatAPI
    LANDSAT_AVAILABLE = True
except ImportError:
    LANDSAT_AVAILABLE = False

try:
    from sentinelsat import SentinelAPI
    SENTINEL_AVAILABLE = True
except ImportError:
    SENTINEL_AVAILABLE = False


class RealImageDownloader:
    """
    Enhanced real satellite image downloader.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self, output_dir: str = "data/images", 
                 metadata_dir: str = "data/metadata"):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded images
            metadata_dir: Directory to save metadata
        """
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.download_stats = {
            'total_attempted': 0,
            'total_successful': 0,
            'total_failed': 0,
            'sources_used': []
        }
    
    def download_from_planetary_computer(self, 
                                        locations: List[Dict],
                                        num_images: int = 10,
                                        max_cloud_cover: int = 20,
                                        date_start: Optional[datetime] = None,
                                        date_end: Optional[datetime] = None) -> List[str]:
        """
        Download images from Microsoft Planetary Computer.
        
        Args:
            locations: List of location dictionaries with 'bbox' and 'name'
            num_images: Number of images to download
            max_cloud_cover: Maximum cloud cover percentage
            date_start: Start date for search
            date_end: End date for search
            
        Returns:
            List of downloaded image paths
        """
        if not PLANETARY_COMPUTER_AVAILABLE:
            print("ERROR: pystac-client and planetary-computer packages required.")
            print("Install: pip install pystac-client planetary-computer")
            return []
        
        print("=" * 60)
        print("Downloading from Microsoft Planetary Computer")
        print("Created by: RSK World (https://rskworld.in)")
        print("=" * 60)
        
        downloaded_files = []
        
        try:
            # Open catalog
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            
            if date_start is None:
                date_start = datetime(2023, 6, 1)
            if date_end is None:
                date_end = datetime(2023, 12, 31)
            
            date_range = f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}"
            
            for i in range(num_images):
                location = locations[i % len(locations)]
                self.download_stats['total_attempted'] += 1
                
                try:
                    # Search for Sentinel-2 data
                    search = catalog.search(
                        collections=["sentinel-2-l2a"],
                        bbox=location["bbox"],
                        datetime=date_range,
                        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                        limit=10
                    )
                    
                    items = list(search.item_collection())
                    
                    if items:
                        item = items[i % len(items)] if len(items) > i else items[0]
                        
                        # Try to get visual asset
                        asset = item.assets.get("visual") or item.assets.get("rendered_preview")
                        
                        if asset:
                            href = planetary_computer.sign(asset.href)
                            
                            print(f"\n[{i+1}/{num_images}] Downloading from {location['name']}...")
                            print(f"  Date: {item.properties.get('datetime', 'N/A')}")
                            print(f"  Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")
                            
                            # Download with retry
                            success = self._download_image_with_retry(
                                href, 
                                self.output_dir / f"real_sample_{i+1:03d}.png",
                                max_retries=3
                            )
                            
                            if success:
                                # Save metadata
                                metadata = self._extract_metadata_from_item(item, location)
                                metadata_path = self.metadata_dir / f"real_sample_{i+1:03d}.json"
                                with open(metadata_path, 'w') as f:
                                    json.dump(metadata, f, indent=4)
                                
                                downloaded_files.append(str(self.output_dir / f"real_sample_{i+1:03d}.png"))
                                self.download_stats['total_successful'] += 1
                                print(f"  [OK] Saved: {self.output_dir / f'real_sample_{i+1:03d}.png'}")
                            else:
                                self.download_stats['total_failed'] += 1
                                print(f"  [ERROR] Download failed")
                        else:
                            print(f"  [SKIP] No visual asset available")
                            self.download_stats['total_failed'] += 1
                    else:
                        print(f"  [SKIP] No images found for {location['name']}")
                        self.download_stats['total_failed'] += 1
                        
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    self.download_stats['total_failed'] += 1
                    continue
                
                # Rate limiting
                time.sleep(1)
            
            if "planetary_computer" not in self.download_stats['sources_used']:
                self.download_stats['sources_used'].append("planetary_computer")
                
        except Exception as e:
            print(f"ERROR: Failed to connect to Planetary Computer: {e}")
        
        return downloaded_files
    
    def _download_image_with_retry(self, url: str, output_path: Path, 
                                   max_retries: int = 3) -> bool:
        """
        Download image with retry logic.
        
        Args:
            url: Image URL
            output_path: Output file path
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60, stream=True)
                
                if response.status_code == 200:
                    # Read and process image
                    img_data = response.content
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 512x512 if needed
                    if img.size != (512, 512):
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # Validate image quality
                    if self._validate_image_quality(img):
                        img.save(output_path, 'PNG')
                        return True
                    else:
                        print(f"    [WARN] Image quality validation failed")
                        return False
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"    [ERROR] Download error: {e}")
                return False
        
        return False
    
    def _validate_image_quality(self, image: Image.Image, 
                               min_brightness: float = 0.1,
                               max_brightness: float = 0.9) -> bool:
        """
        Validate image quality.
        
        Args:
            image: PIL Image object
            min_brightness: Minimum acceptable brightness
            max_brightness: Maximum acceptable brightness
            
        Returns:
            True if image passes validation
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check brightness
        mean_brightness = np.mean(img_array) / 255.0
        if mean_brightness < min_brightness or mean_brightness > max_brightness:
            return False
        
        # Check for all-black or all-white images
        if np.all(img_array == 0) or np.all(img_array == 255):
            return False
        
        # Check variance (too uniform = bad)
        variance = np.var(img_array)
        if variance < 100:  # Too uniform
            return False
        
        return True
    
    def _extract_metadata_from_item(self, item, location: Dict) -> Dict:
        """
        Extract metadata from STAC item.
        
        Args:
            item: STAC item object
            location: Location dictionary
            
        Returns:
            Metadata dictionary
        """
        props = item.properties
        
        metadata = {
            "image_id": item.id,
            "source": "Microsoft Planetary Computer",
            "collection": "sentinel-2-l2a",
            "crs": "EPSG:4326",
            "bounds": {
                "min_lon": location["bbox"][0],
                "min_lat": location["bbox"][1],
                "max_lon": location["bbox"][2],
                "max_lat": location["bbox"][3]
            },
            "location_name": location["name"],
            "acquisition_date": props.get("datetime", ""),
            "cloud_cover": props.get("eo:cloud_cover", 0),
            "resolution": 10.0,  # Sentinel-2 resolution
            "sensor": "Sentinel-2",
            "bands": ["red", "green", "blue", "nir", "red_edge_1", "red_edge_2", 
                     "red_edge_3", "swir1", "swir2"],
            "processing_level": "L2A",
            "provider": "ESA",
            "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": "RSK World",
            "website": "https://rskworld.in"
        }
        
        return metadata
    
    def download_sample_images(self, num_images: int = 10) -> List[str]:
        """
        Download sample real satellite images from multiple locations.
        
        Args:
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        # Define diverse locations
        locations = [
            {"bbox": [-122.5, 37.5, -122.0, 38.0], "name": "San Francisco, USA"},
            {"bbox": [-74.0, 40.6, -73.9, 40.7], "name": "New York, USA"},
            {"bbox": [2.2, 48.8, 2.4, 48.9], "name": "Paris, France"},
            {"bbox": [139.6, 35.6, 139.8, 35.7], "name": "Tokyo, Japan"},
            {"bbox": [-0.2, 51.4, 0.0, 51.5], "name": "London, UK"},
            {"bbox": [103.8, 1.3, 103.9, 1.4], "name": "Singapore"},
            {"bbox": [151.2, -33.8, 151.3, -33.7], "name": "Sydney, Australia"},
            {"bbox": [-46.6, -23.5, -46.5, -23.4], "name": "São Paulo, Brazil"},
            {"bbox": [77.2, 28.5, 77.3, 28.6], "name": "New Delhi, India"},
            {"bbox": [116.4, 39.9, 116.5, 40.0], "name": "Beijing, China"},
        ]
        
        return self.download_from_planetary_computer(
            locations=locations,
            num_images=num_images,
            max_cloud_cover=20
        )
    
    def get_download_statistics(self) -> Dict:
        """
        Get download statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.download_stats.copy()
        stats['success_rate'] = (
            stats['total_successful'] / stats['total_attempted'] * 100
            if stats['total_attempted'] > 0 else 0
        )
        return stats
    
    def print_statistics(self):
        """Print download statistics."""
        stats = self.get_download_statistics()
        print("\n" + "=" * 60)
        print("Download Statistics")
        print("=" * 60)
        print(f"Total attempted: {stats['total_attempted']}")
        print(f"Successful: {stats['total_successful']}")
        print(f"Failed: {stats['total_failed']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Sources used: {', '.join(stats['sources_used']) if stats['sources_used'] else 'None'}")
        print("=" * 60)


def main():
    """
    Main function for downloading real satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("Enhanced Real Satellite Image Downloader")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    
    # Check available sources
    print("Available data sources:")
    print(f"  - Planetary Computer: {'✓ Available' if PLANETARY_COMPUTER_AVAILABLE else '✗ Not installed'}")
    print(f"  - Landsat API: {'✓ Available' if LANDSAT_AVAILABLE else '✗ Not installed'}")
    print(f"  - Sentinel API: {'✓ Available' if SENTINEL_AVAILABLE else '✗ Not installed'}")
    print()
    
    if not PLANETARY_COMPUTER_AVAILABLE:
        print("ERROR: At least one data source package is required.")
        print("Install: pip install pystac-client planetary-computer requests pillow")
        return
    
    # Initialize downloader
    downloader = RealImageDownloader()
    
    # Download images
    print("Starting download...")
    downloaded_files = downloader.download_sample_images(num_images=10)
    
    # Print statistics
    downloader.print_statistics()
    
    if downloaded_files:
        print(f"\nSuccessfully downloaded {len(downloaded_files)} images!")
        print(f"Images saved to: {downloader.output_dir}")
        print(f"Metadata saved to: {downloader.metadata_dir}")
    else:
        print("\nNo images were downloaded. Check your internet connection and try again.")


if __name__ == "__main__":
    main()

