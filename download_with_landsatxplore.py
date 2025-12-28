#!/usr/bin/env python3
"""
Download Real Landsat Data using landsatxplore
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script downloads real Landsat satellite images.
Requires: pip install landsatxplore
"""

from landsatxplore.api import API
from pathlib import Path
import os


def download_landsat_data(username: str, password: str, 
                         latitude: float = 37.7749, 
                         longitude: float = -122.4194,
                         start_date: str = '2023-01-01',
                         end_date: str = '2023-12-31',
                         max_cloud_cover: int = 10,
                         output_dir: str = "data/images"):
    """
    Download real Landsat data.
    Created by: RSK World (https://rskworld.in)
    
    Args:
        username: USGS EarthExplorer username
        password: USGS EarthExplorer password
        latitude: Latitude of area of interest
        longitude: Longitude of area of interest
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        output_dir: Directory to save images
    """
    print("Downloading Landsat Data")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 50)
    
    # Initialize API
    try:
        api = API(username, password)
    except Exception as e:
        print(f"Error initializing API: {e}")
        print("Please check your credentials.")
        return
    
    # Search for scenes
    print(f"Searching for Landsat scenes near ({latitude}, {longitude})...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Max cloud cover: {max_cloud_cover}%")
    
    try:
        scenes = api.search(
            dataset='landsat_ot_c2_l2',  # Landsat 8-9 Collection 2 Level 2
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            max_results=10
        )
        
        print(f"Found {len(scenes)} scenes")
        
        if not scenes:
            print("No scenes found. Try adjusting date range or location.")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download scenes
        for i, scene in enumerate(scenes[:5], 1):  # Download first 5 scenes
            print(f"\n[{i}/{min(5, len(scenes))}] Downloading: {scene['display_id']}")
            print(f"  Date: {scene['acquisition_date']}")
            print(f"  Cloud cover: {scene['cloud_cover']}%")
            
            try:
                api.download(scene['entity_id'], output_dir=str(output_path))
                print(f"  [OK] Downloaded successfully")
            except Exception as e:
                print(f"  [ERROR] Download failed: {e}")
        
        print("\n" + "=" * 50)
        print("Download complete!")
        print(f"Images saved to: {output_path}")
        
    except Exception as e:
        print(f"Error searching for scenes: {e}")
    finally:
        api.logout()


def main():
    """
    Main function - requires user credentials.
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("Landsat Data Downloader")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    print("This script requires:")
    print("1. USGS EarthExplorer account (free): https://earthexplorer.usgs.gov/")
    print("2. landsatxplore package: pip install landsatxplore")
    print()
    
    # Get credentials from environment or user input
    username = os.getenv('USGS_USERNAME')
    password = os.getenv('USGS_PASSWORD')
    
    if not username or not password:
        print("Please set USGS_USERNAME and USGS_PASSWORD environment variables,")
        print("or modify this script to enter credentials directly.")
        print()
        print("Example:")
        print("  export USGS_USERNAME='your_username'")
        print("  export USGS_PASSWORD='your_password'")
        print("  python download_with_landsatxplore.py")
        return
    
    # Download data for San Francisco area (matching sample metadata)
    download_landsat_data(
        username=username,
        password=password,
        latitude=37.7749,  # San Francisco
        longitude=-122.4194,
        start_date='2023-06-01',
        end_date='2023-06-30',
        max_cloud_cover=10
    )


if __name__ == "__main__":
    main()

