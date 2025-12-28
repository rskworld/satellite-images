#!/usr/bin/env python3
"""
Download Real Sentinel-2 Data using sentinelsat
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script downloads real Sentinel-2 satellite images.
Requires: pip install sentinelsat
"""

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from pathlib import Path
import os
import json


def create_geojson_from_bounds(min_lon: float, min_lat: float, 
                               max_lon: float, max_lat: float,
                               output_file: str = "area.geojson"):
    """
    Create a GeoJSON file from bounding box coordinates.
    Created by: RSK World (https://rskworld.in)
    """
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            }
        }]
    }
    
    with open(output_file, 'w') as f:
        json.dump(geojson, f)
    
    return output_file


def download_sentinel2_data(username: str, password: str,
                            min_lon: float = -122.5,
                            min_lat: float = 37.5,
                            max_lon: float = -122.0,
                            max_lat: float = 38.0,
                            start_date: str = '20230601',
                            end_date: str = '20230630',
                            max_cloud_cover: int = 10,
                            output_dir: str = "data/images"):
    """
    Download real Sentinel-2 data.
    Created by: RSK World (https://rskworld.in)
    
    Args:
        username: Copernicus Hub username
        password: Copernicus Hub password
        min_lon, min_lat, max_lon, max_lat: Bounding box coordinates
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        max_cloud_cover: Maximum cloud cover percentage
        output_dir: Directory to save images
    """
    print("Downloading Sentinel-2 Data")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 50)
    
    # Initialize API
    try:
        api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
    except Exception as e:
        print(f"Error initializing API: {e}")
        print("Please check your credentials.")
        return
    
    # Create GeoJSON file for area
    geojson_file = create_geojson_from_bounds(min_lon, min_lat, max_lon, max_lat)
    area_wkt = geojson_to_wkt(read_geojson(geojson_file))
    
    print(f"Searching for Sentinel-2 products...")
    print(f"Area: ({min_lon}, {min_lat}) to ({max_lon}, {max_lat})")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Max cloud cover: {max_cloud_cover}%")
    
    try:
        # Search for products
        products = api.query(
            area=area_wkt,
            date=(start_date, end_date),
            platformname='Sentinel-2',
            processinglevel='Level-2A',
            cloudcoverpercentage=(0, max_cloud_cover)
        )
        
        print(f"Found {len(products)} products")
        
        if not products:
            print("No products found. Try adjusting date range or area.")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download products
        for i, (product_id, product_info) in enumerate(list(products.items())[:5], 1):
            print(f"\n[{i}/{min(5, len(products))}] Downloading: {product_info['title']}")
            print(f"  Date: {product_info['beginposition']}")
            print(f"  Cloud cover: {product_info['cloudcoverpercentage']}%")
            
            try:
                api.download(product_id, directory_path=str(output_path))
                print(f"  [OK] Downloaded successfully")
            except Exception as e:
                print(f"  [ERROR] Download failed: {e}")
        
        print("\n" + "=" * 50)
        print("Download complete!")
        print(f"Images saved to: {output_path}")
        
        # Clean up GeoJSON file
        if Path(geojson_file).exists():
            os.remove(geojson_file)
        
    except Exception as e:
        print(f"Error searching for products: {e}")


def main():
    """
    Main function - requires user credentials.
    Created by: RSK World (https://rskworld.in)
    """
    print("=" * 60)
    print("Sentinel-2 Data Downloader")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    print("This script requires:")
    print("1. Copernicus Hub account (free): https://scihub.copernicus.eu/")
    print("2. sentinelsat package: pip install sentinelsat")
    print()
    
    # Get credentials from environment or user input
    username = os.getenv('COPERNICUS_USERNAME')
    password = os.getenv('COPERNICUS_PASSWORD')
    
    if not username or not password:
        print("Please set COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables,")
        print("or modify this script to enter credentials directly.")
        print()
        print("Example:")
        print("  export COPERNICUS_USERNAME='your_username'")
        print("  export COPERNICUS_PASSWORD='your_password'")
        print("  python download_with_sentinelsat.py")
        return
    
    # Download data for San Francisco area (matching sample metadata)
    download_sentinel2_data(
        username=username,
        password=password,
        min_lon=-122.5,
        min_lat=37.5,
        max_lon=-122.0,
        max_lat=38.0,
        start_date='20230601',
        end_date='20230630',
        max_cloud_cover=10
    )


if __name__ == "__main__":
    main()

