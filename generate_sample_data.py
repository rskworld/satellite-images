#!/usr/bin/env python3
"""
Generate Sample Data for Satellite Image Dataset
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script generates sample data files for testing and demonstration.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random


def generate_label_data(image_id: str, num_regions: int = 4) -> dict:
    """
    Generate sample land cover label data.
    Created by: RSK World (https://rskworld.in)
    """
    classes = ["water", "forest", "urban", "agriculture", "barren", "grassland"]
    selected_classes = random.sample(classes, min(num_regions, len(classes)))
    
    regions = []
    used_areas = []
    
    for i, class_name in enumerate(selected_classes):
        # Generate random polygon coordinates
        x = random.randint(50, 400)
        y = random.randint(50, 400)
        width = random.randint(50, 150)
        height = random.randint(50, 150)
        
        # Avoid overlapping regions
        while any(abs(x - ux) < 100 and abs(y - uy) < 100 for ux, uy in used_areas):
            x = random.randint(50, 400)
            y = random.randint(50, 400)
        
        used_areas.append((x, y))
        
        polygon = [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ]
        
        area = width * height
        confidence = round(random.uniform(0.85, 0.98), 2)
        
        regions.append({
            "class": class_name,
            "polygon": polygon,
            "area": float(area),
            "confidence": confidence
        })
    
    return {
        "image_id": image_id,
        "classes": selected_classes,
        "regions": regions,
        "annotation_date": datetime.now().strftime("%Y-%m-%d"),
        "annotator": "RSK World"
    }


def generate_building_data(image_id: str, num_buildings: int = 5) -> dict:
    """
    Generate sample building detection data.
    Created by: RSK World (https://rskworld.in)
    """
    building_types = ["residential", "commercial", "industrial"]
    buildings = []
    
    used_positions = []
    
    for i in range(num_buildings):
        # Generate random building position
        x = random.randint(100, 500)
        y = random.randint(100, 400)
        width = random.randint(30, 60)
        height = random.randint(30, 55)
        
        # Avoid overlapping buildings
        while any(abs(x - ux) < 80 and abs(y - uy) < 80 for ux, uy in used_positions):
            x = random.randint(100, 500)
            y = random.randint(100, 400)
        
        used_positions.append((x, y))
        
        bbox = [x, y, width, height]
        area = width * height
        confidence = round(random.uniform(0.85, 0.96), 2)
        building_type = random.choice(building_types)
        height_estimate = round(random.uniform(10.0, 30.0), 1)
        
        buildings.append({
            "id": f"bld_{i+1:03d}",
            "bbox": bbox,
            "confidence": confidence,
            "area": float(area),
            "type": building_type,
            "height_estimate": height_estimate
        })
    
    return {
        "image_id": image_id,
        "buildings": buildings,
        "detection_date": datetime.now().strftime("%Y-%m-%d"),
        "model_version": "v1.0",
        "total_buildings": len(buildings)
    }


def generate_metadata(image_id: str) -> dict:
    """
    Generate sample geospatial metadata.
    Created by: RSK World (https://rskworld.in)
    """
    sensors = ["Landsat 8", "Landsat 9", "Sentinel-2", "MODIS"]
    providers = ["USGS", "ESA", "NASA"]
    
    # Random location (San Francisco Bay Area)
    base_lon = -122.4
    base_lat = 37.6
    
    min_lon = round(base_lon + random.uniform(-0.3, 0.3), 2)
    min_lat = round(base_lat + random.uniform(-0.2, 0.2), 2)
    max_lon = round(min_lon + random.uniform(0.3, 0.6), 2)
    max_lat = round(min_lat + random.uniform(0.3, 0.6), 2)
    
    sensor = random.choice(sensors)
    provider = random.choice(providers)
    
    if "Sentinel" in sensor:
        bands = ["red", "green", "blue", "nir", "red_edge_1", "red_edge_2", "red_edge_3", "swir1", "swir2"]
        processing_level = "L2A"
    else:
        bands = ["red", "green", "blue", "nir", "swir1", "swir2"]
        processing_level = "L2"
    
    acquisition_date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
    
    return {
        "image_id": image_id,
        "crs": "EPSG:4326",
        "bounds": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat
        },
        "resolution": round(random.uniform(0.3, 0.6), 1),
        "acquisition_date": acquisition_date,
        "sensor": sensor,
        "bands": bands,
        "cloud_cover": round(random.uniform(0.5, 5.0), 1),
        "image_size": {
            "width": 512,
            "height": 512
        },
        "provider": provider,
        "processing_level": processing_level,
        "created_by": "RSK World",
        "website": "https://rskworld.in"
    }


def generate_all_samples(num_samples: int = 5):
    """
    Generate all sample data files.
    Created by: RSK World (https://rskworld.in)
    """
    base_dir = Path("data")
    labels_dir = base_dir / "labels"
    buildings_dir = base_dir / "building_detection"
    metadata_dir = base_dir / "metadata"
    
    # Ensure directories exist
    labels_dir.mkdir(parents=True, exist_ok=True)
    buildings_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating sample data files...")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    for i in range(1, num_samples + 1):
        image_id = f"sample_{i:03d}"
        
        # Generate label data
        label_data = generate_label_data(image_id, num_regions=random.randint(3, 6))
        label_path = labels_dir / f"{image_id}.json"
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=4)
        print(f"[OK] Generated: {label_path}")
        
        # Generate building detection data
        building_data = generate_building_data(image_id, num_buildings=random.randint(4, 8))
        building_path = buildings_dir / f"{image_id}.json"
        with open(building_path, 'w') as f:
            json.dump(building_data, f, indent=4)
        print(f"[OK] Generated: {building_path}")
        
        # Generate metadata
        metadata = generate_metadata(image_id)
        metadata_path = metadata_dir / f"{image_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[OK] Generated: {metadata_path}")
        print()
    
    print(f"Successfully generated {num_samples} sample datasets!")
    print(f"Total files: {num_samples * 3}")


if __name__ == "__main__":
    # Generate 10 additional samples (we already have 3)
    generate_all_samples(num_samples=10)

