#!/usr/bin/env python3
"""
Satellite Image Dataset - Configuration
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Configuration settings for the satellite image dataset.
"""

# Dataset paths
DATA_DIR = "data"
IMAGES_DIR = "data/images"
LABELS_DIR = "data/labels"
METADATA_DIR = "data/metadata"
BUILDING_DETECTION_DIR = "data/building_detection"
SAMPLES_DIR = "data/samples"
OUTPUT_DIR = "visualizations"

# Image processing settings
DEFAULT_IMAGE_SIZE = (512, 512)
NORMALIZE_IMAGES = True
SUPPORTED_FORMATS = ['.png', '.tiff', '.tif', '.jpg', '.jpeg', '.geotiff']

# Land cover classes
LAND_COVER_CLASSES = [
    "water",
    "forest",
    "urban",
    "agriculture",
    "barren",
    "grassland",
    "wetland",
    "snow"
]

# Visualization settings
VISUALIZATION_DPI = 150
FIGURE_SIZE = (12, 12)

# Project metadata
PROJECT_NAME = "Satellite Image Dataset"
PROJECT_VERSION = "1.0.0"
AUTHOR = "RSK World"
AUTHOR_WEBSITE = "https://rskworld.in"
AUTHOR_EMAIL = "help@rskworld.in"
AUTHOR_PHONE = "+91 93305 39277"

