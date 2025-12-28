#!/usr/bin/env python3
"""
Create Sample Satellite Images
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script creates sample satellite images matching the annotation data.
"""

from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import json


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_satellite_image(image_id: str, labels_path: str = None, 
                          size: tuple = (512, 512)) -> Image.Image:
    """
    Create a sample satellite image based on label data.
    Created by: RSK World (https://rskworld.in)
    
    Args:
        image_id: Image identifier
        labels_path: Path to labels JSON file (optional)
        size: Image size (width, height)
    """
    width, height = size
    
    # Create base image with sky/water gradient (blue tones)
    img = Image.new('RGB', size, color='#1a4d80')
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for y in range(height):
        r = int(26 + (y / height) * 30)
        g = int(77 + (y / height) * 40)
        b = int(128 + (y / height) * 50)
        color = (r, g, b)
        draw.line([(0, y), (width, y)], fill=color)
    
    # Color mapping for land cover classes
    class_colors = {
        'water': '#1e3a5f',
        'forest': '#2d5016',
        'urban': '#555555',
        'agriculture': '#8b7355',
        'barren': '#d4a574',
        'grassland': '#6b8e23'
    }
    
    # If labels file exists, use it to draw regions
    if labels_path and Path(labels_path).exists():
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        # Draw regions from labels
        for region in labels.get('regions', []):
            class_name = region.get('class', '')
            polygon = region.get('polygon', [])
            
            if polygon and class_name in class_colors:
                # Convert polygon to flat list for PIL
                flat_polygon = [coord for point in polygon for coord in point]
                color = hex_to_rgb(class_colors[class_name])
                
                # Draw filled polygon
                draw.polygon(flat_polygon, fill=color, outline=None)
                
                # Add some texture/variation
                for i in range(3):
                    offset_x = np.random.randint(-5, 5)
                    offset_y = np.random.randint(-5, 5)
                    textured_polygon = [[p[0] + offset_x, p[1] + offset_y] for p in polygon]
                    flat_textured = [coord for point in textured_polygon for coord in point]
                    darker_color = tuple(max(0, c - 20) for c in color)
                    draw.polygon(flat_textured, fill=darker_color, outline=None)
    else:
        # Create random regions if no labels
        classes = ['water', 'forest', 'urban', 'agriculture', 'barren', 'grassland']
        for _ in range(5):
            class_name = np.random.choice(classes)
            x = np.random.randint(50, width - 100)
            y = np.random.randint(50, height - 100)
            w = np.random.randint(80, 150)
            h = np.random.randint(80, 150)
            
            color = hex_to_rgb(class_colors.get(class_name, '#555555'))
            draw.rectangle([x, y, x + w, y + h], fill=color)
    
    # Add some noise/texture to make it look more realistic
    pixels = np.array(img)
    noise = np.random.randint(-10, 10, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    return img


def create_all_sample_images():
    """
    Create sample images for all sample data files.
    Created by: RSK World (https://rskworld.in)
    """
    images_dir = Path("data/images")
    labels_dir = Path("data/labels")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample satellite images...")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    # Find all label files
    label_files = list(labels_dir.glob("sample_*.json"))
    
    if not label_files:
        print("No label files found. Creating default sample images...")
        # Create at least 3 sample images
        for i in range(1, 4):
            image_id = f"sample_{i:03d}"
            img = create_satellite_image(image_id, size=(512, 512))
            output_path = images_dir / f"{image_id}.png"
            img.save(output_path, 'PNG')
            print(f"[OK] Created: {output_path}")
    else:
        # Create images based on label files
        for label_file in sorted(label_files):
            image_id = label_file.stem
            img = create_satellite_image(image_id, str(label_file), size=(512, 512))
            output_path = images_dir / f"{image_id}.png"
            img.save(output_path, 'PNG')
            print(f"[OK] Created: {output_path}")
    
    print()
    print(f"Successfully created {len(label_files) if label_files else 3} sample images!")
    print(f"Images saved to: {images_dir}")


if __name__ == "__main__":
    create_all_sample_images()

