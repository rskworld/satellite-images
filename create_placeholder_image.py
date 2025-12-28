#!/usr/bin/env python3
"""
Create Placeholder Image for Satellite Image Dataset
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script creates a placeholder image for the project.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_placeholder_image(output_path: str = "satellite-images.png", 
                            size: tuple = (800, 600)):
    """
    Create a placeholder image for the satellite dataset.
    Created by: RSK World (https://rskworld.in)
    
    Args:
        output_path: Path to save the image
        size: Image size (width, height)
    """
    # Create image with gradient background (simulating satellite imagery)
    img = Image.new('RGB', size, color='#1a4d80')
    draw = ImageDraw.Draw(img)
    
    # Create gradient effect
    width, height = size
    for y in range(height):
        # Create a gradient from dark blue to lighter blue
        r = int(26 + (y / height) * 20)
        g = int(77 + (y / height) * 30)
        b = int(128 + (y / height) * 40)
        color = (r, g, b)
        draw.line([(0, y), (width, y)], fill=color)
    
    # Add some "land" patches (green/brown rectangles)
    land_colors = ['#2d5016', '#4a7c2a', '#6b8e23', '#8b7355']
    for i in range(5):
        x = np.random.randint(0, width - 100)
        y = np.random.randint(height // 3, height - 50)
        w = np.random.randint(50, 150)
        h = np.random.randint(30, 80)
        color = np.random.choice(land_colors)
        draw.rectangle([x, y, x + w, y + h], fill=color)
    
    # Add some "urban" patches (gray rectangles)
    for i in range(3):
        x = np.random.randint(0, width - 80)
        y = np.random.randint(height // 2, height - 40)
        w = np.random.randint(40, 100)
        h = np.random.randint(20, 60)
        draw.rectangle([x, y, x + w, y + h], fill='#555555')
    
    # Add title text
    try:
        # Try to use a larger font
        font_size = 40
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text = "Satellite Image Dataset"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw text with shadow
    text_x = (width - text_width) // 2
    text_y = height // 4
    
    # Shadow
    draw.text((text_x + 2, text_y + 2), text, font=font, fill='#000000')
    # Main text
    draw.text((text_x, text_y), text, font=font, fill='#ffffff')
    
    # Add subtitle
    try:
        subtitle_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except:
        subtitle_font = ImageFont.load_default()
    
    subtitle = "RSK World - rskworld.in"
    bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = bbox[2] - bbox[0]
    subtitle_x = (width - subtitle_width) // 2
    subtitle_y = text_y + text_height + 20
    
    draw.text((subtitle_x + 1, subtitle_y + 1), subtitle, 
             font=subtitle_font, fill='#000000')
    draw.text((subtitle_x, subtitle_y), subtitle, 
             font=subtitle_font, fill='#0dcaf0')
    
    # Save image
    img.save(output_path, 'PNG')
    print(f"Placeholder image created: {output_path}")
    print("Created by: RSK World (https://rskworld.in)")


if __name__ == "__main__":
    create_placeholder_image()

