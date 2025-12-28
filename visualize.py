#!/usr/bin/env python3
"""
Satellite Image Dataset - Visualization Script
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script provides visualization utilities for satellite images and labels.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import Optional
from data_loader import SatelliteDatasetLoader
from process_images import SatelliteImageProcessor


def visualize_land_cover(image: np.ndarray, labels: dict, save_path: Optional[str] = None):
    """
    Visualize satellite image with land cover classifications.
    Created by: RSK World (https://rskworld.in)
    
    Args:
        image: Satellite image array
        labels: Label data with regions
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Satellite Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Image with land cover overlay
    axes[1].imshow(image)
    
    if 'regions' in labels:
        # Color map for different land cover classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels.get('classes', []))))
        class_colors = {cls: colors[i] for i, cls in enumerate(labels.get('classes', []))}
        
        for region in labels['regions']:
            if 'polygon' in region:
                polygon = np.array(region['polygon'])
                class_name = region.get('class', 'unknown')
                color = class_colors.get(class_name, 'red')
                
                axes[1].fill(polygon[:, 0], polygon[:, 1], 
                           color=color, alpha=0.3, label=class_name)
                axes[1].plot(polygon[:, 0], polygon[:, 1], 
                           color=color, linewidth=2)
    
    axes[1].set_title('Land Cover Classification', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_buildings(image: np.ndarray, buildings: dict, save_path: Optional[str] = None):
    """
    Visualize satellite image with building detections.
    Created by: RSK World (https://rskworld.in)
    
    Args:
        image: Satellite image array
        buildings: Building detection data
        save_path: Optional path to save the visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    
    if 'buildings' in buildings:
        for building in buildings['buildings']:
            if 'bbox' in building:
                bbox = building['bbox']
                x, y, w, h = bbox
                confidence = building.get('confidence', 1.0)
                
                # Draw bounding box
                rect = plt.Rectangle((x, y), w, h, 
                                    linewidth=2, edgecolor='red', 
                                    facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
                # Add confidence label
                ax.text(x, y - 5, f'{confidence:.2f}', 
                       color='red', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('Building Detection', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_visualization(loader: SatelliteDatasetLoader, 
                                 output_dir: str = "visualizations"):
    """
    Create summary visualizations for the dataset.
    Created by: RSK World (https://rskworld.in)
    
    Args:
        loader: Dataset loader instance
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    info = loader.get_dataset_info()
    
    if info['total_images'] == 0:
        print("No images found in dataset.")
        return
    
    # Create visualizations for first few images
    for i, image_file in enumerate(info['images'][:5]):
        image_id = Path(image_file).stem
        image, labels = loader.load_image_pair(image_id)
        
        if image is None:
            continue
        
        # Visualize with labels if available
        if labels:
            save_path = output_path / f"{image_id}_landcover.png"
            visualize_land_cover(image, labels, str(save_path))
        
        # Visualize buildings if available
        buildings = loader.load_building_detections(image_id)
        if buildings:
            save_path = output_path / f"{image_id}_buildings.png"
            visualize_buildings(image, buildings, str(save_path))
        
        print(f"Processed {i+1}/{min(5, len(info['images']))} images")


def main():
    """
    Main function for visualization script.
    Created by: RSK World (https://rskworld.in)
    """
    print("Satellite Image Dataset - Visualization Tool")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    loader = SatelliteDatasetLoader()
    processor = SatelliteImageProcessor()
    
    # Create summary visualizations
    create_summary_visualization(loader)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

