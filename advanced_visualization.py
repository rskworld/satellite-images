#!/usr/bin/env python3
"""
Satellite Image Dataset - Advanced Visualization
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Advanced visualization features including:
- Interactive plots
- 3D visualizations
- Statistical analysis plots
- Comparison views
- Time series visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection


class AdvancedVisualizer:
    """
    Advanced visualization tools for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def create_comparison_view(self, images: List[np.ndarray],
                              titles: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Create side-by-side comparison of multiple images.
        
        Args:
            images: List of images to compare
            titles: Optional list of titles
            save_path: Optional path to save figure
        """
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        for i, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img)
            ax.axis('off')
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison view to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_statistics(self, image: np.ndarray,
                           save_path: Optional[str] = None) -> None:
        """
        Create comprehensive statistical visualization.
        
        Args:
            image: Input image
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Histogram
        ax2 = plt.subplot(2, 3, 2)
        if len(image.shape) == 3:
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = np.histogram(image[:, :, i], bins=256, range=(0, 256))[0]
                ax2.plot(hist, color=color, alpha=0.7, label=color.capitalize())
            ax2.legend()
        else:
            hist = np.histogram(image, bins=256, range=(0, 256))[0]
            ax2.plot(hist, color='black')
        ax2.set_title('Histogram', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Pixel Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Statistics text
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')
        stats_text = f"""
        Image Statistics:
        
        Shape: {image.shape}
        Dtype: {image.dtype}
        Min: {np.min(image)}
        Max: {np.max(image)}
        Mean: {np.mean(image):.2f}
        Std: {np.std(image):.2f}
        Median: {np.median(image):.2f}
        """
        if len(image.shape) == 3:
            for i, color in enumerate(['Red', 'Green', 'Blue']):
                stats_text += f"\n{color} Channel:\n"
                stats_text += f"  Mean: {np.mean(image[:, :, i]):.2f}\n"
                stats_text += f"  Std: {np.std(image[:, :, i]):.2f}\n"
        
        ax3.text(0.1, 0.5, stats_text, fontsize=10, 
                verticalalignment='center', family='monospace')
        
        # Distribution plot
        ax4 = plt.subplot(2, 3, 4)
        if len(image.shape) == 3:
            for i, color in enumerate(['red', 'green', 'blue']):
                data = image[:, :, i].flatten()
                ax4.hist(data, bins=50, alpha=0.5, label=color.capitalize(), color=color)
            ax4.legend()
        else:
            ax4.hist(image.flatten(), bins=50, color='black', alpha=0.7)
        ax4.set_title('Pixel Value Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Box plot
        ax5 = plt.subplot(2, 3, 5)
        if len(image.shape) == 3:
            data = [image[:, :, i].flatten() for i in range(3)]
            labels = ['Red', 'Green', 'Blue']
            bp = ax5.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['red', 'green', 'blue']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            ax5.boxplot([image.flatten()], labels=['Grayscale'])
        ax5.set_title('Box Plot', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Pixel Value')
        ax5.grid(True, alpha=0.3)
        
        # Heatmap (sample region)
        ax6 = plt.subplot(2, 3, 6)
        sample_size = min(100, image.shape[0], image.shape[1])
        if len(image.shape) == 3:
            sample = np.mean(image[:sample_size, :sample_size], axis=2)
        else:
            sample = image[:sample_size, :sample_size]
        im = ax6.imshow(sample, cmap='viridis', aspect='auto')
        ax6.set_title('Sample Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved statistics visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_3d_surface(self, image: np.ndarray,
                           sample_size: int = 100,
                           save_path: Optional[str] = None) -> None:
        """
        Create 3D surface plot of image.
        
        Args:
            image: Input image
            sample_size: Size of sample region for 3D plot
            save_path: Optional path to save figure
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Sample region
        h, w = gray.shape
        sample_h = min(sample_size, h)
        sample_w = min(sample_size, w)
        sample = gray[:sample_h, :sample_w]
        
        # Create meshgrid
        x = np.arange(sample_w)
        y = np.arange(sample_h)
        X, Y = np.meshgrid(x, y)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, sample, cmap='terrain', 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Pixel Value')
        ax.set_title('3D Surface Plot', fontsize=14, fontweight='bold')
        
        plt.colorbar(surf, ax=ax, shrink=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_time_series(self, images: List[np.ndarray],
                            dates: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Visualize time series of images.
        
        Args:
            images: List of images over time
            dates: Optional list of date strings
            save_path: Optional path to save figure
        """
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        for i, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img)
            ax.axis('off')
            if dates and i < len(dates):
                ax.set_title(dates[i], fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'Time {i+1}', fontsize=10, fontweight='bold')
        
        plt.suptitle('Time Series Visualization', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved time series to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_with_overlays(self, image: np.ndarray,
                               buildings: Optional[List[Dict]] = None,
                               regions: Optional[List[Dict]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Visualize image with building and region overlays.
        
        Args:
            image: Base image
            buildings: Optional list of building detections
            regions: Optional list of land cover regions
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        
        # Draw buildings
        if buildings:
            for building in buildings:
                if 'bbox' in building:
                    x, y, w, h = building['bbox']
                    rect = Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='red', facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
                    
                    # Add confidence label
                    conf = building.get('confidence', 0)
                    ax.text(x, y - 5, f'{conf:.2f}', color='red', 
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Draw regions
        if regions:
            colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
            for region, color in zip(regions, colors):
                if 'polygon' in region:
                    polygon = np.array(region['polygon'])
                    poly = Polygon(polygon, closed=True, 
                                 edgecolor=color, facecolor=color, 
                                 alpha=0.3, linewidth=2)
                    ax.add_patch(poly)
                    
                    # Add label
                    if 'class' in region:
                        center = polygon.mean(axis=0)
                        ax.text(center[0], center[1], region['class'],
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Satellite Image with Overlays', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved overlay visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_dashboard(self, image: np.ndarray,
                        features: Optional[Dict] = None,
                        metadata: Optional[Dict] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive dashboard visualization.
        
        Args:
            image: Main image
            features: Optional extracted features
            metadata: Optional metadata
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Main image
        ax1 = plt.subplot(2, 3, (1, 2))
        ax1.imshow(image)
        ax1.set_title('Satellite Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Histogram
        ax2 = plt.subplot(2, 3, 3)
        if len(image.shape) == 3:
            for i, color in enumerate(['red', 'green', 'blue']):
                hist = np.histogram(image[:, :, i], bins=50)[0]
                ax2.plot(hist, color=color, alpha=0.7, label=color.capitalize())
            ax2.legend()
        else:
            hist = np.histogram(image, bins=50)[0]
            ax2.plot(hist, color='black')
        ax2.set_title('Histogram', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Features (if provided)
        if features:
            ax3 = plt.subplot(2, 3, 4)
            ax3.axis('off')
            features_text = "Extracted Features:\n\n"
            for key, value in list(features.items())[:10]:
                if isinstance(value, (int, float)):
                    features_text += f"{key}: {value:.4f}\n"
                elif isinstance(value, list) and len(value) < 20:
                    features_text += f"{key}: {value}\n"
            ax3.text(0.1, 0.5, features_text, fontsize=9, 
                    verticalalignment='center', family='monospace')
        
        # Metadata (if provided)
        if metadata:
            ax4 = plt.subplot(2, 3, 5)
            ax4.axis('off')
            metadata_text = "Metadata:\n\n"
            for key, value in list(metadata.items())[:10]:
                metadata_text += f"{key}: {value}\n"
            ax4.text(0.1, 0.5, metadata_text, fontsize=9,
                    verticalalignment='center', family='monospace')
        
        # Statistics
        ax5 = plt.subplot(2, 3, 6)
        stats = {
            'Mean': np.mean(image),
            'Std': np.std(image),
            'Min': np.min(image),
            'Max': np.max(image)
        }
        ax5.bar(stats.keys(), stats.values(), color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax5.set_title('Statistics', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved dashboard to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """
    Example usage of AdvancedVisualizer.
    Created by: RSK World (https://rskworld.in)
    """
    print("Advanced Visualization - Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    visualizer = AdvancedVisualizer()
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Statistics visualization
    visualizer.visualize_statistics(sample_image, save_path="visualizations/stats_example.png")
    
    # 3D surface
    visualizer.visualize_3d_surface(sample_image, save_path="visualizations/3d_example.png")
    
    print("\nAdvanced visualization complete!")


if __name__ == "__main__":
    main()
