#!/usr/bin/env python3
"""
Satellite Image Dataset - Advanced Image Processing
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Advanced image processing features including:
- Edge detection
- Image segmentation
- Feature extraction (HOG, LBP, GLCM)
- Image enhancement
- Noise reduction
- Histogram analysis
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from skimage import feature, filters, segmentation, morphology
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
import matplotlib.pyplot as plt


class AdvancedImageProcessor:
    """
    Advanced image processing for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self):
        """Initialize the advanced processor."""
        pass
    
    def detect_edges(self, image: np.ndarray, method: str = 'canny', 
                    **kwargs) -> np.ndarray:
        """
        Detect edges in satellite image.
        
        Args:
            image: Input image (grayscale or RGB)
            method: Edge detection method ('canny', 'sobel', 'laplacian', 'scharr')
            **kwargs: Additional parameters for edge detection
            
        Returns:
            Edge map as binary image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method.lower() == 'canny':
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        elif method.lower() == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(np.absolute(edges))
        
        elif method.lower() == 'laplacian':
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        
        elif method.lower() == 'scharr':
            scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            edges = np.sqrt(scharrx**2 + scharry**2)
            edges = np.uint8(np.absolute(edges))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return edges
    
    def segment_image(self, image: np.ndarray, method: str = 'watershed',
                     num_segments: int = 10, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Segment satellite image into regions.
        
        Args:
            image: Input image
            method: Segmentation method ('watershed', 'slic', 'felzenszwalb', 'quickshift')
            num_segments: Number of segments (for SLIC)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (segmented image, segment properties)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method.lower() == 'watershed':
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Remove noise
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find sure background
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Find sure foreground
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # Find unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            if len(image.shape) == 3:
                markers = cv2.watershed(image, markers)
            else:
                markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), markers)
            
            segmented = markers.astype(np.uint8)
        
        elif method.lower() == 'slic':
            segments = segmentation.slic(image, n_segments=num_segments, compactness=10, 
                                         sigma=1, start_label=1)
            segmented = segments.astype(np.uint8)
        
        elif method.lower() == 'felzenszwalb':
            segments = segmentation.felzenszwalb(image, scale=kwargs.get('scale', 100),
                                                sigma=kwargs.get('sigma', 0.5),
                                                min_size=kwargs.get('min_size', 50))
            segmented = segments.astype(np.uint8)
        
        elif method.lower() == 'quickshift':
            segments = segmentation.quickshift(image, kernel_size=kwargs.get('kernel_size', 3),
                                               max_dist=kwargs.get('max_dist', 6),
                                               ratio=kwargs.get('ratio', 0.5))
            segmented = segments.astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate segment properties
        num_segments = len(np.unique(segmented))
        properties = {
            'num_segments': num_segments,
            'method': method
        }
        
        return segmented, properties
    
    def extract_hog_features(self, image: np.ndarray, 
                         orientations: int = 9,
                         pixels_per_cell: Tuple[int, int] = (8, 8),
                         cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features.
        
        Args:
            image: Input image
            orientations: Number of orientation bins
            pixels_per_cell: Size of cells
            cells_per_block: Number of cells per block
            
        Returns:
            HOG feature vector
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features_hog = feature.hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True
        )
        
        return features_hog
    
    def extract_lbp_features(self, image: np.ndarray,
                             radius: int = 3,
                             n_points: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Local Binary Pattern (LBP) features.
        
        Args:
            image: Input image
            radius: Radius of the circle
            n_points: Number of points to sample
            
        Returns:
            Tuple of (LBP image, histogram)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return lbp, hist
    
    def extract_glcm_features(self, image: np.ndarray,
                              distances: List[int] = [1],
                              angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict:
        """
        Extract Gray-Level Co-occurrence Matrix (GLCM) features.
        
        Args:
            image: Input image
            distances: List of pixel pair distances
            angles: List of angles in radians
            
        Returns:
            Dictionary of GLCM features
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Quantize image to reduce computation
        gray_quantized = (gray / 16).astype(np.uint8) * 16
        
        # Calculate GLCM
        glcm = graycomatrix(gray_quantized, distances=distances, angles=angles,
                           levels=16, symmetric=True, normed=True)
        
        # Extract properties
        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[prop] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values.flatten().tolist()
            }
        
        return features
    
    def enhance_image(self, image: np.ndarray, method: str = 'clahe',
                     **kwargs) -> np.ndarray:
        """
        Enhance satellite image.
        
        Args:
            image: Input image
            method: Enhancement method ('clahe', 'histogram_eq', 'gamma', 'unsharp')
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            enhanced = image.copy()
            if method.lower() == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit', 2.0),
                                       tileGridSize=kwargs.get('tile_size', (8, 8)))
                for i in range(image.shape[2]):
                    enhanced[:, :, i] = clahe.apply(image[:, :, i])
            
            elif method.lower() == 'histogram_eq':
                for i in range(image.shape[2]):
                    enhanced[:, :, i] = cv2.equalizeHist(image[:, :, i])
            
            elif method.lower() == 'gamma':
                gamma = kwargs.get('gamma', 1.5)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(image, table)
            
            elif method.lower() == 'unsharp':
                gaussian = cv2.GaussianBlur(image, (0, 0), kwargs.get('sigma', 2.0))
                enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            if method.lower() == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit', 2.0),
                                       tileGridSize=kwargs.get('tile_size', (8, 8)))
                enhanced = clahe.apply(image)
            elif method.lower() == 'histogram_eq':
                enhanced = cv2.equalizeHist(image)
            elif method.lower() == 'gamma':
                gamma = kwargs.get('gamma', 1.5)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(image, table)
            elif method.lower() == 'unsharp':
                gaussian = cv2.GaussianBlur(image, (0, 0), kwargs.get('sigma', 2.0))
                enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return enhanced
    
    def reduce_noise(self, image: np.ndarray, method: str = 'gaussian',
                    **kwargs) -> np.ndarray:
        """
        Reduce noise in satellite image.
        
        Args:
            image: Input image
            method: Denoising method ('gaussian', 'bilateral', 'median', 'nlm')
            **kwargs: Additional parameters
            
        Returns:
            Denoised image
        """
        if method.lower() == 'gaussian':
            kernel_size = kwargs.get('kernel_size', 5)
            sigma = kwargs.get('sigma', 1.0)
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        elif method.lower() == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        elif method.lower() == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            denoised = cv2.medianBlur(image, kernel_size)
        
        elif method.lower() == 'nlm':
            h = kwargs.get('h', 10)
            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, None, h, h, template_window_size, search_window_size)
            else:
                denoised = cv2.fastNlMeansDenoising(
                    image, None, h, template_window_size, search_window_size)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return denoised
    
    def analyze_histogram(self, image: np.ndarray) -> Dict:
        """
        Analyze image histogram.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with histogram statistics
        """
        if len(image.shape) == 3:
            histograms = {}
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms[color] = {
                    'histogram': hist.flatten().tolist(),
                    'mean': float(np.mean(image[:, :, i])),
                    'std': float(np.std(image[:, :, i])),
                    'min': int(np.min(image[:, :, i])),
                    'max': int(np.max(image[:, :, i]))
                }
            return histograms
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return {
                'histogram': hist.flatten().tolist(),
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'min': int(np.min(image)),
                'max': int(np.max(image))
            }
    
    def extract_all_features(self, image: np.ndarray) -> Dict:
        """
        Extract all available features from image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with all extracted features
        """
        features = {}
        
        # Basic features
        features['basic'] = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image))
        }
        
        # HOG features
        try:
            features['hog'] = self.extract_hog_features(image).tolist()
        except Exception as e:
            features['hog'] = f"Error: {str(e)}"
        
        # LBP features
        try:
            _, lbp_hist = self.extract_lbp_features(image)
            features['lbp'] = lbp_hist.tolist()
        except Exception as e:
            features['lbp'] = f"Error: {str(e)}"
        
        # GLCM features
        try:
            features['glcm'] = self.extract_glcm_features(image)
        except Exception as e:
            features['glcm'] = f"Error: {str(e)}"
        
        # Histogram analysis
        try:
            features['histogram'] = self.analyze_histogram(image)
        except Exception as e:
            features['histogram'] = f"Error: {str(e)}"
        
        return features


def main():
    """
    Example usage of AdvancedImageProcessor.
    Created by: RSK World (https://rskworld.in)
    """
    print("Advanced Image Processing - Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    processor = AdvancedImageProcessor()
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Edge detection
    edges = processor.detect_edges(sample_image, method='canny')
    print(f"Edge detection: {edges.shape}")
    
    # Segmentation
    segmented, props = processor.segment_image(sample_image, method='slic', num_segments=10)
    print(f"Segmentation: {segmented.shape}, {props['num_segments']} segments")
    
    # Feature extraction
    features = processor.extract_all_features(sample_image)
    print(f"Extracted {len(features)} feature types")
    
    print("\nAdvanced processing complete!")


if __name__ == "__main__":
    main()
