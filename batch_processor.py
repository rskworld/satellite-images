#!/usr/bin/env python3
"""
Satellite Image Dataset - Batch Processing Utilities
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Batch processing utilities including:
- Batch image processing
- Data augmentation
- Parallel processing
- Progress tracking
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import cv2
from data_loader import SatelliteDatasetLoader
from process_images import SatelliteImageProcessor
from advanced_processing import AdvancedImageProcessor


class BatchProcessor:
    """
    Batch processing utilities for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self, data_dir: str = "data", max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            data_dir: Data directory
            max_workers: Maximum number of worker threads/processes
        """
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.loader = SatelliteDatasetLoader(str(data_dir))
        self.processor = SatelliteImageProcessor(str(data_dir))
        self.advanced_processor = AdvancedImageProcessor()
    
    def process_batch(self, image_ids: List[str],
                     processing_function: Callable,
                     output_dir: Optional[str] = None,
                     save_results: bool = True) -> List[Dict]:
        """
        Process a batch of images.
        
        Args:
            image_ids: List of image IDs to process
            processing_function: Function to apply to each image
            output_dir: Optional output directory
            save_results: Whether to save processing results
            
        Returns:
            List of processing results
        """
        results = []
        output_path = Path(output_dir) if output_dir else None
        
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {len(image_ids)} images...")
        print("Created by: RSK World (https://rskworld.in)")
        
        for image_id in tqdm(image_ids, desc="Processing"):
            try:
                # Load image
                image, labels = self.loader.load_image_pair(image_id)
                
                if image is None:
                    results.append({
                        'image_id': image_id,
                        'status': 'failed',
                        'error': 'Image not found'
                    })
                    continue
                
                # Process image
                result = processing_function(image, labels)
                result['image_id'] = image_id
                result['status'] = 'success'
                
                # Save if requested
                if save_results and output_path:
                    result_path = output_path / f"{image_id}_result.json"
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=4, default=str)
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'image_id': image_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def process_batch_parallel(self, image_ids: List[str],
                              processing_function: Callable,
                              output_dir: Optional[str] = None,
                              use_processes: bool = False) -> List[Dict]:
        """
        Process batch in parallel.
        
        Args:
            image_ids: List of image IDs
            processing_function: Function to apply
            output_dir: Optional output directory
            use_processes: Use processes instead of threads
            
        Returns:
            List of results
        """
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        def process_single(image_id: str) -> Dict:
            try:
                image, labels = self.loader.load_image_pair(image_id)
                if image is None:
                    return {'image_id': image_id, 'status': 'failed', 'error': 'Image not found'}
                
                result = processing_function(image, labels)
                result['image_id'] = image_id
                result['status'] = 'success'
                return result
            except Exception as e:
                return {'image_id': image_id, 'status': 'failed', 'error': str(e)}
        
        print(f"Processing {len(image_ids)} images in parallel...")
        print("Created by: RSK World (https://rskworld.in)")
        
        results = []
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single, img_id) for img_id in image_ids]
            for future in tqdm(futures, desc="Processing"):
                results.append(future.result())
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            results_path = output_path / "batch_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
        
        return results


class DataAugmenter:
    """
    Data augmentation for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    @staticmethod
    def augment_image(image: np.ndarray, augmentation_type: str = 'all') -> List[np.ndarray]:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation ('all', 'rotation', 'flip', 'brightness', 'noise')
            
        Returns:
            List of augmented images
        """
        augmented = [image.copy()]
        
        if augmentation_type in ['all', 'rotation']:
            # Rotations
            for angle in [90, 180, 270]:
                center = (image.shape[1] // 2, image.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
                augmented.append(rotated)
        
        if augmentation_type in ['all', 'flip']:
            # Flips
            augmented.append(cv2.flip(image, 0))  # Vertical
            augmented.append(cv2.flip(image, 1))  # Horizontal
            augmented.append(cv2.flip(image, -1))  # Both
        
        if augmentation_type in ['all', 'brightness']:
            # Brightness adjustments
            for factor in [0.7, 1.3]:
                bright = np.clip(image * factor, 0, 1 if image.max() <= 1 else 255)
                augmented.append(bright.astype(image.dtype))
        
        if augmentation_type in ['all', 'noise']:
            # Add noise
            noise = np.random.normal(0, 0.05, image.shape)
            noisy = np.clip(image + noise, 0, 1 if image.max() <= 1 else 255)
            augmented.append(noisy.astype(image.dtype))
        
        return augmented
    
    @staticmethod
    def augment_dataset(image_ids: List[str], 
                       loader: SatelliteDatasetLoader,
                       output_dir: str,
                       augmentation_type: str = 'all') -> List[str]:
        """
        Augment entire dataset.
        
        Args:
            image_ids: List of image IDs
            loader: Dataset loader
            output_dir: Output directory
            augmentation_type: Type of augmentation
            
        Returns:
            List of augmented image paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        augmented_paths = []
        
        print(f"Augmenting {len(image_ids)} images...")
        print("Created by: RSK World (https://rskworld.in)")
        
        for image_id in tqdm(image_ids, desc="Augmenting"):
            image, labels = loader.load_image_pair(image_id)
            
            if image is None:
                continue
            
            # Normalize for augmentation
            if image.max() > 1:
                image_norm = image.astype(np.float32) / 255.0
            else:
                image_norm = image.astype(np.float32)
            
            # Augment
            augmented_images = DataAugmenter.augment_image(image_norm, augmentation_type)
            
            # Save augmented images
            for i, aug_img in enumerate(augmented_images):
                if aug_img.max() <= 1:
                    aug_img = (aug_img * 255).astype(np.uint8)
                
                output_file = output_path / f"{image_id}_aug_{i}.png"
                cv2.imwrite(str(output_file), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                augmented_paths.append(str(output_file))
        
        return augmented_paths


def example_usage():
    """Example usage of batch processing."""
    print("Batch Processing - Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    processor = BatchProcessor()
    
    # Get available images
    info = processor.loader.get_dataset_info()
    image_ids = [Path(img).stem for img in info['images'][:5]]
    
    if not image_ids:
        print("No images found. Please add images to the dataset first.")
        return
    
    # Define processing function
    def process_function(image, labels):
        # Extract features using available methods
        features = processor.advanced_processor.extract_all_features(image)
        # Get basic statistics
        stats = {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'shape': image.shape
        }
        return {
            'features': {k: v for k, v in features.items() if not isinstance(v, str)},
            'statistics': stats
        }
    
    # Process batch
    results = processor.process_batch(
        image_ids=image_ids,
        processing_function=process_function,
        output_dir="output/batch_results"
    )
    
    print(f"\nProcessed {len(results)} images")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")


if __name__ == "__main__":
    example_usage()

