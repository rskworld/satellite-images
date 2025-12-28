#!/usr/bin/env python3
"""
Satellite Image Dataset - Batch Processing and Augmentation
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Batch processing features:
- Process multiple images
- Data augmentation
- Parallel processing
- Export utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import random


class BatchProcessor:
    """
    Batch processing utilities for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            num_workers: Number of parallel workers (None = auto)
        """
        self.num_workers = num_workers or mp.cpu_count()
    
    def process_batch(self, image_paths: List[Path],
                     processor_func: Callable,
                     output_dir: Optional[Path] = None,
                     **kwargs) -> List[Dict]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image file paths
            processor_func: Function to process each image
            output_dir: Optional output directory
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processing results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images with {self.num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for img_path in image_paths:
                future = executor.submit(self._process_single, 
                                       img_path, processor_func, output_dir, **kwargs)
                futures.append(future)
            
            for future in tqdm(futures, desc="Processing"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    results.append({'error': str(e)})
        
        return results
    
    def _process_single(self, img_path: Path, processor_func: Callable,
                       output_dir: Optional[Path], **kwargs) -> Dict:
        """Process a single image."""
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                return {'error': f'Could not load {img_path}'}
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process
            result = processor_func(image, **kwargs)
            
            # Save if output directory provided
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{img_path.stem}_processed.png"
                cv2.imwrite(str(output_path), 
                           cv2.cvtColor(result if isinstance(result, np.ndarray) else image, 
                                      cv2.COLOR_RGB2BGR))
            
            return {
                'input': str(img_path),
                'output': str(output_path) if output_dir else None,
                'success': True
            }
        except Exception as e:
            return {'error': str(e), 'input': str(img_path)}


class ImageAugmenter:
    """
    Image augmentation utilities for satellite images.
    Created by: RSK World (https://rskworld.in)
    """
    
    def __init__(self):
        """Initialize augmenter."""
        pass
    
    def augment_image(self, image: np.ndarray, 
                     augmentations: List[str] = None) -> List[np.ndarray]:
        """
        Apply multiple augmentations to an image.
        
        Args:
            image: Input image
            augmentations: List of augmentation names
            
        Returns:
            List of augmented images
        """
        if augmentations is None:
            augmentations = ['flip_horizontal', 'flip_vertical', 'rotate', 
                           'brightness', 'contrast', 'noise']
        
        augmented = []
        
        for aug_name in augmentations:
            try:
                aug_func = getattr(self, f'_{aug_name}')
                aug_img = aug_func(image.copy())
                augmented.append(aug_img)
            except AttributeError:
                print(f"Unknown augmentation: {aug_name}")
        
        return augmented
    
    def _flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def _flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically."""
        return cv2.flip(image, 0)
    
    def _rotate(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """Rotate image."""
        if angle is None:
            angle = random.choice([90, 180, 270])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        return cv2.warpAffine(image, matrix, (w, h))
    
    def _brightness(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust brightness."""
        if factor is None:
            factor = random.uniform(0.7, 1.3)
        
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def _contrast(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust contrast."""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _noise(self, image: np.ndarray, amount: float = None) -> np.ndarray:
        """Add noise."""
        if amount is None:
            amount = random.uniform(0.01, 0.05)
        
        noise = np.random.normal(0, amount * 255, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def _crop(self, image: np.ndarray, crop_size: Tuple[int, int] = None) -> np.ndarray:
        """Random crop."""
        h, w = image.shape[:2]
        if crop_size is None:
            crop_size = (int(h * 0.8), int(w * 0.8))
        
        y = random.randint(0, h - crop_size[0])
        x = random.randint(0, w - crop_size[1])
        
        return image[y:y+crop_size[0], x:x+crop_size[1]]
    
    def _scale(self, image: np.ndarray, scale_factor: float = None) -> np.ndarray:
        """Scale image."""
        if scale_factor is None:
            scale_factor = random.uniform(0.8, 1.2)
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def create_augmented_dataset(self, image_paths: List[Path],
                                output_dir: Path,
                                num_augmentations: int = 5) -> Dict:
        """
        Create augmented dataset from images.
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory
            num_augmentations: Number of augmentations per image
            
        Returns:
            Dictionary with dataset info
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'original_images': len(image_paths),
            'augmentations_per_image': num_augmentations,
            'total_images': len(image_paths) * (1 + num_augmentations),
            'files': []
        }
        
        augmentation_methods = ['flip_horizontal', 'flip_vertical', 'rotate',
                               'brightness', 'contrast', 'noise', 'crop', 'scale']
        
        for img_path in tqdm(image_paths, desc="Augmenting"):
            # Load original
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Save original
            orig_output = output_dir / f"{img_path.stem}_orig.png"
            cv2.imwrite(str(orig_output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            dataset_info['files'].append(str(orig_output))
            
            # Create augmentations
            selected_augs = random.sample(augmentation_methods, 
                                        min(num_augmentations, len(augmentation_methods)))
            
            for i, aug_name in enumerate(selected_augs):
                try:
                    aug_func = getattr(self, f'_{aug_name}')
                    aug_img = aug_func(image.copy())
                    
                    aug_output = output_dir / f"{img_path.stem}_aug_{i+1}_{aug_name}.png"
                    cv2.imwrite(str(aug_output), 
                               cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    dataset_info['files'].append(str(aug_output))
                except Exception as e:
                    print(f"Error applying {aug_name}: {e}")
        
        # Save dataset info
        info_file = output_dir / "augmented_dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        return dataset_info


class ExportUtils:
    """
    Export utilities for processed data.
    Created by: RSK World (https://rskworld.in)
    """
    
    @staticmethod
    def export_to_numpy(images: List[np.ndarray], output_path: Path) -> None:
        """Export images to NumPy array file."""
        array = np.array(images)
        np.save(output_path, array)
        print(f"Exported {len(images)} images to {output_path}")
    
    @staticmethod
    def export_metadata(metadata: List[Dict], output_path: Path) -> None:
        """Export metadata to JSON."""
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Exported metadata to {output_path}")
    
    @staticmethod
    def create_manifest(image_paths: List[Path], 
                       output_path: Path,
                       metadata: Optional[List[Dict]] = None) -> None:
        """Create manifest file for dataset."""
        manifest = {
            'total_images': len(image_paths),
            'images': [str(p) for p in image_paths],
            'metadata': metadata or []
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        
        print(f"Created manifest at {output_path}")


def main():
    """
    Example usage of batch processing.
    Created by: RSK World (https://rskworld.in)
    """
    print("Batch Processing - Example")
    print("Created by: RSK World (https://rskworld.in)")
    print("-" * 50)
    
    # Example: Process images
    processor = BatchProcessor()
    augmenter = ImageAugmenter()
    
    # Create sample augmentation
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    augmented = augmenter.augment_image(sample_image, ['flip_horizontal', 'brightness'])
    print(f"Created {len(augmented)} augmented versions")
    
    print("\nBatch processing utilities ready!")


if __name__ == "__main__":
    main()

