#!/usr/bin/env python3
"""
Data Augmentation for Boring Bar Images
Increases dataset size through transformations
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm

def augment_image(image, augmentation_type):
    """Apply augmentation to image"""
    
    if augmentation_type == 'flip_horizontal':
        return cv2.flip(image, 1)
    
    elif augmentation_type == 'flip_vertical':
        return cv2.flip(image, 0)
    
    elif augmentation_type == 'rotate_90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    elif augmentation_type == 'rotate_270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 5, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image

def augment_dataset(input_dir, output_dir, augmentations_per_image=3):
    """Augment entire dataset"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    aug_types = ['flip_horizontal', 'flip_vertical', 'rotate_90', 
                 'brightness', 'noise']
    
    for state in ['stable', 'transition', 'severe']:
        state_input = input_dir / state
        state_output = output_dir / state
        state_output.mkdir(exist_ok=True)
        
        # Copy original images
        for img_path in state_input.glob('*.png'):
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(state_output / img_path.name), img)
        
        # Generate augmented images
        for img_path in tqdm(list(state_input.glob('*.png')), 
                            desc=f"Augmenting {state}"):
            img = cv2.imread(str(img_path))
            
            for i in range(augmentations_per_image):
                aug_type = np.random.choice(aug_types)
                aug_img = augment_image(img, aug_type)
                
                # Save with augmentation suffix
                aug_name = img_path.stem + f'_aug{i}' + img_path.suffix
                cv2.imwrite(str(state_output / aug_name), aug_img)
    
    print(f"\n✓ Augmentation complete! Output: {output_dir}")

if __name__ == "__main__":
    augment_dataset(
        input_dir='boring_bar_tfr_images',
        output_dir='boring_bar_tfr_augmented',
        augmentations_per_image=3
    )