#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPWVD Transformation for Boring Bar Data
-----------------------------------------
Converts time-series acceleration data to 256x256 time-frequency images
as per Liu et al. (2023) methodology
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import stft
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

class SPWVDTransformer:
    """Smoothed Pseudo-Wigner-Ville Distribution transformer"""
    
    def __init__(self, fs=40000, image_size=256):
        self.fs = fs
        self.image_size = image_size
        
    def compute_spwvd(self, x, window_length=256, time_window=None, freq_window=None):
        """
        Compute Smoothed Pseudo-Wigner-Ville Distribution
        
        Parameters:
        - x: input signal
        - window_length: length for STFT
        - time_window: time smoothing window (Gaussian)
        - freq_window: frequency smoothing window (Gaussian)
        """
        
        # Use STFT as approximation (computationally efficient)
        # For production, you can implement full SPWVD
        f, t, Zxx = stft(x, fs=self.fs, nperseg=window_length, 
                         noverlap=window_length//2)
        
        # Convert to power
        tfr = np.abs(Zxx)**2
        
        # Apply Gaussian smoothing (approximates SPWVD kernel)
        if time_window is not None and freq_window is not None:
            from scipy.ndimage import gaussian_filter
            tfr = gaussian_filter(tfr, sigma=[freq_window, time_window])
        
        return t, f, tfr
    
    def normalize_to_image(self, tfr):
        """Convert TFR to 256x256 grayscale image (0-255)"""
        
        # Resize to 256x256
        tfr_resized = cv2.resize(tfr, (self.image_size, self.image_size))
        
        # Log scale for better visualization
        tfr_log = np.log10(tfr_resized + 1e-10)
        
        # Normalize to 0-255
        tfr_norm = (tfr_log - tfr_log.min()) / (tfr_log.max() - tfr_log.min())
        image = (tfr_norm * 255).astype(np.uint8)
        
        return image
    
    def process_experiment(self, df_exp):
        """Process single experiment to create 256x256x3 matrix"""
        
        # Extract acceleration data
        a_x = df_exp['accel_x_m_s2'].values
        
        # Compute SPWVD
        t, f, tfr = self.compute_spwvd(a_x)
        
        # Convert to image
        image = self.normalize_to_image(tfr)
        
        # Create 3-channel image (grayscale replicated)
        # In actual implementation, you could use 3 different sensor channels
        image_3ch = np.stack([image, image, image], axis=-1)
        
        return image_3ch, t, f, tfr


def process_dataset(input_csv, output_dir):
    """Process entire dataset and save transformed images"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each class
    for state in ['stable', 'transition', 'severe']:
        (output_dir / state).mkdir(exist_ok=True)
    
    print(f"\nLoading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Get unique experiments
    exp_ids = df['exp_id'].unique()
    print(f"Total experiments to process: {len(exp_ids)}")
    
    # Initialize transformer
    transformer = SPWVDTransformer()
    
    # Process each experiment
    processed_data = []
    
    for exp_id in tqdm(exp_ids, desc="Processing experiments"):
        df_exp = df[df['exp_id'] == exp_id]
        state = df_exp['state'].iloc[0]
        
        # Transform to time-frequency image
        image, t, f, tfr = transformer.process_experiment(df_exp)
        
        # Save image
        image_path = output_dir / state / f"{exp_id}.png"
        Image.fromarray(image).save(image_path)
        
        # Store metadata
        processed_data.append({
            'exp_id': exp_id,
            'state': state,
            'image_path': str(image_path)
        })
    
    # Save processed metadata
    df_processed = pd.DataFrame(processed_data)
    metadata_path = output_dir / 'processed_metadata.csv'
    df_processed.to_csv(metadata_path, index=False)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Saved {len(processed_data)} images to: {output_dir}")
    print(f"✓ Saved metadata to: {metadata_path}")
    
    return df_processed


if __name__ == "__main__":
    # Process the generated dataset
    process_dataset(
        input_csv='boring_bar_dataset/boring_bar_full_dataset.csv',
        output_dir='boring_bar_tfr_images'
    )