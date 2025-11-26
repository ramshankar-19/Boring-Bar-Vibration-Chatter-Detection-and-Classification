"""
Signal Processing Module: Wavelet Denoising + SPWVD Time-Frequency Analysis
"""

import numpy as np
import pywt
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import pickle


class SignalProcessor:
    """Signal processing with wavelet denoising and SPWVD"""
    
    def __init__(self, config):
        self.config = config
        self.fs = config['data']['sampling_rate']
        self.image_size = tuple(config['data']['image_size'])
        
    def wavelet_denoise(self, signal_data, wavelet='coif5', level=3, threshold_method='hard'):
        """
        Wavelet threshold denoising
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet basis (coif5 as per paper)
            level: Decomposition level (3 as per paper)
            threshold_method: 'hard' or 'soft'
        
        Returns:
            Denoised signal
        """
        # Decompose signal
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Calculate threshold using unbiased likelihood estimation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # Apply thresholding to detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients
        for detail_coeff in coeffs[1:]:
            if threshold_method == 'hard':
                thresholded = pywt.threshold(detail_coeff, threshold, mode='hard')
            else:
                thresholded = pywt.threshold(detail_coeff, threshold, mode='soft')
            denoised_coeffs.append(thresholded)
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
        
        # Handle length mismatch
        if len(denoised_signal) > len(signal_data):
            denoised_signal = denoised_signal[:len(signal_data)]
        elif len(denoised_signal) < len(signal_data):
            denoised_signal = np.pad(denoised_signal, 
                                    (0, len(signal_data) - len(denoised_signal)), 
                                    mode='edge')
        
        return denoised_signal
    
    def smoothed_pseudo_wvd(self, sig, time_window=None, freq_window=None):
        """
        Smoothed Pseudo Wigner-Ville Distribution (SPWVD)
        
        Args:
            sig: Input signal
            time_window: Time smoothing window (Gaussian)
            freq_window: Frequency smoothing window (Gaussian)
        
        Returns:
            2D time-frequency distribution
        """
        N = len(sig)
        
        # Default window sizes
        if time_window is None:
            time_window = signal.windows.gaussian(51, std=10)
        if freq_window is None:
            freq_window = signal.windows.gaussian(51, std=10)
        
        # Normalize windows
        time_window /= np.sum(time_window)
        freq_window /= np.sum(freq_window)
        
        # Initialize SPWVD matrix
        n_freq = 512  # Frequency bins
        n_time = min(N // 4, 256)  # Time bins
        spwvd = np.zeros((n_freq, n_time))
        
        time_indices = np.linspace(0, N - 1, n_time, dtype=int)
        
        for i, t in enumerate(time_indices):
            # Local autocorrelation
            max_lag = min(t, N - 1 - t, len(freq_window) // 2)
            
            local_corr = np.zeros(2 * max_lag + 1, dtype=complex)
            for tau in range(-max_lag, max_lag + 1):
                if 0 <= t + tau < N and 0 <= t - tau < N:
                    local_corr[tau + max_lag] = sig[t + tau] * np.conj(sig[t - tau])
            
            # Apply frequency window (smoothing)
            window_slice = freq_window[len(freq_window)//2 - max_lag:len(freq_window)//2 + max_lag + 1]
            if len(window_slice) == len(local_corr):
                local_corr *= window_slice
            
            # FFT to get frequency distribution
            freq_dist = np.fft.fft(local_corr, n=n_freq)
            spwvd[:, i] = np.abs(np.fft.fftshift(freq_dist))
        
        # Apply time smoothing
        for f in range(n_freq):
            spwvd[f, :] = np.convolve(spwvd[f, :], time_window, mode='same')
        
        return spwvd
    
    def signal_to_image(self, sig, channel_name='signal'):
        """
        Convert signal to 2D time-frequency image using SPWVD
        
        Args:
            sig: Input signal (1D array)
            channel_name: Name for debugging
        
        Returns:
            2D grayscale image (256x256)
        """
        # Apply wavelet denoising
        denoised = self.wavelet_denoise(sig)
        
        # Compute SPWVD
        tf_distribution = self.smoothed_pseudo_wvd(denoised)
        
        # Convert to logarithmic scale for better visualization
        tf_distribution = np.log1p(tf_distribution)
        
        # Normalize to 0-255
        tf_min, tf_max = tf_distribution.min(), tf_distribution.max()
        if tf_max > tf_min:
            tf_normalized = 255 * (tf_distribution - tf_min) / (tf_max - tf_min)
        else:
            tf_normalized = np.zeros_like(tf_distribution)
        
        tf_normalized = tf_normalized.astype(np.uint8)
        
        # Resize to target image size
        img = Image.fromarray(tf_normalized)
        img_resized = img.resize(self.image_size, Image.BILINEAR)
        
        return np.array(img_resized)
    
    def process_multi_channel_to_rgb(self, multi_channel_signal):
        """
        Process multi-channel signal (4 channels) to RGB image (3 channels)
        
        Strategy: Use 3-axis acceleration for RGB channels
        (Sound pressure used as auxiliary feature, can be averaged with one axis)
        
        Args:
            multi_channel_signal: (4, num_samples) array
                                 [acc_x, acc_y, acc_z, sound]
        
        Returns:
            RGB image (256, 256, 3)
        """
        acc_x, acc_y, acc_z, sound = multi_channel_signal
        
        # Convert each acceleration axis to time-frequency image
        img_x = self.signal_to_image(acc_x, 'acc_x')
        img_y = self.signal_to_image(acc_y, 'acc_y')
        img_z = self.signal_to_image(acc_z, 'acc_z')
        
        # Stack as RGB channels
        rgb_image = np.stack([img_x, img_y, img_z], axis=-1)
        
        return rgb_image
    
    def process_dataset(self, raw_data_path, save_path):
        """
        Process entire dataset: raw signals → RGB time-frequency images
        
        Args:
            raw_data_path: Path to raw signals pickle file
            save_path: Path to save processed images
        """
        # Load raw data
        print(f"Loading raw data from {raw_data_path}...")
        with open(os.path.join(raw_data_path, 'raw_signals.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        signals = dataset['signals']
        labels = dataset['labels']
        
        print(f"Processing {len(signals)} samples...")
        
        processed_images = []
        processed_labels = []
        
        for i, (multi_channel, label) in enumerate(tqdm(zip(signals, labels), 
                                                        total=len(signals),
                                                        desc="Processing signals")):
            try:
                # Convert to RGB time-frequency image
                rgb_img = self.process_multi_channel_to_rgb(multi_channel)
                processed_images.append(rgb_img)
                processed_labels.append(label)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        processed_images = np.array(processed_images)
        processed_labels = np.array(processed_labels)
        
        # Save processed dataset
        os.makedirs(save_path, exist_ok=True)
        processed_dataset = {
            'images': processed_images,
            'labels': processed_labels,
            'class_names': dataset['class_names']
        }
        
        with open(os.path.join(save_path, 'processed_images.pkl'), 'wb') as f:
            pickle.dump(processed_dataset, f)
        
        print(f"\nProcessed dataset saved to {save_path}")
        print(f"Images shape: {processed_images.shape}")
        print(f"Labels shape: {processed_labels.shape}")
        
        return processed_dataset


if __name__ == '__main__':
    import yaml
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processor = SignalProcessor(config)
    dataset = processor.process_dataset(
        config['paths']['raw_data'],
        config['paths']['processed_data']
    )
