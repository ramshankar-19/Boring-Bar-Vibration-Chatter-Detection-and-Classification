"""
Visualize training data: signals and time-frequency images
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def visualize_training_examples():
    """Visualize processed training images and raw signals"""
    
    # Load processed images
    print("Loading processed training data...")
    with open('data/processed/processed_images.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    images = processed_data['images']
    labels = processed_data['labels']
    class_names = processed_data['class_names']
    
    # Load raw signals
    print("Loading raw signal data...")
    with open('data/raw/raw_signals.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    
    signals = raw_data['signals']
    fs = raw_data['sampling_rate']
    
    print(f"Loaded {len(images)} samples")
    print(f"Class distribution: {class_names}")
    print(f"Signal shape: {signals.shape}")
    print(f"Image shape: {images.shape}\n")
    
    # Create visualizations
    visualize_image_grid(images, labels, class_names)
    visualize_rgb_channels(images, labels, class_names)
    visualize_signal_to_image(signals, images, labels, class_names, fs)
    visualize_class_comparison(signals, images, labels, class_names, fs)
    
    print("\n✅ All visualizations saved to results/ folder!")


def visualize_image_grid(images, labels, class_names):
    """Display grid of time-frequency images from each class"""
    
    print("Creating image grid visualization...")
    
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('Training Data: Time-Frequency Images (SPWVD)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for class_idx, class_name in enumerate(class_names):
        # Get indices for this class
        class_indices = np.where(labels == class_idx)[0]
        
        # Select 6 random samples
        sample_indices = np.random.choice(class_indices, size=6, replace=False)
        
        for col, sample_idx in enumerate(sample_indices):
            ax = axes[class_idx, col]
            
            # Display image
            img = images[sample_idx]
            ax.imshow(img)
            ax.axis('off')
            
            # Add title for first column
            if col == 0:
                ax.set_ylabel(class_name, fontsize=13, fontweight='bold', rotation=0, 
                            ha='right', va='center')
            
            # Add sample number at top
            ax.set_title(f'#{sample_idx}', fontsize=9)
    
    plt.tight_layout()
    save_path = 'results/training_images_grid.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def visualize_rgb_channels(images, labels, class_names):
    """Show RGB channel breakdown for each class"""
    
    print("Creating RGB channel breakdown...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('RGB Channel Breakdown (R=Acc_X, G=Acc_Y, B=Acc_Z)', 
                 fontsize=15, fontweight='bold')
    
    for class_idx, class_name in enumerate(class_names):
        # Get one sample from this class
        sample_idx = np.where(labels == class_idx)[0][0]
        img = images[sample_idx]
        
        # Original RGB
        ax = fig.add_subplot(gs[class_idx, 0])
        ax.imshow(img)
        ax.set_title(f'{class_name} - Full RGB', fontweight='bold')
        ax.axis('off')
        
        # Red channel (X-axis acceleration)
        ax = fig.add_subplot(gs[class_idx, 1])
        ax.imshow(img[:, :, 0], cmap='Reds')
        ax.set_title('R: Acc_X', fontsize=10)
        ax.axis('off')
        
        # Green channel (Y-axis acceleration)
        ax = fig.add_subplot(gs[class_idx, 2])
        ax.imshow(img[:, :, 1], cmap='Greens')
        ax.set_title('G: Acc_Y', fontsize=10)
        ax.axis('off')
        
        # Blue channel (Z-axis acceleration)
        ax = fig.add_subplot(gs[class_idx, 3])
        ax.imshow(img[:, :, 2], cmap='Blues')
        ax.set_title('B: Acc_Z', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = 'results/rgb_channel_breakdown.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def visualize_signal_to_image(signals, images, labels, class_names, fs):
    """Show transformation from raw signal to time-frequency image"""
    
    print("Creating signal-to-image transformation visualization...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Signal Processing Pipeline: Raw Signal → Time-Frequency Image', 
                 fontsize=15, fontweight='bold')
    
    for class_idx, class_name in enumerate(class_names):
        # Get one sample
        sample_idx = np.where(labels == class_idx)[0][0]
        signal = signals[sample_idx]  # (4, 40000) - 4 channels
        img = images[sample_idx]
        
        # Plot raw signal (X-axis acceleration)
        ax = fig.add_subplot(gs[class_idx, 0])
        time = np.arange(len(signal[0])) / fs
        ax.plot(time[:2000], signal[0][:2000], linewidth=0.5, color='steelblue')
        ax.set_title(f'{class_name}\nRaw Signal (Acc_X)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 0.05])  # First 50ms
        
        # Plot frequency spectrum
        ax = fig.add_subplot(gs[class_idx, 1])
        from scipy.fft import fft, fftfreq
        fft_vals = np.abs(fft(signal[0]))
        freqs = fftfreq(len(signal[0]), 1/fs)
        
        # Only positive frequencies up to 5 kHz
        mask = (freqs > 0) & (freqs < 5000)
        ax.plot(freqs[mask], fft_vals[mask], linewidth=0.8, color='green')
        ax.set_title('Frequency Spectrum', fontweight='bold', fontsize=11)
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('Magnitude', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Plot time-frequency image
        ax = fig.add_subplot(gs[class_idx, 2])
        ax.imshow(img)
        ax.set_title('Time-Frequency Image\n(SPWVD)', fontweight='bold', fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = 'results/signal_to_image_pipeline.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def visualize_class_comparison(signals, images, labels, class_names, fs):
    """Compare characteristics across all three classes"""
    
    print("Creating class comparison visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Class Comparison: Stable vs Transition vs Violent', 
                 fontsize=16, fontweight='bold')
    
    colors = ['green', 'orange', 'red']
    
    # Time domain comparison
    ax = fig.add_subplot(gs[0, :])
    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        sample_idx = np.where(labels == class_idx)[0][0]
        signal = signals[sample_idx][0]  # X-axis
        time = np.arange(len(signal)) / fs
        
        # Plot only first 0.1 seconds
        mask = time < 0.1
        ax.plot(time[mask], signal[mask], label=class_name, alpha=0.7, 
               linewidth=1.5, color=color)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Time Domain Signals (First 100ms)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Time-frequency images comparison
    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        ax = fig.add_subplot(gs[1, class_idx])
        sample_idx = np.where(labels == class_idx)[0][0]
        img = images[sample_idx]
        
        ax.imshow(img)
        ax.set_title(f'{class_name}', fontsize=13, fontweight='bold', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = 'results/class_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print("VISUALIZING TRAINING DATA")
    print("="*60 + "\n")
    
    visualize_training_examples()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files in results/:")
    print("  1. training_images_grid.png       - 18 sample images (6 per class)")
    print("  2. rgb_channel_breakdown.png      - RGB channel analysis")
    print("  3. signal_to_image_pipeline.png   - Processing pipeline")
    print("  4. class_comparison.png           - Class characteristics")
