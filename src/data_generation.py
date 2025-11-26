"""
Synthetic Data Generation for Boring Bar Vibration Signals

Generates realistic vibration signals for three states:
- Stable: Low amplitude, low frequency oscillations
- Transition: Medium amplitude, mixed frequencies
- Violent: High amplitude, chaotic patterns with chatter
"""

import numpy as np
from scipy import signal
import os
import pickle
from tqdm import tqdm


class DataGenerator:
    """Generate synthetic boring bar vibration signals"""
    
    def __init__(self, config):
        self.config = config
        self.fs = config['data']['sampling_rate']
        self.duration = config['data']['duration']
        self.num_samples = int(self.fs * self.duration)
        self.noise_level = config['signal']['noise_level']
        
    def generate_stable_signal(self):
        """
        Stable vibration: Low amplitude, dominant low frequencies
        Characteristics: 50-200 Hz, smooth sinusoidal patterns
        """
        t = np.linspace(0, self.duration, self.num_samples)
        
        # Dominant low-frequency components
        f1, f2, f3 = 50, 120, 180
        a1, a2, a3 = 1.0, 0.5, 0.3
        
        signal_base = (a1 * np.sin(2 * np.pi * f1 * t) +
                       a2 * np.sin(2 * np.pi * f2 * t + np.random.rand()) +
                       a3 * np.sin(2 * np.pi * f3 * t + np.random.rand()))
        
        # Add gentle amplitude modulation
        modulation = 0.1 * np.sin(2 * np.pi * 5 * t)
        signal_base *= (1 + modulation)
        
        return signal_base
    
    def generate_transition_signal(self):
        """
        Transition vibration: Medium amplitude, mixed frequencies
        Characteristics: 100-800 Hz, increasing instability
        """
        t = np.linspace(0, self.duration, self.num_samples)
        
        # Mixed frequency components
        freqs = [100, 300, 500, 700]
        amps = [1.5, 1.0, 0.8, 0.6]
        phases = np.random.rand(len(freqs)) * 2 * np.pi
        
        signal_base = sum(a * np.sin(2 * np.pi * f * t + p) 
                         for f, a, p in zip(freqs, amps, phases))
        
        # Add time-varying amplitude (instability)
        instability = 0.3 * np.sin(2 * np.pi * 8 * t) * np.exp(t / self.duration)
        signal_base *= (1 + instability)
        
        # Add intermittent bursts
        burst_times = np.random.choice(self.num_samples, size=5, replace=False)
        for bt in burst_times:
            burst_window = signal.windows.gaussian(1000, 100)
            start = max(0, bt - 500)
            end = min(self.num_samples, bt + 500)
            window_slice = slice(max(0, 500 - bt), min(1000, 500 + self.num_samples - bt))
            signal_base[start:end] += 2.0 * burst_window[window_slice]
        
        return signal_base
    
    def generate_violent_signal(self):
        """
        Violent vibration (Chatter): High amplitude, chaotic patterns
        Characteristics: 500-3000 Hz, strong harmonics, random chaos
        """
        t = np.linspace(0, self.duration, self.num_samples)
        
        # High-frequency chatter components
        chatter_freqs = [800, 1200, 1600, 2400, 3000]
        chatter_amps = [3.0, 2.5, 2.0, 1.5, 1.0]
        phases = np.random.rand(len(chatter_freqs)) * 2 * np.pi
        
        signal_base = sum(a * np.sin(2 * np.pi * f * t + p) 
                         for f, a, p in zip(chatter_freqs, chatter_amps, phases))
        
        # Add chaotic amplitude modulation
        chaos = 0.5 * np.sin(2 * np.pi * 15 * t) * np.random.randn(self.num_samples) * 0.3
        signal_base *= (1 + chaos)
        
        # Add random impulses (tool impacts)
        num_impulses = np.random.randint(10, 20)
        impulse_positions = np.random.choice(self.num_samples, size=num_impulses, replace=False)
        for pos in impulse_positions:
            impulse = signal.unit_impulse(self.num_samples, pos) * np.random.uniform(5, 10)
            signal_base += impulse
        
        # Add broadband noise
        signal_base += np.random.randn(self.num_samples) * 1.5
        
        return signal_base
    
    def add_realistic_noise(self, sig):
        """Add realistic measurement noise"""
        # White Gaussian noise
        noise = np.random.randn(len(sig)) * self.noise_level * np.std(sig)
        
        # Low-frequency drift (sensor drift)
        t = np.linspace(0, self.duration, len(sig))
        drift = 0.05 * np.sin(2 * np.pi * 2 * t) * np.mean(np.abs(sig))
        
        # Occasional spikes (electromagnetic interference)
        num_spikes = np.random.randint(0, 3)
        spike_positions = np.random.choice(len(sig), size=num_spikes, replace=False)
        for pos in spike_positions:
            noise[pos] += np.random.uniform(-2, 2) * np.std(sig)
        
        return sig + noise + drift
    
    def generate_multi_channel_sample(self, state_label):
        """
        Generate 3-axis acceleration + sound pressure signal
        
        Args:
            state_label: 0=Stable, 1=Transition, 2=Violent
        """
        # Generate base signal
        if state_label == 0:
            base_signal = self.generate_stable_signal()
        elif state_label == 1:
            base_signal = self.generate_transition_signal()
        else:
            base_signal = self.generate_violent_signal()
        
        # 3-axis acceleration (with different phase/amplitude characteristics)
        acc_x = base_signal * np.random.uniform(0.8, 1.2)
        acc_y = base_signal * np.random.uniform(0.7, 1.1) + \
                np.random.randn(self.num_samples) * 0.1 * np.std(base_signal)
        acc_z = base_signal * np.random.uniform(0.9, 1.3) + \
                np.random.randn(self.num_samples) * 0.15 * np.std(base_signal)
        
        # Sound pressure (correlated but with different frequency response)
        sound = signal.filtfilt(*signal.butter(4, 0.8), base_signal)
        sound *= np.random.uniform(1.0, 1.5)
        
        # Add realistic noise to all channels
        acc_x = self.add_realistic_noise(acc_x)
        acc_y = self.add_realistic_noise(acc_y)
        acc_z = self.add_realistic_noise(acc_z)
        sound = self.add_realistic_noise(sound)
        
        # Stack into multi-channel array
        multi_channel = np.stack([acc_x, acc_y, acc_z, sound], axis=0)
        
        return multi_channel
    
    def generate_dataset(self, save_path):
        """
        Generate complete dataset with all three classes
        
        Returns:
            Dictionary with signals and labels
        """
        num_per_class = self.config['data']['num_samples_per_class']
        
        data = []
        labels = []
        
        print("Generating synthetic boring bar vibration dataset...")
        
        # Generate samples for each class
        for state_label, state_name in enumerate(['Stable', 'Transition', 'Violent']):
            print(f"\nGenerating {num_per_class} samples for {state_name} state...")
            
            for i in tqdm(range(num_per_class), desc=state_name):
                multi_channel_signal = self.generate_multi_channel_sample(state_label)
                data.append(multi_channel_signal)
                labels.append(state_label)
        
        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)
        
        # Save raw signals
        os.makedirs(save_path, exist_ok=True)
        dataset = {
            'signals': data,
            'labels': labels,
            'sampling_rate': self.fs,
            'duration': self.duration,
            'class_names': ['Stable', 'Transition', 'Violent']
        }
        
        with open(os.path.join(save_path, 'raw_signals.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\nDataset saved to {save_path}")
        print(f"Total samples: {len(data)}")
        print(f"Signal shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return dataset


if __name__ == '__main__':
    # Test data generation
    import yaml
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generator = DataGenerator(config)
    dataset = generator.generate_dataset(config['paths']['raw_data'])
