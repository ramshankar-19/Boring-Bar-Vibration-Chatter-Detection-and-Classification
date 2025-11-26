"""
Generate fresh test data to verify model robustness
"""

from data_generation import DataGenerator
from signal_processing import SignalProcessor
from utils import load_config
import os

# Load config - use correct relative path
config = load_config('config.yaml')  # Changed from '../config.yaml'

# Modify config for test data
config['data']['num_samples_per_class'] = 30  # Generate 30 new test samples

print("Generating new test dataset...")
generator = DataGenerator(config)
test_dataset = generator.generate_dataset('data/test')

print("\nProcessing test signals to images...")
processor = SignalProcessor(config)
processed_test = processor.process_dataset('data/test', 'data/test_processed')

print("\n✅ Test data generation complete!")
print(f"Raw signals: data/test/raw_signals.pkl")
print(f"Processed images: data/test_processed/processed_images.pkl")
