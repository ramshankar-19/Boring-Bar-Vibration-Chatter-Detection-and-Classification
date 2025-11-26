"""
Utility functions for the project
"""

import yaml
import random
import numpy as np
import torch
import os
from pathlib import Path


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    # Handle relative paths from anywhere
    if not os.path.isabs(config_path):
        # Get the project root (parent of src/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config['paths']['raw_data'],
        config['paths']['processed_data'],
        config['paths']['dataset_path'],
        config['paths']['model_save'],
        config['paths']['results'],
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
