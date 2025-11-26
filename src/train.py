"""
Training Script for Shuffle-BiLSTM Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from model import ShuffleBiLSTM
from utils import load_config, set_seed, create_directories, count_parameters, AverageMeter


class BoringBarDataset(Dataset):
    """PyTorch Dataset for boring bar vibration images"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).float() / 255.0
        
        # Change from (H, W, C) to (C, H, W)
        image = image.permute(2, 0, 1)
        
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Trainer:
    """Training manager"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        create_directories(config)
        
        # Set random seed
        set_seed(42)
        
        # Load processed data
        self.load_data()
        
        # Create model
        self.model = ShuffleBiLSTM(config).to(self.device)
        print(f"\nModel parameters: {count_parameters(self.model):,}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def load_data(self):
        """Load processed images and create data loaders"""
        data_path = os.path.join(self.config['paths']['processed_data'], 'processed_images.pkl')
        
        print(f"\nLoading processed data from {data_path}...")
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        images = dataset['images']
        labels = dataset['labels']
        self.class_names = dataset['class_names']
        
        print(f"Loaded {len(images)} samples")
        print(f"Class distribution: {np.bincount(labels)}")
        
        # Create full dataset
        full_dataset = BoringBarDataset(images, labels)
        
        # Split into train and validation
        train_size = int(self.config['data']['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean().item() * 100
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
        
        return losses.avg, accuracies.avg
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).float().mean().item() * 100
                
                # Update meters
                losses.update(loss.item(), images.size(0))
                accuracies.update(accuracy, images.size(0))
                
                # Store predictions
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return losses.avg, accuracies.avg, all_predictions, all_labels
    
    def train(self):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50 + "\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                save_path = os.path.join(self.config['paths']['model_save'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_path)
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{patience}")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*50 + "\n")
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_history(self):
        """Save training history"""
        save_path = os.path.join(self.config['paths']['results'], 'training_history.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.history, f)
        print(f"Training history saved to {save_path}")
    
    def plot_training_curves(self):
        """Plot loss and accuracy curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config['paths']['results'], 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.close()


if __name__ == '__main__':
    config = load_config('config.yaml')
    trainer = Trainer(config)
    trainer.train()
