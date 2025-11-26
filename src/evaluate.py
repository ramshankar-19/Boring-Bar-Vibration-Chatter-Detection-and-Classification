"""
Evaluation Script: Test model and visualize results
"""

import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader

from model import ShuffleBiLSTM
from train import BoringBarDataset
from utils import load_config


class Evaluator:
    """Model evaluation and visualization"""
    
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device('cpu')
        
        # Load model
        self.model = ShuffleBiLSTM(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        # Load test data
        self.load_test_data()
    
    def load_test_data(self):
        """Load processed test data"""
        data_path = os.path.join(self.config['paths']['processed_data'], 'processed_images.pkl')
        
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        images = dataset['images']
        labels = dataset['labels']
        self.class_names = dataset['class_names']
        
        # Use same split as training (only validation portion)
        from torch.utils.data import random_split
        
        full_dataset = BoringBarDataset(images, labels)
        train_size = int(self.config['data']['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        _, test_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        print(f"Test samples: {len(test_dataset)}")
    
    def evaluate(self):
        """Evaluate model on test set"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        
        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        for i, class_name in enumerate(self.class_names):
            class_mask = np.array(all_labels) == i
            if class_mask.sum() > 0:
                class_acc = (np.array(all_predictions)[class_mask] == i).mean() * 100
                class_count = class_mask.sum()
                correct_count = (np.array(all_predictions)[class_mask] == i).sum()
                print(f"  {class_name}: {class_acc:.2f}% ({correct_count}/{class_count})")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                   target_names=self.class_names))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        # Plot sample predictions
        self.plot_sample_predictions(all_labels, all_predictions, all_probabilities)
        
        return accuracy, all_labels, all_predictions
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.config['paths']['results'], 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
    
    def plot_sample_predictions(self, true_labels, predictions, probabilities):
        """Plot sample predictions with confidence"""
        # Get indices for correct and incorrect predictions
        correct_idx = np.where(np.array(true_labels) == np.array(predictions))[0]
        incorrect_idx = np.where(np.array(true_labels) != np.array(predictions))[0]
        
        # Sample a few examples
        num_samples = min(6, len(incorrect_idx))
        
        if num_samples > 0:
            sample_idx = np.random.choice(incorrect_idx, size=num_samples, replace=False)
        else:
            sample_idx = np.random.choice(correct_idx, size=6, replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_idx):
            true_label = true_labels[idx]
            pred_label = predictions[idx]
            probs = probabilities[idx]
            
            # Create bar plot of probabilities
            axes[i].bar(range(len(self.class_names)), probs, color='steelblue', alpha=0.7)
            axes[i].set_xticks(range(len(self.class_names)))
            axes[i].set_xticklabels(self.class_names, rotation=45)
            axes[i].set_ylabel('Probability')
            axes[i].set_ylim([0, 1])
            
            # Highlight true and predicted
            axes[i].axvline(true_label, color='green', linestyle='--', linewidth=2, label='True')
            axes[i].axvline(pred_label, color='red', linestyle='--', linewidth=2, label='Predicted')
            
            title_color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f"True: {self.class_names[true_label]}\n"
                            f"Pred: {self.class_names[pred_label]} ({probs[pred_label]:.2f})",
                            color=title_color, fontweight='bold')
            axes[i].legend()
            axes[i].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.config['paths']['results'], 'sample_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
        plt.close()


if __name__ == '__main__':
    config = load_config('config.yaml')
    model_path = os.path.join(config['paths']['model_save'], 'best_model.pth')
    
    evaluator = Evaluator(config, model_path)
    evaluator.evaluate()
