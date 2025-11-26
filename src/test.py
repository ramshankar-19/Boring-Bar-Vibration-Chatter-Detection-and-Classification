"""
Evaluate model on fresh test data
"""

import torch
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from model import ShuffleBiLSTM
from train import BoringBarDataset
from utils import load_config
from torch.utils.data import DataLoader


def evaluate_on_test_data():
    """Evaluate trained model on new test dataset"""
    
    config = load_config('config.yaml')
    device = torch.device('cpu')
    
    # Load trained model
    print("Loading trained model...")
    model = ShuffleBiLSTM(config).to(device)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model training accuracy: {checkpoint['val_acc']:.2f}%\n")
    
    # Load new test data
    print("Loading new test data...")
    test_data_path = 'data/test_processed/processed_images.pkl'
    
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    
    images = test_dataset['images']
    labels = test_dataset['labels']
    class_names = test_dataset['class_names']
    
    print(f"Test samples: {len(images)}")
    print(f"Class distribution: {np.bincount(labels)}\n")
    
    # Create dataset and loader
    test_dataset = BoringBarDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    print("Evaluating on new test data...\n")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    
    print("="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy on New Data: {accuracy:.2f}%\n")
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_acc = (np.array(all_predictions)[class_mask] == i).mean() * 100
            class_count = class_mask.sum()
            correct_count = (np.array(all_predictions)[class_mask] == i).sum()
            print(f"  {class_name:12s}: {class_acc:6.2f}% ({correct_count}/{class_count})")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                               target_names=class_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names,
               yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Test Set Confusion Matrix\nAccuracy: {accuracy:.2f}%', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'results/test_set_confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to {save_path}")
    plt.close()
    
    # Probability distribution analysis
    analyze_confidence(all_labels, all_predictions, all_probabilities, class_names)
    
    return accuracy, all_labels, all_predictions


def analyze_confidence(true_labels, predictions, probabilities, class_names):
    """Analyze prediction confidence"""
    
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Get max confidence for each prediction
    max_confidences = probabilities.max(axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = true_labels == predictions
    correct_confidences = max_confidences[correct_mask]
    incorrect_confidences = max_confidences[~correct_mask]
    
    print("\nConfidence Analysis:")
    print(f"  Correct predictions: {len(correct_confidences)} samples")
    print(f"    Mean confidence: {correct_confidences.mean()*100:.2f}%")
    print(f"    Min confidence:  {correct_confidences.min()*100:.2f}%")
    
    if len(incorrect_confidences) > 0:
        print(f"\n  Incorrect predictions: {len(incorrect_confidences)} samples")
        print(f"    Mean confidence: {incorrect_confidences.mean()*100:.2f}%")
        print(f"    Max confidence:  {incorrect_confidences.max()*100:.2f}%")
    else:
        print(f"\n  Incorrect predictions: 0 samples (Perfect classification!)")
    
    # Plot confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence histogram
    axes[0].hist(correct_confidences, bins=20, alpha=0.7, color='green', 
                label=f'Correct ({len(correct_confidences)})', edgecolor='black')
    if len(incorrect_confidences) > 0:
        axes[0].hist(incorrect_confidences, bins=20, alpha=0.7, color='red', 
                    label=f'Incorrect ({len(incorrect_confidences)})', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Per-class confidence
    class_confidences = []
    for i, class_name in enumerate(class_names):
        class_mask = (true_labels == i) & (predictions == i)
        if class_mask.sum() > 0:
            class_confidences.append(max_confidences[class_mask])
        else:
            class_confidences.append([])
    
    axes[1].boxplot(class_confidences, labels=class_names, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Confidence Score', fontsize=12)
    axes[1].set_title('Confidence by Class (Correct Predictions)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = 'results/test_confidence_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confidence analysis saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    evaluate_on_test_data()
