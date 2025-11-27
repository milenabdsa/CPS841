import pickle
from collections import Counter
import os
import numpy as np

import matplotlib.pyplot as plt

def load_pickle_data(pickle_path):
    """Load data from pickle file"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_class_distribution(data):
    """Plot the distribution of samples per class"""
    # Extract labels from the data
    if isinstance(data, dict) and 'labels' in data:
        labels = data['labels']
    elif isinstance(data, tuple) and len(data) >= 2:
        labels = data[1]  # Assuming (features, labels) format
    else:
        labels = data
    
    # Convert numpy arrays to hashable format if needed
    if len(labels) > 0 and isinstance(labels[0], np.ndarray):
        # Convert each array to a tuple or take the first element if it's a single-value array
        if labels[0].size == 1:
            labels = [label.item() if hasattr(label, 'item') else label[0] for label in labels]
        else:
            labels = [tuple(label) for label in labels]
    
    # Count occurrences of each class
    class_counts = Counter(labels)
    
    # Sort by class name/number
    sorted_classes = sorted(class_counts.items())
    classes = [str(c[0]) for c in sorted_classes]
    counts = [c[1] for c in sorted_classes]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of Samples per Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    # Save the plot
    output_filename = 'class_distribution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    
    plt.show()

if __name__ == '__main__':
    # Specify the path to your pickle file
    pickle_path = 'y26.p'
    
    # Load data
    print(f"Loading data from {pickle_path}...")
    data = load_pickle_data(pickle_path)
    
    # Plot distribution
    print("Generating class distribution plot...")
    plot_class_distribution(data)