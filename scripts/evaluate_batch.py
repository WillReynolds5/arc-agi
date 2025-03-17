#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from dotenv import load_dotenv

# Add the root directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()
from utils.visualization import visualize_all_in_one, visualize_multiple_attempts
from utils.batch_inference import run_multiple_inferences, load_task


def plot_score_distribution(results):
    """
    Plot the distribution of IoU scores from multiple inference attempts.
    
    Args:
        results: Dictionary of results from run_multiple_inferences
    """
    metrics = results['metrics']
    
    # Extract IoU scores
    iou_scores = [m.get('mean_iou', 0.0) for m in metrics if m is not None]
    
    if not iou_scores:
        print("No valid IoU scores to plot.")
        return
    
    avg_iou = np.mean(iou_scores)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Score distribution histogram
    plt.hist(iou_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add average line
    plt.axvline(avg_iou, color='red', linestyle='dashed', linewidth=2, label=f'Average: {avg_iou:.4f}')
    
    plt.title('Distribution of IoU Scores Across Inference Attempts')
    plt.xlabel('Mean IoU Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def display_best_and_above_average(results):
    """
    Display the best attempt and all attempts above average.
    
    Args:
        results: Dictionary of results from run_multiple_inferences
    """
    analysis = results['analysis']
    task = results['task']
    
    # Display the best attempt
    best_idx = analysis['best_attempt_index']
    if best_idx is not None:
        best_grid = results['predicted_grids'][best_idx]
        best_metrics = results['metrics'][best_idx]
        
        print(f"\nBest Attempt (#{best_idx + 1}):")
        print(f"IoU Score: {best_metrics['mean_iou']:.4f}")
        print(f"Exact Match: {'Yes' if best_metrics['exact_match'] else 'No'}")
        
        visualize_all_in_one(task, best_grid, task_id)
    
    # List all above-average attempts
    print("\nAbove Average Attempts:")
    for idx in analysis['above_average_indices']:
        grid = results['predicted_grids'][idx]
        metrics = results['metrics'][idx]
        print(f"Attempt #{idx + 1}: IoU = {metrics['mean_iou']:.4f}")


def display_all_attempts(results):
    """
    Display all inference attempts in a single visualization.
    
    Args:
        results: Dictionary of results from run_multiple_inferences
    """
    task = results['task']
    predicted_grids = results['predicted_grids']
    metrics = results['metrics']
    
    # Get task_id if available
    task_id = None
    if hasattr(task, 'get'):
        task_id = task.get('id', None)
    
    # Visualize all attempts
    visualize_multiple_attempts(task, predicted_grids, metrics, task_id)


def main():
    parser = argparse.ArgumentParser(description='Run multiple inferences on ARC tasks')
    parser.add_argument('task_id', help='The ARC task ID to solve')
    parser.add_argument('-n', '--num_attempts', type=int, default=5, help='Number of inference attempts')
    parser.add_argument('-m', '--model', default="google/gemini-2.0-flash-001", help='Model to use for inference')
    args = parser.parse_args()
    
    # Load task
    task_id = args.task_id
    task = load_task(task_id)
    
    if task is None:
        print(f"Task ID {task_id} not found in dataset.")
        return
    
    print(f"Task {task_id} loaded successfully.")
    
    # Run multiple inferences
    results = run_multiple_inferences(task, args.num_attempts, args.model)
    
    # Plot score distribution
    plot_score_distribution(results)
    
    # Display all attempts in one visualization
    display_all_attempts(results)
    
    # Display best and above average attempts
    display_best_and_above_average(results)


if __name__ == "__main__":
    main() 