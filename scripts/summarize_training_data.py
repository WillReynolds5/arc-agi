#!/usr/bin/env python
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def analyze_training_data(training_dir):
    """
    Analyze training data files and produce summary statistics.
    
    Args:
        training_dir: Directory containing training data files
    
    Returns:
        Dictionary with summary statistics
    """
    # Get all JSON files in the directory
    files = [f for f in os.listdir(training_dir) if f.endswith('.json') and not f.startswith('results_summary')]
    
    if not files:
        print(f"No training data files found in {training_dir}")
        return None
    
    print(f"Found {len(files)} training examples")
    
    # Extract task IDs and metrics from filenames
    tasks = {}
    iou_scores = []
    exact_matches = 0
    
    for filename in files:
        # Parse filename to extract info
        # Example format: task_id_exact/partial_iouX.XXXX_timestamp_idx.json
        parts = filename.split('_')
        if len(parts) >= 4:
            task_id = parts[0]
            match_type = parts[1]
            iou_str = parts[2].replace('iou', '')
            
            try:
                iou = float(iou_str)
                iou_scores.append(iou)
                
                if match_type == 'exact':
                    exact_matches += 1
                
                # Track tasks
                if task_id not in tasks:
                    tasks[task_id] = {
                        'count': 0,
                        'exact_matches': 0,
                        'iou_scores': []
                    }
                
                tasks[task_id]['count'] += 1
                tasks[task_id]['iou_scores'].append(iou)
                
                if match_type == 'exact':
                    tasks[task_id]['exact_matches'] += 1
            except:
                print(f"Couldn't parse IoU from filename: {filename}")
    
    # Calculate statistics
    stats = {
        'total_examples': len(files),
        'unique_tasks': len(tasks),
        'exact_matches': exact_matches,
        'partial_matches': len(files) - exact_matches,
        'average_iou': np.mean(iou_scores) if iou_scores else 0,
        'median_iou': np.median(iou_scores) if iou_scores else 0,
        'tasks': tasks
    }
    
    return stats


def plot_statistics(stats):
    """
    Plot visualizations of the training data statistics.
    
    Args:
        stats: Dictionary of statistics from analyze_training_data
    """
    # Plot distribution of IoU scores
    plt.figure(figsize=(12, 8))
    
    # Extract IoU scores from all examples
    all_scores = []
    for task_id, task_info in stats['tasks'].items():
        all_scores.extend(task_info['iou_scores'])
    
    plt.subplot(2, 2, 1)
    plt.hist(all_scores, bins=20, alpha=0.7)
    plt.axvline(stats['average_iou'], color='red', linestyle='dashed', linewidth=2)
    plt.title('Distribution of IoU Scores')
    plt.xlabel('IoU Score')
    plt.ylabel('Count')
    
    # Plot examples per task
    plt.subplot(2, 2, 2)
    examples_per_task = [task_info['count'] for task_id, task_info in stats['tasks'].items()]
    plt.hist(examples_per_task, bins=range(1, max(examples_per_task) + 2), alpha=0.7)
    plt.title('Examples per Task')
    plt.xlabel('Number of Examples')
    plt.ylabel('Number of Tasks')
    
    # Plot exact vs partial matches
    plt.subplot(2, 2, 3)
    match_types = ['Exact Matches', 'Partial Matches']
    counts = [stats['exact_matches'], stats['partial_matches']]
    plt.bar(match_types, counts, alpha=0.7)
    plt.title('Exact vs Partial Matches')
    plt.ylabel('Count')
    
    # Plot tasks by average IoU
    plt.subplot(2, 2, 4)
    task_avg_iou = []
    for task_id, task_info in stats['tasks'].items():
        if task_info['iou_scores']:
            task_avg_iou.append(np.mean(task_info['iou_scores']))
    
    plt.hist(task_avg_iou, bins=10, alpha=0.7)
    plt.title('Distribution of Average Task IoU')
    plt.xlabel('Average IoU per Task')
    plt.ylabel('Number of Tasks')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze training data statistics')
    parser.add_argument('--training_dir', default='training_data', help='Directory containing training data')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    stats = analyze_training_data(args.training_dir)
    
    if stats:
        print("\nTraining Data Statistics:")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Unique tasks: {stats['unique_tasks']}")
        print(f"Exact matches: {stats['exact_matches']} ({stats['exact_matches']/stats['total_examples']*100:.1f}%)")
        print(f"Partial matches: {stats['partial_matches']} ({stats['partial_matches']/stats['total_examples']*100:.1f}%)")
        print(f"Average IoU: {stats['average_iou']:.4f}")
        print(f"Median IoU: {stats['median_iou']:.4f}")
        
        if args.plot:
            plot_statistics(stats)


if __name__ == "__main__":
    main() 