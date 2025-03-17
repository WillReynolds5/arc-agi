#!/usr/bin/env python
"""
Functions for visualizing ARC tasks and solutions.
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def visualize_all_in_one(task, predicted_grid=None, task_id=None):
    """
    Visualize all inputs and outputs in a single figure.
    
    Args:
        task: ARC task dictionary with 'train' and 'test' examples
        predicted_grid: Predicted output grid (optional)
        task_id: Optional task ID to display
    """
    n_train = len(task['train'])
    
    # Calculate grid size - we need a slot for each train input/output pair,
    # plus the test input and predicted output
    total_items = n_train * 2 + 2  # train pairs + test input + (prediction or true output)
    if predicted_grid is not None:
        total_items += 1  # Add one more for ground truth if we have a prediction
    
    # Find a reasonable grid layout
    cols = min(4, total_items)
    rows = (total_items + cols - 1) // cols
    
    # Create figure
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    title = f"ARC Task {task_id}" if task_id else "ARC Task"
    fig.suptitle(title, fontsize=16)
    
    gs = GridSpec(rows, cols, figure=fig)
    
    # Plot training examples
    item_count = 0
    for i in range(n_train):
        # Plot input
        ax1 = fig.add_subplot(gs[item_count // cols, item_count % cols])
        ax1.imshow(task['train'][i]['input'], cmap='viridis', aspect='equal')  # Force square cells
        ax1.set_title(f"Train {i+1} Input")
        ax1.axis('off')
        item_count += 1
        
        # Plot output
        ax2 = fig.add_subplot(gs[item_count // cols, item_count % cols])
        ax2.imshow(task['train'][i]['output'], cmap='viridis', aspect='equal')  # Force square cells
        ax2.set_title(f"Train {i+1} Output")
        ax2.axis('off')
        item_count += 1
    
    # Plot test input
    ax_test = fig.add_subplot(gs[item_count // cols, item_count % cols])
    ax_test.imshow(task['test'][0]['input'], cmap='viridis', aspect='equal')  # Force square cells
    ax_test.set_title("Test Input")
    ax_test.axis('off')
    item_count += 1
    
    # Plot predicted output if available
    if predicted_grid is not None:
        ax_pred = fig.add_subplot(gs[item_count // cols, item_count % cols])
        ax_pred.imshow(predicted_grid, cmap='viridis', aspect='equal')  # Force square cells
        ax_pred.set_title("Predicted Output")
        ax_pred.axis('off')
        item_count += 1
        
        # Plot ground truth
        ax_true = fig.add_subplot(gs[item_count // cols, item_count % cols])
        ax_true.imshow(task['test'][0]['output'], cmap='viridis', aspect='equal')  # Force square cells
        ax_true.set_title("Ground Truth")
        ax_true.axis('off')
    else:
        # Just show ground truth if no prediction
        ax_true = fig.add_subplot(gs[item_count // cols, item_count % cols])
        ax_true.imshow(task['test'][0]['output'], cmap='viridis', aspect='equal')  # Force square cells
        ax_true.set_title("Test Output (Ground Truth)")
        ax_true.axis('off')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Adjust for the suptitle
    plt.show()
    
    return fig


def visualize_multiple_attempts(task, predicted_grids, metrics, task_id=None, max_attempts_to_show=10):
    """
    Visualize multiple inference attempts in a single figure along with their metrics.
    
    Args:
        task: ARC task dictionary with 'train' and 'test' examples
        predicted_grids: List of predicted output grids
        metrics: List of metric dictionaries corresponding to each prediction
        task_id: Optional task ID to display
        max_attempts_to_show: Maximum number of attempts to display (to avoid overcrowding)
    """
    # Filter out None predictions
    valid_attempts = [(i, grid, m) for i, (grid, m) in enumerate(zip(predicted_grids, metrics)) 
                     if grid is not None]
    
    if not valid_attempts:
        print("No valid predictions to visualize.")
        return
    
    # Limit the number of attempts to show to avoid overcrowding
    if len(valid_attempts) > max_attempts_to_show:
        # Sort by IoU score and take the top max_attempts_to_show
        valid_attempts = sorted(valid_attempts, key=lambda x: x[2].get('mean_iou', 0), reverse=True)[:max_attempts_to_show]
        print(f"Showing only the top {max_attempts_to_show} attempts based on IoU score.")
    
    n_train = len(task['train'])
    n_attempts = len(valid_attempts)
    
    # Calculate grid layout
    # First row: training examples (input-output pairs side by side)
    # Second row: test input and ground truth
    # Remaining rows: predictions (3 per row)
    
    # Determine how many columns we need for training examples
    train_cols = n_train * 2  # Each training example has input and output
    
    # For predictions, arrange in rows of 3
    pred_rows = (n_attempts + 2) // 3  # +2 for test input and ground truth
    pred_cols = min(3, n_attempts + 2)
    
    # Total rows needed
    total_rows = 2 + pred_rows  # 1 for training, 1 for test+truth, and pred_rows for predictions
    
    # Determine overall column count (max of training columns and prediction columns)
    total_cols = max(train_cols, pred_cols)
    
    # Create figure with flexible height based on number of rows
    fig = plt.figure(figsize=(total_cols * 3, total_rows * 3))
    title = f"ARC Task {task_id} - Multiple Inference Attempts" if task_id else "ARC Task - Multiple Inference Attempts"
    fig.suptitle(title, fontsize=16)
    
    gs = GridSpec(total_rows, total_cols, figure=fig)
    
    # Plot training examples (first row)
    for i in range(n_train):
        # Plot input
        ax_in = fig.add_subplot(gs[0, i*2])
        ax_in.imshow(task['train'][i]['input'], cmap='viridis', aspect='equal')  # Force square cells
        ax_in.set_title(f"Train {i+1} Input")
        ax_in.axis('off')
        
        # Plot output
        ax_out = fig.add_subplot(gs[0, i*2+1])
        ax_out.imshow(task['train'][i]['output'], cmap='viridis', aspect='equal')  # Force square cells
        ax_out.set_title(f"Train {i+1} Output")
        ax_out.axis('off')
    
    # Plot test input and ground truth (second row)
    ax_test = fig.add_subplot(gs[1, 0])
    ax_test.imshow(task['test'][0]['input'], cmap='viridis', aspect='equal')  # Force square cells
    ax_test.set_title("Test Input")
    ax_test.axis('off')
    
    ax_truth = fig.add_subplot(gs[1, 1])
    ground_truth = np.array(task['test'][0]['output'])
    ax_truth.imshow(ground_truth, cmap='viridis', aspect='equal')  # Force square cells
    ax_truth.set_title("Ground Truth")
    ax_truth.axis('off')
    
    # Plot predictions (remaining rows)
    for idx, (attempt_idx, grid, metric) in enumerate(valid_attempts):
        row = 2 + (idx // 3)
        col = idx % 3
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(grid, cmap='viridis', aspect='equal')  # Force square cells
        
        # Format metrics for the title
        iou = metric.get('mean_iou', 0)
        exact = '✓' if metric.get('exact_match', False) else '✗'
        
        ax.set_title(f"Attempt #{attempt_idx+1}: IoU={iou:.3f} {exact}")
        ax.axis('off')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Adjust for the suptitle
    plt.show()
    
    return fig 