#!/usr/bin/env python
"""
Functions for visualizing ARC task data and results.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Any, Tuple

# Define the color map for ARC grids (10 colors)
ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]

def visualize_task_result(
    task: Dict, 
    prediction_output: Optional[np.ndarray] = None, 
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Visualize ARC task with input, expected output, and prediction.
    
    Args:
        task: ARC task dict with train, test pairs
        prediction_output: Optional prediction grid (if available)
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
    """
    # Determine how many rows and columns we need
    num_train = len(task['train'])
    num_test = len(task['test'])
    num_rows = max(num_train, num_test) 
    
    # Create a larger figure with a grid layout
    fig, axes = plt.subplots(
        nrows=num_rows, 
        ncols=3 if prediction_output is not None else 2,
        figsize=(15, 5 * num_rows)
    )
    
    cmap = ListedColormap(ARC_COLORS)
    
    # If there's only one row, we need to adjust the axes shape
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Fill in empty spots in the grid with invisible subplots
    for i in range(num_rows):
        for j in range(3 if prediction_output is not None else 2):
            if i >= num_train and j == 0:  # Empty training input
                axes[i, j].axis('off')
            elif i >= num_test and j >= 1:  # Empty test output or prediction
                axes[i, j].axis('off')
    
    # Plot training examples
    for i, example in enumerate(task['train']):
        if i < num_rows:
            # Plot input grid
            ax = axes[i, 0]
            im = ax.imshow(example['input'], cmap=cmap, vmin=0, vmax=9)
            ax.set_title(f"Train Input {i+1}")
            ax.set_xticks(np.arange(len(example['input'][0])))
            ax.set_yticks(np.arange(len(example['input'])))
            ax.grid(color='white', linestyle='-', linewidth=1.5)
            
            # Add grid values as text
            for y in range(len(example['input'])):
                for x in range(len(example['input'][0])):
                    ax.text(x, y, str(example['input'][y][x]), 
                           ha="center", va="center", color="w", fontweight="bold")
            
            # Plot output grid
            ax = axes[i, 1]
            im = ax.imshow(example['output'], cmap=cmap, vmin=0, vmax=9)
            ax.set_title(f"Train Output {i+1}")
            ax.set_xticks(np.arange(len(example['output'][0])))
            ax.set_yticks(np.arange(len(example['output'])))
            ax.grid(color='white', linestyle='-', linewidth=1.5)
            
            # Add grid values as text
            for y in range(len(example['output'])):
                for x in range(len(example['output'][0])):
                    ax.text(x, y, str(example['output'][y][x]), 
                           ha="center", va="center", color="w", fontweight="bold")
    
    # Plot test examples
    for i, example in enumerate(task['test']):
        if i < num_rows:
            # Plot input grid
            ax = axes[i, 0]
            if i >= num_train:  # Only set if not already set by training examples
                im = ax.imshow(example['input'], cmap=cmap, vmin=0, vmax=9)
                ax.set_title(f"Test Input {i+1}")
                ax.set_xticks(np.arange(len(example['input'][0])))
                ax.set_yticks(np.arange(len(example['input'])))
                ax.grid(color='white', linestyle='-', linewidth=1.5)
                
                # Add grid values as text
                for y in range(len(example['input'])):
                    for x in range(len(example['input'][0])):
                        ax.text(x, y, str(example['input'][y][x]), 
                               ha="center", va="center", color="w", fontweight="bold")
            
            # Plot expected output grid
            ax = axes[i, 1]
            im = ax.imshow(example['output'], cmap=cmap, vmin=0, vmax=9)
            ax.set_title(f"Expected Output {i+1}")
            ax.set_xticks(np.arange(len(example['output'][0])))
            ax.set_yticks(np.arange(len(example['output'])))
            ax.grid(color='white', linestyle='-', linewidth=1.5)
            
            # Add grid values as text
            for y in range(len(example['output'])):
                for x in range(len(example['output'][0])):
                    ax.text(x, y, str(example['output'][y][x]), 
                           ha="center", va="center", color="w", fontweight="bold")
            
            # Plot prediction if provided
            if prediction_output is not None:
                ax = axes[i, 2]
                
                # Only plot prediction for the first test example
                if i == 0:
                    im = ax.imshow(prediction_output, cmap=cmap, vmin=0, vmax=9)
                    ax.set_title(f"Prediction")
                    ax.set_xticks(np.arange(len(prediction_output[0])))
                    ax.set_yticks(np.arange(len(prediction_output)))
                    ax.grid(color='white', linestyle='-', linewidth=1.5)
                    
                    # Add grid values as text
                    for y in range(len(prediction_output)):
                        for x in range(len(prediction_output[0])):
                            ax.text(x, y, str(prediction_output[y][x]), 
                                   ha="center", va="center", color="w", fontweight="bold")
    
    plt.tight_layout()
    
    # Save the visualization if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def parse_grid_from_text(text: str) -> Optional[np.ndarray]:
    """
    Parse a grid from text representation.
    
    Args:
        text: Text containing a grid representation
        
    Returns:
        Numpy array of grid or None if parsing fails
    """
    try:
        # Look for grid-like patterns in the text
        lines = text.strip().split('\n')
        grid_lines = []
        
        # Find contiguous lines that look like grids
        current_grid = []
        for line in lines:
            # Clean up the line and check if it looks like a grid row
            clean_line = line.strip()
            if clean_line and all(c.isdigit() or c in '[] ,' for c in clean_line):
                # Extract just the digits
                digits = [int(c) for c in clean_line if c.isdigit()]
                if digits:
                    current_grid.append(digits)
            elif current_grid:
                # End of a grid section
                if len(current_grid) > 0:
                    grid_lines.append(current_grid)
                current_grid = []
        
        # Add the last grid if it exists
        if current_grid:
            grid_lines.append(current_grid)
        
        # Use the largest grid found
        if grid_lines:
            largest_grid = max(grid_lines, key=lambda g: len(g) * len(g[0]) if g and g[0] else 0)
            
            # Check if all rows have the same length
            if all(len(row) == len(largest_grid[0]) for row in largest_grid):
                return np.array(largest_grid)
    
    except Exception as e:
        print(f"Error parsing grid: {e}")
    
    return None