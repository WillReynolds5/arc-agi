import json
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Backward compatibility layer
from arc.data import load_arc_data, convert_to_chat_messages

def create_arc_dataframe(tasks):
    """
    Convert ARC tasks dictionary to a pandas DataFrame.
    
    Args:
        tasks: Dictionary of ARC tasks
        
    Returns:
        DataFrame with task information
    """
    records = []
    
    for task_id, task in tasks.items():
        record = {
            'task_id': task_id,
            'train': task['train'],
            'test': task['test']
        }
        records.append(record)
    
    return pd.DataFrame(records)

def grid_to_string(grid):
    """
    Convert a grid (2D array) to a string representation.
    
    Args:
        grid: 2D numpy array or list of lists
        
    Returns:
        String representation of the grid
    """
    result = []
    for row in grid:
        result.append(' '.join(str(cell) for cell in row))
    return '\n'.join(result)

# Additional utility functions can be added as needed 