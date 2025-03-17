#!/usr/bin/env python
"""
Functions for loading and processing ARC data.
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

def load_arc_data(data_path='data', subset=None):
    """
    Load ARC tasks from data directory.
    
    Args:
        data_path: Path to ARC data
        subset: Optional subset ('training' or 'evaluation')
    
    Returns:
        Dictionary of tasks with task IDs as keys
    """
    # Move implementation from utils/data_utils.py
    tasks = {}
    
    if subset:
        data_dirs = [os.path.join(data_path, subset)]
    else:
        data_dirs = [
            os.path.join(data_path, 'training'),
            os.path.join(data_path, 'evaluation')
        ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    task_path = os.path.join(data_dir, filename)
                    task_id = filename.replace('.json', '')
                    
                    with open(task_path, 'r') as f:
                        task_data = json.load(f)
                        tasks[task_id] = task_data
    
    return tasks

def convert_to_chat_messages(task, solution_response, include_extraction=False):
    """
    Convert task and solution into chat message format for training.
    
    Args:
        task: The ARC task
        solution_response: The model's solution text
        include_extraction: Whether to include extraction prompt/response
        
    Returns:
        List of message dictionaries in OpenAI format
    """
    # Move implementation from build_training_data.py
    from arc.prompts import create_arc_solve_prompt
    
    # Create the system message
    system_content = (
        "You are an expert at solving abstract reasoning puzzles. "
        "Given examples of input and output grids, your task is to identify the pattern "
        "and apply it to a new input grid. First explain your reasoning step by step, "
        "then provide the solution grid."
    )
    
    # Create the user message (the task description/prompt)
    user_content = create_arc_solve_prompt(task)
    
    # Create the assistant message (the solution)
    assistant_content = solution_response
    
    # Construct the message list
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    # If we want to include the extraction part as well
    if include_extraction:
        from arc.prompts import create_extraction_prompt
        extraction_prompt = create_extraction_prompt(solution_response)
        # This would require additional logic
    
    return messages 

def load_task(task_id, data_path='data'):
    """
    Load a specific ARC task by ID.
    
    Args:
        task_id: The ID of the task to load
        data_path: Path to the data directory
        
    Returns:
        The task dictionary or None if not found
    """
    tasks = load_arc_data(data_path)
    return tasks.get(task_id) 