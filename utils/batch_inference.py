import os
import numpy as np
from tqdm import tqdm

from utils.data_utils import load_arc_data
from prompts import create_arc_prompt, execute_arc_prompt, extract_grid_from_solution
from utils.cloud_inference import inference
from utils.metrics import evaluate_prediction, mean_iou

# Backward compatibility layer
from arc.inference import run_multiple_inferences
from arc.evaluation import analyze_results
from arc.data import load_task

# Keep any other functions for backward compatibility

def run_multiple_inferences(task, n_attempts=5, model="google/gemini-2.0-flash-001", verbose=True):
    """
    Run inference multiple times on a single ARC task and compute metrics.
    
    Args:
        task: The ARC task to solve
        n_attempts: Number of inference attempts to make
        model: Model name to use for inference
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing results, metrics, and analyzed data
    """
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Create the base prompt
    prompt = create_arc_prompt(task)
    
    # Prepare to store results
    results = {
        'prompts': [],
        'responses': [],
        'predicted_grids': [],
        'metrics': [],
        'task': task
    }
    
    # Ground truth for comparison
    ground_truth = np.array(task['test'][0]['output'])
    
    # Run multiple inferences
    if verbose:
        print(f"Running {n_attempts} inference attempts...")
    
    for i in tqdm(range(n_attempts), disable=not verbose):
        # For now, we use the same prompt each time, but this could be varied
        current_prompt = prompt
        results['prompts'].append(current_prompt)
        
        try:
            # First stage: Get solution
            solution = execute_arc_prompt(task, model)
            results['responses'].append(solution)
            
            # Second stage: Extract grid
            predicted_grid = extract_grid_from_solution(solution, model)
            
            if predicted_grid is not None:
                results['predicted_grids'].append(predicted_grid)
                
                # Compute metrics
                metrics = evaluate_prediction(ground_truth, predicted_grid)
                results['metrics'].append(metrics)
            else:
                # Handle failed grid extraction
                if verbose:
                    print(f"Warning: Failed to extract grid from attempt {i+1}")
                # Add None for the grid and empty metrics
                results['predicted_grids'].append(None)
                results['metrics'].append({
                    'exact_match': False,
                    'shape_match': False,
                    'mean_iou': 0.0,
                    'accuracy': 0.0,
                    'extraction_failed': True
                })
        
        except Exception as e:
            if verbose:
                print(f"Error in attempt {i+1}: {e}")
            # Add None for response and grid, and empty metrics
            results['responses'].append(None)
            results['predicted_grids'].append(None)
            results['metrics'].append({
                'exact_match': False,
                'shape_match': False,
                'mean_iou': 0.0,
                'accuracy': 0.0,
                'inference_failed': True
            })
    
    # Analyze results
    analyzed_data = analyze_results(results, verbose)
    results['analysis'] = analyzed_data
    
    return results


def analyze_results(results, verbose=True):
    """
    Analyze the results of multiple inference attempts.
    
    Args:
        results: Dictionary of results from run_multiple_inferences
        verbose: Whether to print analysis
        
    Returns:
        Dictionary of analyzed data
    """
    metrics = results['metrics']
    
    # Extract IoU scores and filter out None values
    iou_scores = [m.get('mean_iou', 0.0) for m in metrics if m is not None]
    
    if not iou_scores:
        if verbose:
            print("No valid IoU scores found.")
        return {
            'average_iou': 0.0,
            'max_iou': 0.0,
            'min_iou': 0.0,
            'above_average_indices': [],
            'best_attempt_index': None,
            'success_rate': 0.0
        }
    
    # Calculate statistics
    average_iou = np.mean(iou_scores)
    max_iou = np.max(iou_scores)
    min_iou = np.min(iou_scores)
    
    # Find attempts above average
    above_average_indices = [i for i, score in enumerate(iou_scores) if score > average_iou]
    
    # Find the best attempt
    best_attempt_index = np.argmax(iou_scores)
    
    # Calculate success rate (exact matches)
    exact_matches = sum(1 for m in metrics if m.get('exact_match', False))
    success_rate = exact_matches / len(metrics) if metrics else 0
    
    if verbose:
        print("\nResults Analysis:")
        print(f"Average IoU Score: {average_iou:.4f}")
        print(f"Max IoU Score: {max_iou:.4f} (Attempt #{best_attempt_index + 1})")
        print(f"Min IoU Score: {min_iou:.4f}")
        print(f"Success Rate (Exact Matches): {success_rate:.2%}")
        print(f"Attempts Above Average: {len(above_average_indices)}/{len(iou_scores)}")
    
    return {
        'average_iou': average_iou,
        'max_iou': max_iou,
        'min_iou': min_iou,
        'above_average_indices': above_average_indices,
        'best_attempt_index': best_attempt_index,
        'success_rate': success_rate
    }


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