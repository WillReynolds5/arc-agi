#!/usr/bin/env python
"""
Functions for evaluating model performance on ARC tasks.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

def intersection_over_union(y_true, y_pred, class_val):
    """
    Calculate Intersection over Union for a specific class value.
    
    Args:
        y_true: Ground truth grid (2D numpy array)
        y_pred: Predicted grid (2D numpy array)
        class_val: The class value to calculate IoU for
        
    Returns:
        IoU score for the specified class (float between 0 and 1)
    """
    # Create binary masks
    true_mask = (y_true == class_val)
    pred_mask = (y_pred == class_val)
    
    # Calculate intersection and union
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    
    # Calculate IoU (handle division by zero)
    if union == 0:
        # If the class doesn't appear in either ground truth or prediction
        if intersection == 0:
            return 1.0  # Both agree this class is absent
        else:
            return 0.0  # This should not happen mathematically
    
    iou = intersection / union
    return float(iou)

def mean_iou(actual, predicted):
    """
    Calculate mean Intersection over Union (IoU) between actual and predicted grids.
    
    Args:
        actual: Ground truth grid (numpy array)
        predicted: Predicted grid (numpy array)
        
    Returns:
        Mean IoU score (0-1 scale, higher is better)
    """
    # Check for dimension mismatch and return 0 if dimensions don't match
    if actual.shape != predicted.shape:
        return 0.0
    
    # Get unique values across both grids
    all_values = np.unique(np.concatenate((actual.flatten(), predicted.flatten())))
    
    # Calculate IoU for each unique value
    iou_scores = []
    for val in all_values:
        # Skip background (0s) for IoU calculation
        if val == 0:
            continue
            
        # Create binary masks for this value
        mask_actual = (actual == val)
        mask_predicted = (predicted == val)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask_actual, mask_predicted).sum()
        union = np.logical_or(mask_actual, mask_predicted).sum()
        
        # Calculate IoU for this value
        if union > 0:
            iou = intersection / union
            iou_scores.append(iou)
    
    # Calculate mean IoU across all values
    if len(iou_scores) > 0:
        return np.mean(iou_scores)
    else:
        return 0.0  # If no non-background values matched

def evaluate_solution(task, solution_text):
    """
    Evaluate a solution against the task's output grid.
    
    Args:
        task: ARC task
        solution_text: Model's solution text
    
    Returns:
        Dict with evaluation metrics
    """
    from arc.prompts import extract_grid_from_solution
    
    # Ground truth for comparison
    ground_truth = np.array(task['test'][0]['output'])
    
    # Try to extract the predicted grid from the solution text
    try:
        predicted_grid = extract_grid_from_solution(solution_text)
        
        if predicted_grid is not None:
            # Compute metrics
            return evaluate_prediction(ground_truth, predicted_grid)
        else:
            # Handle failed grid extraction
            return {
                'exact_match': False,
                'shape_match': False,
                'mean_iou': 0.0,
                'accuracy': 0.0,
                'extraction_failed': True
            }
    except Exception as e:
        # Handle any errors
        return {
            'exact_match': False,
            'shape_match': False,
            'mean_iou': 0.0,
            'accuracy': 0.0,
            'inference_failed': True,
            'error': str(e)
        }

def evaluate_prediction(actual, predicted):
    """
    Evaluate prediction with multiple metrics.
    
    Args:
        actual: Ground truth grid (numpy array)
        predicted: Predicted grid (numpy array)
        
    Returns:
        Dictionary of metrics
    """
    # Check for dimension mismatch
    shape_match = (actual.shape == predicted.shape)
    
    # If shapes don't match, we can't do exact match
    exact_match = False
    accuracy = 0.0
    miou = 0.0
    
    # Only compute detailed metrics if shapes match
    if shape_match:
        # Calculate exact match
        exact_match = np.array_equal(actual, predicted)
        
        # Calculate cell-wise accuracy
        total_cells = actual.size
        correct_cells = (actual == predicted).sum()
        accuracy = correct_cells / total_cells
        
        # Calculate mean IoU
        miou = mean_iou(actual, predicted)
    
    return {
        'exact_match': exact_match,
        'shape_match': shape_match,
        'mean_iou': miou,
        'accuracy': accuracy
    }

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