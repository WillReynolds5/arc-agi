# Backward compatibility layer
from arc.evaluation import (
    intersection_over_union,
    mean_iou,
    evaluate_prediction
)

import numpy as np

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