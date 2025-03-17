#!/usr/bin/env python
"""
Templates for prompts used in ARC tasks.
"""
import numpy as np
import logging

def create_arc_solve_prompt(task):
    """
    Create prompt for solving an ARC task.
    
    Args:
        task: ARC task data
    
    Returns:
        Formatted prompt string
    """
    prompt = "# Abstract Reasoning Challenge\n\n"
    prompt += "You will be given examples of input-output grid transformations. Your task is to infer the pattern and apply it to a new input.\n\n"
    
    # Add training examples
    for i, example in enumerate(task['train']):
        prompt += f"## Training Example {i+1}\n\n"
        prompt += "Input:\n"
        prompt += format_grid(example['input'])
        prompt += "\n\nOutput:\n"
        prompt += format_grid(example['output'])
        prompt += "\n\n"
    
    # Add test input
    prompt += "## Test\n\n"
    prompt += "Input:\n"
    prompt += format_grid(task['test'][0]['input'])
    
    prompt += "\n\n## Your Task\n\n"
    prompt += "1. Study the training examples carefully.\n"
    prompt += "2. Identify the pattern or transformation rule.\n"
    prompt += "3. Apply the same rule to the test input.\n"
    prompt += "4. Explain your reasoning step by step.\n"
    prompt += "5. Provide the output grid in the exact same format as the training examples.\n\n"
    
    prompt += "First, explain your reasoning. Then, provide your final answer in a format that looks exactly like the training examples.\n"
    
    return prompt


def format_grid(grid):
    """Format a grid as a string with spaces between cells"""
    return '\n'.join(' '.join(str(cell) for cell in row) for row in grid)


def create_extraction_prompt(solution_text):
    """
    Create a prompt for grid extraction from model solution.
    
    Args:
        solution_text: The solution text from the model
        
    Returns:
        Prompt string for extraction
    """
    return f"""
TASK: Extract the numeric grid from the following AI solution to an ARC puzzle.
Format the grid as a Python nested list, like [[0,1,2],[3,4,5]].
Only return the grid in proper Python list format, nothing else.

AI SOLUTION:
{solution_text}

EXTRACTED GRID:
"""

def direct_grid_extraction(text):
    """
    Extract grid using direct parsing techniques.
    
    Args:
        text: Text containing grid representation
        
    Returns:
        Numpy array or None if extraction fails
    """
    from arc.visualization import parse_grid_from_text
    import re
    import numpy as np
    
    logger = logging.getLogger('arc.prompts')
    
    try:
        # Try standard parsing first
        grid = parse_grid_from_text(text)
        if grid is not None:
            logger.info("Successfully extracted grid using standard parser")
            return grid
            
        # Try to find arrays in the format [[0,0,0],[1,1,1]]
        array_pattern = r'\[\s*\[(?:\s*\d+\s*,\s*)*\s*\d+\s*\]\s*(?:,\s*\[\s*(?:\d+\s*,\s*)*\d+\s*\]\s*)*\]'
        array_matches = re.findall(array_pattern, text)
        
        for match in array_matches:
            try:
                # Replace single quotes with double quotes for JSON parsing
                json_str = match.replace("'", '"')
                import json
                grid_data = json.loads(json_str)
                
                # Validate it's a proper grid (list of lists of same length)
                if (isinstance(grid_data, list) and 
                    all(isinstance(row, list) for row in grid_data) and
                    all(len(row) == len(grid_data[0]) for row in grid_data)):
                    logger.info("Successfully extracted grid using JSON parsing")
                    return np.array(grid_data)
            except Exception as e:
                logger.debug(f"Failed to parse array match: {e}")
                continue
                
        logger.warning("Direct grid extraction failed - no valid grid found")
        return None
        
    except Exception as e:
        logger.error(f"Error in direct_grid_extraction: {e}")
        return None

def extract_grid_from_solution(solution_text, model=None):
    """
    Extract the grid from a solution text, either directly or using an API call.
    
    Args:
        solution_text: The solution text
        model: Optional model name for API extraction
        
    Returns:
        Numpy array representing the grid, or None if extraction fails
    """
    logger = logging.getLogger('arc.prompts')
    
    try:
        logger.info("Attempting grid extraction from solution")
        
        # First try direct extraction using regex or parsing
        logger.info("Trying direct grid extraction...")
        grid = direct_grid_extraction(solution_text)
        if grid is not None:
            return grid
        
        # If direct extraction fails and a model is provided, try API extraction
        if model is not None:
            logger.info(f"Direct extraction failed. Trying API extraction with model {model}...")
            from arc.inference import run_api_inference
            
            extraction_prompt = create_extraction_prompt(solution_text)
            extracted_text = run_api_inference(extraction_prompt, model)
            logger.info(f"API extraction result: {extracted_text[:100]}...")
            
            # Try direct extraction again with the API result
            grid = direct_grid_extraction(extracted_text)
            if grid is not None:
                logger.info("Successfully extracted grid using API")
                return grid
            
            logger.warning("API extraction failed to produce a valid grid")
        else:
            logger.warning("No model provided for API extraction fallback")
        
        return None
    except Exception as e:
        logger.error(f"Grid extraction failed: {e}", exc_info=True)
        return None 