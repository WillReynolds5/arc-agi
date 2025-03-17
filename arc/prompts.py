#!/usr/bin/env python
"""
Templates for prompts used in ARC tasks.
"""
import numpy as np

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
    Create prompt for extracting grid from solution text.
    
    Args:
        solution_text: Model's solution text
    
    Returns:
        Extraction prompt string
    """
    prompt = "# Grid Extraction Task\n\n"
    prompt += "Below is a solution to an Abstract Reasoning Challenge. Extract only the final output grid from the solution.\n\n"
    prompt += "## Solution Text\n\n"
    prompt += solution_text
    
    prompt += "\n\n## Instructions\n\n"
    prompt += "1. Identify the final output grid in the solution.\n"
    prompt += "2. Extract ONLY the grid values, maintaining the exact format (spaces between numbers, newlines between rows).\n"
    prompt += "3. Provide ONLY the grid, with no additional text or explanation.\n\n"
    
    return prompt


def extract_grid_from_solution(solution_text, model=None):
    """
    Extract the grid from a solution text, either directly or using an API call.
    
    Args:
        solution_text: The solution text
        model: Optional model name for API extraction
        
    Returns:
        Numpy array representing the grid, or None if extraction fails
    """
    try:
        # First try direct extraction using regex or parsing
        grid = direct_grid_extraction(solution_text)
        if grid is not None:
            return grid
        
        # If direct extraction fails and a model is provided, try API extraction
        if model is not None:
            from arc.inference import run_api_inference
            
            extraction_prompt = create_extraction_prompt(solution_text)
            extracted_text = run_api_inference(extraction_prompt, model)
            return parse_grid_from_text(extracted_text)
        
        return None
    except Exception as e:
        print(f"Grid extraction failed: {e}")
        return None


def direct_grid_extraction(text):
    """
    Try to directly extract a grid from text using pattern matching.
    
    Args:
        text: The text containing grid
        
    Returns:
        Numpy array of the grid or None if extraction fails
    """
    import re
    
    # Look for grid-like patterns in the text
    # This is a simplified approach and might need refinement
    
    # Strategy 1: Look for a sequence of lines with digits/spaces
    grid_pattern = r'(\d+(\s+\d+)+\n)+'
    matches = re.findall(grid_pattern, text)
    
    if matches:
        # Extract the longest match which is likely the grid
        longest_match = max([m[0] for m in matches], key=len)
        return parse_grid_from_text(longest_match)
    
    # Strategy 2: Look for grid inside markdown code blocks or other common formats
    grid_in_block = re.search(r'```\n([\d\s]+)\n```', text)
    if grid_in_block:
        return parse_grid_from_text(grid_in_block.group(1))
    
    # Strategy 3: Find a section that says "output" or "final grid" and extract text after it
    output_section = re.search(r'(output:|final grid:|solution grid:)(.*?)(\n\n|$)', 
                               text, re.IGNORECASE | re.DOTALL)
    if output_section:
        return parse_grid_from_text(output_section.group(2))
    
    # If all else fails, return None
    return None


def parse_grid_from_text(text):
    """
    Parse a grid from text representation.
    
    Args:
        text: Text representation of grid
        
    Returns:
        Numpy array of the grid
    """
    # Clean the text
    text = text.strip()
    
    # Split into lines and parse each line
    lines = text.split('\n')
    grid = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Process line: split by whitespace and convert to integers
        try:
            row = [int(cell) for cell in line.split()]
            if row:  # Only add non-empty rows
                grid.append(row)
        except ValueError:
            # Skip lines that can't be parsed as integers
            continue
    
    # Convert to numpy array if we have a valid grid
    if grid and all(len(row) == len(grid[0]) for row in grid):
        return np.array(grid)
    
    return None 