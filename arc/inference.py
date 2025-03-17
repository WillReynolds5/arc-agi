#!/usr/bin/env python
"""
Functions for running inference with models on ARC tasks.
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Sync client for non-async operations
try:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
except Exception:
    openrouter_client = None

def run_model_inference(
    task: Dict[str, Any],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Run inference on a local model.
    
    Args:
        task: The ARC task data
        model: The loaded model for inference
        tokenizer: The tokenizer for the model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top p for nucleus sampling
    
    Returns:
        Generated text response
    """
    from arc.prompts import create_arc_solve_prompt
    
    # Create system and user prompts
    system_content = (
        "You are an expert at solving abstract reasoning puzzles. "
        "Given examples of input and output grids, your task is to identify the pattern "
        "and apply it to a new input grid. First explain your reasoning step by step, "
        "then provide the solution grid."
    )
    
    user_content = create_arc_solve_prompt(task)
    
    # Combine into a single prompt
    prompt = f"{system_content}\n\n{user_content}"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text


def run_api_inference(prompt: str, model: str = "google/gemini-2.0-flash-001"):
    """
    Run inference using OpenRouter API.
    
    Args:
        prompt: The prompt to send to the API
        model: Model name to use for inference
        
    Returns:
        Text response from the API
    """
    if openrouter_client is None:
        raise ValueError("OpenRouter client not initialized. Set OPENROUTER_API_KEY environment variable or use local models only.")
    
    try:
        completion = openrouter_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        response_content = completion.choices[0].message.content
        return response_content
        
    except Exception as e:
        print(f"API inference error: {str(e)}")
        # Return fallback response that won't break downstream processing
        return "I apologize, but I couldn't solve this puzzle due to an API error."


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
    from arc.prompts import create_arc_solve_prompt, extract_grid_from_solution
    from arc.evaluation import evaluate_prediction, analyze_results
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Create the base prompt
    prompt = create_arc_solve_prompt(task)
    
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
            solution = run_api_inference(current_prompt, model)
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


class LlamaInference:
    def __init__(self, model_path):
        """
        Initialize the Llama inference engine.
        
        Args:
            model_path: Path to the Llama model
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.1):
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part (remove the prompt)
        generated_text = response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        return generated_text


def run_inference_on_tasks(model, tasks, prompt_creator, output_dir=None):
    """
    Run inference on a set of ARC tasks.
    
    Args:
        model: LlamaInference instance
        tasks: Dictionary of ARC tasks
        prompt_creator: Function to create prompts from tasks
        output_dir: Directory to save results (optional)
        
    Returns:
        Dictionary of task IDs and model responses
    """
    from tqdm.auto import tqdm
    import os
    
    results = {}
    
    for task_id, task in tqdm(tasks.items(), desc="Processing tasks"):
        prompt = prompt_creator(task)
        response = model.generate(prompt)
        results[task_id] = response
        
        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{task_id}.txt"), "w") as f:
                f.write(response)
    
    return results 