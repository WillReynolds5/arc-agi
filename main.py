#!/usr/bin/env python
"""
Main script for iterative training of ARC solver models.
"""
import os
import sys
import json
import argparse
import torch
from datetime import datetime
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType

# For Gemma 3 models
try:
    from transformers import Gemma3ForCausalLM
    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False
    print("Warning: Gemma3ForCausalLM not available. Install with: pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3")

# Import our modules
from arc.data import load_arc_data, convert_to_chat_messages
from arc.inference import run_model_inference
from arc.training import train_model
from arc.evaluation import evaluate_solution, configure_logging
from arc.visualization import visualize_task_result


def build_dataset_with_model(
    model,
    tokenizer,
    data_path='data',
    output_dir='training_data',
    attempts_per_task=3,
    min_iou=0.7,
    success_only=False,
    limit=None,
    iteration=0,
    stop_after_success=False
):
    """
    Build a dataset using a model for inference.
    
    Args:
        model: The model to use for inference
        tokenizer: The tokenizer for the model
        data_path: Path to ARC data
        output_dir: Directory to save training data
        attempts_per_task: Number of attempts per task
        min_iou: Minimum IoU score required
        success_only: Whether to include only perfect solutions
        limit: Limit processing to this many tasks
        iteration: Current iteration number
        stop_after_success: Whether to stop after finding a perfect solution
        
    Returns:
        Path to the generated dataset and Dataset object
    """
    # Create iteration-specific output directory
    iter_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iter_output_dir, exist_ok=True)
    
    # Load all tasks
    tasks = load_arc_data(data_path)
    print(f"Loaded {len(tasks)} tasks from {data_path}")
    
    # Limit tasks if specified
    if limit is not None:
        task_ids = list(tasks.keys())[:limit]
        tasks = {task_id: tasks[task_id] for task_id in task_ids}
        print(f"Limited to {len(tasks)} tasks for testing")
    
    # Prepare results tracking
    results_summary = {
        'iteration': iteration,
        'tasks_processed': 0,
        'successful_tasks': 0,
        'total_attempts': 0,
        'successful_attempts': 0,
        'above_threshold_attempts': 0,
        'training_examples_generated': 0,
        'task_stats': {}
    }
    
    # Collect all training examples
    all_examples = []
    
    # Process each task
    for task_id, task in tqdm(tasks.items(), desc=f"Processing tasks (iteration {iteration})"):
        print(f"\nProcessing task {task_id}...")
        task_results = {
            'responses': [],
            'metrics': [],
            'analysis': {
                'success_rate': 0,
                'average_iou': 0,
                'max_iou': 0
            }
        }
        
        # Run multiple attempts for this task
        for attempt in range(attempts_per_task):
            # Run model inference
            print(f"\nInference attempt {attempt+1} for task {task_id}")
            print(f"Task has {len(task['train'])} training examples")

            solution_text = run_model_inference(task, model, tokenizer)

            if not solution_text:
                print("Warning: Empty solution text returned!")
            else:
                print(f"Got solution with {len(solution_text)} characters")
                print(f"Preview: {solution_text[:100]}...")

            task_results['responses'].append(solution_text)
            
            # Evaluate the solution
            metrics = evaluate_solution(task, solution_text)
            if metrics is None:
                # Provide a default metrics dict if evaluation returned None
                metrics = {
                    'exact_match': False,
                    'mean_iou': 0.0,
                    'valid_solution': False,
                    'error': 'Evaluation returned None'
                }
            task_results['metrics'].append(metrics)
            
            # Visualize if prediction is available
            if metrics is not None and 'prediction' in metrics:
                # Create visualization directory
                viz_dir = os.path.join(iter_output_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                
                # Generate filename
                iou_score = metrics.get('mean_iou', 0)
                exact_match = "exact" if metrics.get('exact_match', False) else "partial"
                viz_filename = f"{task_id}_attempt{attempt}_{exact_match}_iou{iou_score:.4f}.png"
                viz_path = os.path.join(viz_dir, viz_filename)
                
                # Create and save visualization
                visualize_task_result(
                    task, 
                    prediction_output=metrics['prediction'],
                    save_path=viz_path,
                    show_plot=False  # Don't display in non-interactive environments
                )
        
        # Analyze results for this task
        successful_count = sum(1 for m in task_results['metrics'] if m.get('exact_match', False))
        iou_scores = [m.get('mean_iou', 0) for m in task_results['metrics']]
        
        task_results['analysis'] = {
            'success_rate': successful_count / attempts_per_task,
            'average_iou': sum(iou_scores) / len(iou_scores) if iou_scores else 0,
            'max_iou': max(iou_scores) if iou_scores else 0
        }
        
        # Update task-specific stats
        task_stats = {
            'average_iou': float(task_results['analysis']['average_iou']),
            'max_iou': float(task_results['analysis']['max_iou']),
            'success_rate': float(task_results['analysis']['success_rate']),
            'attempts': attempts_per_task,
            'examples_saved': 0
        }
        
        # Update overall stats
        results_summary['tasks_processed'] += 1
        results_summary['total_attempts'] += attempts_per_task
        
        if task_results['analysis']['success_rate'] > 0:
            results_summary['successful_tasks'] += 1
            results_summary['successful_attempts'] += successful_count
        
        # Select which solutions to save
        selected_indices = []
        
        # Check if any solutions have a perfect score (exact match with IoU=1.0)
        perfect_indices = []
        for i, metrics in enumerate(task_results['metrics']):
            if metrics.get('exact_match', False) and metrics.get('mean_iou', 0) == 1.0:
                perfect_indices.append(i)
        
        # If we have perfect solutions, only use those
        if perfect_indices:
            selected_indices = perfect_indices
            print(f"Found {len(perfect_indices)} perfect solutions for task {task_id}")
        elif success_only:
            # Only include exact matches
            for i, metrics in enumerate(task_results['metrics']):
                if metrics.get('exact_match', False):
                    selected_indices.append(i)
        else:
            # Include solutions above threshold
            threshold = max(min_iou, task_results['analysis']['average_iou'])
            for i, metrics in enumerate(task_results['metrics']):
                if metrics.get('mean_iou', 0) >= threshold:
                    selected_indices.append(i)
        
        # Add early termination option after finding a perfect solution
        if stop_after_success and perfect_indices:
            print(f"Found perfect solution for task {task_id}, skipping additional attempts")
            break
        
        # Save selected solutions
        import time
        
        for idx in selected_indices:
            solution = task_results['responses'][idx]
            metrics = task_results['metrics'][idx]
            
            # Convert to chat messages
            messages = convert_to_chat_messages(task, solution)
            
            # Check if we need special handling for Gemma-3 models
            if "gemma-3" in model.__class__.__name__.lower():
                # Convert the standard messages format to the Gemma-3 format if needed
                if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
                    # Format expected by Gemma-3 according to model card
                    formatted_messages = [
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": messages[0]["content"]}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": messages[1]["content"]}]
                            }
                        ]
                    ]
                    # Store both formats
                    example = {
                        "system": messages[0]["content"],
                        "user": messages[1]["content"],
                        "assistant": messages[2]["content"],
                        "formatted_messages": formatted_messages
                    }
                else:
                    example = {
                        "system": messages[0]["content"],
                        "user": messages[1]["content"], 
                        "assistant": messages[2]["content"]
                    }
            else:
                example = {
                    "system": messages[0]["content"],
                    "user": messages[1]["content"],
                    "assistant": messages[2]["content"]
                }
            
            # Add to our collection
            all_examples.append(example)
            
            # Prepare filename with metrics
            iou_score = metrics.get('mean_iou', 0)
            exact_match = "exact" if metrics.get('exact_match', False) else "partial"
            timestamp = int(time.time())
            
            filename = f"{task_id}_{exact_match}_iou{iou_score:.4f}_{timestamp}_{idx}.json"
            filepath = os.path.join(iter_output_dir, filename)
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(messages, f, indent=2)
            
            # Update stats
            task_stats['examples_saved'] += 1
            results_summary['training_examples_generated'] += 1
            results_summary['above_threshold_attempts'] += 1
        
        # Save task stats
        results_summary['task_stats'][task_id] = task_stats
    
    # Save final results summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(iter_output_dir, f"results_summary_{timestamp}.json")
    
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDataset building complete for iteration {iteration}!")
    print(f"Generated {results_summary['training_examples_generated']} training examples.")
    
    # Create Dataset object
    dataset = Dataset.from_list(all_examples)
    
    return iter_output_dir, dataset


def run_iterative_training(
    initial_model_name="google/gemma-3-4b-it",
    data_path="data",
    base_output_dir="arc_iterations",
    max_iterations=5,
    attempts_per_task=3,
    min_iou=0.7,
    success_only=False,
    limit=None,
    use_peft=True,
    learning_rate=2e-5,
    epochs=3
):
    """
    Run the iterative training loop.
    
    Args:
        initial_model_name: Starting model for the first iteration
        data_path: Path to ARC data
        base_output_dir: Base directory for all outputs
        max_iterations: Maximum number of iterations to run
        attempts_per_task: Number of attempts per task when building datasets
        min_iou: Minimum IoU for including examples in training
        success_only: Whether to only include perfect solutions
        limit: Maximum number of tasks to process (for testing)
        use_peft: Whether to use PEFT for training
        learning_rate: Learning rate for training
        epochs: Number of training epochs per iteration
    """
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "initial_model_name": initial_model_name,
        "data_path": data_path,
        "max_iterations": max_iterations,
        "attempts_per_task": attempts_per_task,
        "min_iou": min_iou,
        "success_only": success_only,
        "limit": limit,
        "use_peft": use_peft,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "started_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(base_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set up iteration tracking
    current_model_name = initial_model_name
    current_model = None
    current_tokenizer = None
    
    # Set up PEFT config if requested
    peft_config = None
    if use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    
    # Run the loop
    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"Starting iteration {iteration + 1}/{max_iterations}")
        print(f"{'='*80}")
        
        # Create iteration directory
        iter_dir = os.path.join(base_output_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Load model for inference if not already loaded
        if current_model is None or current_tokenizer is None:
            print(f"Loading model: {current_model_name}")
            
            # Update the model loading for Gemma-3 models
            if "gemma-3" in current_model_name.lower():
                print("Loading Gemma-3 model with specialized configuration...")
                try:
                    # Check that we're using a compatible transformers version
                    if not GEMMA3_AVAILABLE:
                        raise ImportError("Gemma3ForCausalLM is not available.")
                    
                    # Load tokenizer first
                    current_tokenizer = AutoTokenizer.from_pretrained(current_model_name)
                    
                    # Configure the model with appropriate settings for Gemma-3
                    current_model = Gemma3ForCausalLM.from_pretrained(
                        current_model_name,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto"
                    )
                    
                    # Ensure we have the proper tokenizer settings
                    if current_tokenizer.pad_token is None:
                        current_tokenizer.pad_token = current_tokenizer.eos_token
                        
                    print("Gemma-3 model loaded successfully")
                except Exception as e:
                    print(f"Error loading Gemma-3 model: {e}")
                    print("Make sure you're using the correct version of Transformers. For Gemma-3, install:")
                    print("pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3")
                    raise
            else:
                # Standard loading for other models
                current_model = AutoModelForCausalLM.from_pretrained(
                    current_model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                
                current_tokenizer = AutoTokenizer.from_pretrained(current_model_name)
                if current_tokenizer.pad_token is None:
                    current_tokenizer.pad_token = current_tokenizer.eos_token
        
        # Build dataset with current model
        print(f"Building dataset with model: {current_model_name}")
        dataset_path, dataset = build_dataset_with_model(
            model=current_model,
            tokenizer=current_tokenizer,
            data_path=data_path,
            output_dir=iter_dir,
            attempts_per_task=attempts_per_task,
            min_iou=min_iou,
            success_only=success_only,
            limit=limit,
            iteration=iteration,
            stop_after_success=False
        )
        
        # Train the next model
        print(f"Training model for iteration {iteration}")
        new_model_path = train_model(
            training_data=dataset,
            model_name=current_model_name,
            output_dir=os.path.join(iter_dir, "model"),
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            peft_config=peft_config
        )
        
        # Update for next iteration
        current_model_name = new_model_path
        
        # Free up memory by deleting old model
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache()
        current_model = None
        current_tokenizer = None
        
        # Save iteration results
        iteration_result = {
            "iteration": iteration,
            "input_model": current_model_name,
            "output_model": new_model_path,
            "dataset_path": dataset_path,
            "completed_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(iter_dir, "iteration_result.json"), "w") as f:
            json.dump(iteration_result, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Iterative training complete!")
    print(f"{'='*80}")
    print(f"Final model: {current_model_name}")
    print(f"Output directory: {base_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run iterative training loop for ARC solver')
    parser.add_argument('--initial_model', default='google/gemma-3-4b-it', 
                       help='Initial model to start with')
    parser.add_argument('--data_path', default='data', help='Path to ARC data')
    parser.add_argument('--output_dir', default='arc_iterations', 
                       help='Base directory for outputs')
    parser.add_argument('--iterations', type=int, default=5, 
                       help='Maximum number of iterations')
    parser.add_argument('--attempts', type=int, default=3, 
                       help='Attempts per task when building datasets')
    parser.add_argument('--min_iou', type=float, default=0.7, 
                       help='Minimum IoU score for training examples')
    parser.add_argument('--success_only', action='store_true', 
                       help='Only include perfect solutions')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of tasks (for testing)')
    parser.add_argument('--use_peft', action='store_true', 
                       help='Use PEFT for more efficient training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                       help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of epochs per training iteration')
    parser.add_argument('--logging_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.logging_level)
    
    # Run iterative training
    run_iterative_training(
        initial_model_name=args.initial_model,
        data_path=args.data_path,
        base_output_dir=args.output_dir,
        max_iterations=args.iterations,
        attempts_per_task=args.attempts,
        min_iou=args.min_iou,
        success_only=args.success_only,
        limit=args.limit,
        use_peft=args.use_peft,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main() 