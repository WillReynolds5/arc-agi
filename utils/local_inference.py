import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# Backward compatibility layer
from arc.inference import LlamaInference, run_inference_on_tasks

class LlamaInference:
    def __init__(self, model_path):
        """
        Initialize the Llama inference engine.
        
        Args:
            model_path: Path to the Llama model
        """
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