#!/usr/bin/env python
"""
Functions for training models on ARC data.
"""
import os
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from datetime import datetime
from typing import Dict, List, Optional, Any

def format_prompt(example):
    """Format the prompt for training"""
    system = example["system"]
    user = example["user"]
    
    # Format: system message followed by user message
    prompt = f"{system}\n\n{user}"
    
    return {
        "text": f"{prompt}\n\n{example['assistant']}"
    }


def train_model(
    training_data,
    model_name="google/gemma-2b",
    output_dir="models",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    max_seq_length=4096,
    peft_config=None,
):
    """
    Train a model using the SFTTrainer from TRL.
    
    Args:
        training_data: Dataset object with training examples
        model_name: Pretrained model to fine-tune
        output_dir: Directory to save the trained model
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        gradient_checkpointing: Whether to use gradient checkpointing
        max_seq_length: Maximum sequence length
        peft_config: Optional PEFT configuration
    
    Returns:
        Path to the trained model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create a timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"{os.path.basename(model_name)}_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Process dataset to create formatted examples
    processed_dataset = training_data.map(format_prompt)
    
    print(f"Prepared {len(processed_dataset)} examples for training")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=model_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),  # Use bfloat16 if available
        report_to="tensorboard",
        save_total_limit=2,  # Keep only the 2 best checkpoints
    )
    
    # Create SFT Trainer with updated parameters
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        peft_config=peft_config,
        dataset_text_field="text",  # Use the key from format_prompt output
        processing_class=tokenizer,  # Replace deprecated 'tokenizer' parameter
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(model_output_dir)
    
    print(f"Training complete! Model saved to {model_output_dir}")
    return model_output_dir 