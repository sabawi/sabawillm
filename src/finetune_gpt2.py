#!/usr/bin/env python3
"""
Memory-efficient GPT-2 fine-tuning script with CPU offloading
"""

import os
import json
import argparse
import torch
import numpy as np
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging
import gc

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def prepare_datasets(data_files, tokenizer, max_length=512, split_ratio=0.9):
    """
    Prepare datasets from text or JSON files with reduced memory usage
    
    Args:
        data_files: Path to data files (can be text or JSON)
        tokenizer: GPT-2 tokenizer
        max_length: Maximum sequence length
        split_ratio: Train/validation split ratio
        
    Returns:
        Train and validation datasets
    """
    logger.info(f"Loading data from {data_files}")
    
    # Check file extension
    file_extension = os.path.splitext(data_files)[1]
    
    if file_extension == '.json':
        # Load JSON data
        with open(data_files, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            # If it's a list of strings
            if isinstance(data[0], str):
                texts = data
            # If it's a list of objects with a 'text' field
            elif isinstance(data[0], dict) and 'text' in data[0]:
                texts = [item['text'] for item in data]
            else:
                raise ValueError("JSON format not recognized. Should be a list of strings or objects with 'text' field.")
        else:
            raise ValueError("JSON should contain a list of texts or objects.")
        
        logger.info(f"Loaded {len(texts)} text samples from JSON")
    else:
        # For text files, use simple loading
        with open(data_files, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(texts)} lines from text file")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Process dataset in smaller batches to reduce memory usage
    batch_size = min(16, len(dataset))
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length,
            padding="max_length"  # This helps with uniform sequence lengths
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        desc="Tokenizing the dataset"
    )
    
    # Clean up to free memory
    del dataset
    gc.collect()
    
    # Split into training and validation
    split = int(split_ratio * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(split))
    val_dataset = tokenized_dataset.select(range(split, len(tokenized_dataset)))
    
    logger.info(f"Split into {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    # Add labels for causal language modeling
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    train_dataset = train_dataset.map(
        add_labels,
        batched=True,
        batch_size=batch_size,
        desc="Adding labels to train dataset"
    )
    
    val_dataset = val_dataset.map(
        add_labels,
        batched=True,
        batch_size=batch_size,
        desc="Adding labels to validation dataset"
    )
    
    # Force release memory
    del tokenized_dataset
    gc.collect()
    
    return train_dataset, val_dataset

def fine_tune_gpt2(
    model_name="gpt2",
    data_files=None,
    output_dir="./gpt2-finetuned",
    batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=4,  # Added gradient accumulation 
    num_epochs=3,
    learning_rate=5e-5,
    max_length=512,  # Reduced max length
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    cpu_offload=True,  # New parameter for CPU offloading
    save_steps=1000,
    eval_steps=500
):
    """
    Memory-efficient fine-tuning of GPT-2
    
    Args:
        model_name: Pretrained model name (gpt2, gpt2-medium, etc.)
        data_files: Path to data files
        output_dir: Output directory for the fine-tuned model
        batch_size: Batch size for training (small for memory efficiency)
        gradient_accumulation_steps: Accumulate gradients over multiple steps
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_length: Maximum sequence length (reduced for memory efficiency)
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for regularization
        fp16: Use mixed precision training
        cpu_offload: Use CPU offloading for memory efficiency
        save_steps: Save checkpoint every X steps
        eval_steps: Evaluate every X steps
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for CUDA and free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        memory_stats = torch.cuda.memory_stats()
        logger.info(f"GPU memory: {memory_stats.get('allocated_bytes.all.current', 0) / 1024**3:.2f} GB allocated")
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Memory-efficient model loading
    device_map = "auto" if cpu_offload else None
    
    if cpu_offload:
        logger.info("Using CPU offloading for memory efficiency")
        try:
            # Try to use accelerate for CPU offloading
            from accelerate import init_empty_weights
            from accelerate.utils import get_balanced_memory
            
            model_config = GPT2LMHeadModel.config_class.from_pretrained(model_name)
            
            # Calculate optimal device map
            max_memory = {0: "3GB", "cpu": "16GB"}  # Adjust based on your system
            device_map = get_balanced_memory(model_config, max_memory=max_memory)
            
            with init_empty_weights():
                model = GPT2LMHeadModel.from_config(model_config)
            
            model = GPT2LMHeadModel.from_pretrained(
                model_name,
                device_map=device_map,
                offload_folder="offload",
                offload_state_dict=True
            )
        except (ImportError, Exception) as e:
            logger.warning(f"CPU offloading with accelerate failed: {e}")
            logger.info("Falling back to standard model loading with smaller model size")
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.resize_token_embeddings(len(tokenizer))
            if device == "cuda":
                model = model.to(device)
    else:
        # Standard model loading
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        if device == "cuda":
            model = model.to(device)
    
    # Prepare datasets
    if data_files:
        train_dataset, val_dataset = prepare_datasets(
            data_files=data_files,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Use causal language modeling
        )
        
        # Training arguments - optimized for memory efficiency
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            save_steps=save_steps,
            save_total_limit=3,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            fp16=fp16 and device == "cuda",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=2,  # Parallelize data loading
            dataloader_pin_memory=True if device == "cuda" else False,
            optim="adamw_torch",  # More memory efficient optimizer
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training complete! Model saved to {output_dir}")
    else:
        logger.error("No data files provided for fine-tuning!")

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient fine-tuning of GPT-2")
    parser.add_argument("--model", default="gpt2", help="Base model to use (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--data_files", required=True, help="Path to data files (text or JSON)")
    parser.add_argument("--output_dir", default="./gpt2-finetuned", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (smaller values use less memory)")
    parser.add_argument("--gradient_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--no_cpu_offload", action="store_true", help="Disable CPU offloading")
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    logger.info(args)
    
    fine_tune_gpt2(
        model_name=args.model,
        data_files=args.data_files,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.seq_length,
        cpu_offload=not args.no_cpu_offload,
        weight_decay=args.weight_decay,
        fp16=not args.no_fp16
    )

if __name__ == "__main__":
    main()