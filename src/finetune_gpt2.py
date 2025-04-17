#!/usr/bin/env python3
"""
Enhanced GPT-2 fine-tuning script with XML tag support, dataset format handling,
and continued fine-tuning support
"""

import os
import json
import argparse
import torch
import numpy as np
import re
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    GPT2Config,
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset, DatasetDict
import logging
import gc
from typing import List, Dict, Any, Optional, Tuple, Union
import random

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define XML tags that will be used as special tokens
DEFAULT_SPECIAL_TOKENS = {
    # Basic structural tags
    "section_open": "<section>",
    "section_close": "</section>",
    "question_open": "<question>",
    "question_close": "</question>",
    "answer_open": "<answer>",
    "answer_close": "</answer>",
    "context_open": "<context>",
    "context_close": "</context>",
    "dialog_open": "<dialog>",
    "dialog_close": "</dialog>",
    # Additional tags for specialized datasets
    "instruction_open": "<instruction>",
    "instruction_close": "</instruction>",
    "document_open": "<document>",
    "document_close": "</document>",
    "text_open": "<text>",
    "text_close": "</text>",
    # Story and fiction tags
    "story_open": "<story>",
    "story_close": "</story>",
    "fiction_open": "<fiction>",
    "fiction_close": "</fiction>",
    # Title tags
    "title_open": "<title>",
    "title_close": "</title>",
    "think_open": "<think>",
    "think_close": "</think>",
    # Miscellaneous tags
    "note_open": "<note>",
    "note_close": "</note>",
    "warning_open": "<warning>",
    "warning_close": "</warning>",
    "error_open": "<error>",
    "error_close": "</error>",
    "hint_open": "<hint>",
    "hint_close": "</hint>",
    "document_title_open": "<document_title>",
    "document_title_close": "</document_title>",
    "paragraph_open": "<paragraph>",
    "paragraph_close": "</paragraph>",
    "list_open": "<list>",
    "list_close": "</list>",
    "item_open": "<item>",
    "item_close": "</item>",
    "link_open": "<link>",
    "link_close": "</link>",
    "code_open": "<code>",
    "code_close": "</code>",
    "quote_open": "<quote>",
    "quote_close": "</quote>",
    "example_open": "<example>",
    "example_close": "</example>",
    "question_id_open": "<question_id>",
    "question_id_close": "</question_id>",
    "document_id_open": "<document_id>",
    "document_id_close": "</document_id>",
    "passage_open": "<passage>",
    "passage_close": "</passage>",
    "metadata_open": "<metadata>",
    "metadata_close": "</metadata>"
}


def detect_format(file_path: str) -> str:
    """
    Detect the format of the dataset file
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Detected format ('json', 'txt', 'huggingface', 'unknown')
    """
    if file_path.startswith("hf://"):
        return "huggingface"
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.json':
        return "json"
    elif file_extension in ['.txt', '.text', '.md']:
        return "txt"
    else:
        # Try to infer format by reading the first few lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = "".join([f.readline() for _ in range(5)])
                
                if first_lines.strip().startswith('{') or first_lines.strip().startswith('['):
                    return "json"
                else:
                    return "txt"
        except:
            return "unknown"

def preprocess_text_with_tags(text: str) -> str:
    """
    Preprocess text with XML tags to ensure consistent formatting
    
    Args:
        text: Raw text that may contain XML tags
        
    Returns:
        Preprocessed text with consistent tag spacing
    """
    # Fix spacing around tags (no space inside tags, space outside)
    text = re.sub(r'<([/\w]+)>\s+', r'<\1> ', text)  # Add space after closing tag
    text = re.sub(r'\s+<([/\w]+)>', r' <\1>', text)  # Add space before opening tag
    text = re.sub(r'\s+<([/\w]+)>\s+', r' <\1> ', text)  # Normalize spaces around tags
    
    # Fix common tag format issues
    text = re.sub(r'<\s+([/\w]+)\s+>', r'<\1>', text)  # Remove spaces inside tags
    
    # Ensure consistent newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def load_and_prepare_data(
    data_path: str, 
    format_type: Optional[str] = None,
    huggingface_config: Optional[str] = None,
    huggingface_split: Optional[str] = None,
    add_special_tokens: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Load and prepare data from various sources
    
    Args:
        data_path: Path to the data file or Hugging Face dataset name
        format_type: Format type ('json', 'txt', 'huggingface', or None for auto-detection)
        huggingface_config: Config name for Hugging Face dataset
        huggingface_split: Split name for Hugging Face dataset
        add_special_tokens: Whether to add special tokens to the text
        
    Returns:
        Tuple of train_texts and validation_texts
    """
    # Auto-detect format if not specified
    if not format_type:
        format_type = detect_format(data_path)
    
    logger.info(f"Loading data from {data_path} (format: {format_type})")
    
    texts = []
    
    # Handle Hugging Face datasets
    if format_type == "huggingface" or data_path.startswith("hf://"):
        # Extract dataset name
        dataset_name = data_path
        if dataset_name.startswith("hf://"):
            dataset_name = dataset_name[5:]
        
        # Load the dataset
        try:
            if huggingface_config:
                dataset = load_dataset(dataset_name, huggingface_config)
            else:
                dataset = load_dataset(dataset_name)
            
            logger.info(f"Loaded Hugging Face dataset: {dataset_name}")
            
            # Handle different dataset structures
            if isinstance(dataset, DatasetDict):
                # Use specified split or default to first available split
                split = huggingface_split or list(dataset.keys())[0]
                if split not in dataset:
                    logger.warning(f"Split {split} not found. Available splits: {list(dataset.keys())}")
                    split = list(dataset.keys())[0]
                
                dataset = dataset[split]
                logger.info(f"Using split: {split}")
            
            # Check for text fields
            text_fields = []
            for key in dataset.features.keys():
                if key in ['text', 'content', 'sentence', 'question', 'answer', 'context']:
                    text_fields.append(key)
            
            if not text_fields:
                logger.warning(f"No standard text fields found. Available fields: {list(dataset.features.keys())}")
                # Use the first string field as fallback
                for key in dataset.features.keys():
                    if isinstance(dataset[0][key], str):
                        text_fields.append(key)
                        break
            
            if not text_fields:
                raise ValueError(f"No suitable text fields found in dataset {dataset_name}")
            
            # Extract texts from the dataset
            for example in dataset:
                example_texts = []
                
                for field in text_fields:
                    value = example.get(field)
                    
                    if isinstance(value, str) and value.strip():
                        # Add field tags if using special tokens
                        if add_special_tokens:
                            example_texts.append(f"<{field}>{value}</{field}>")
                        else:
                            example_texts.append(value)
                
                if example_texts:
                    texts.append(" ".join(example_texts))
            
            logger.info(f"Extracted {len(texts)} texts from dataset {dataset_name}")
        
        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {e}")
            raise
    
    # Handle JSON files
    elif format_type == "json":
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Handle list of strings
                if all(isinstance(item, str) for item in data):
                    texts = data
                # Handle list of objects with text fields
                elif all(isinstance(item, dict) for item in data):
                    for item in data:
                        # Check for common text fields
                        for field in ['text', 'content', 'sentence', 'question', 'answer', 'context']:
                            if field in item and isinstance(item[field], str) and item[field].strip():
                                if add_special_tokens:
                                    texts.append(f"<{field}>{item[field]}</{field}>")
                                else:
                                    texts.append(item[field])
                                break
                        else:
                            # If no standard fields found, use the first string field
                            for key, value in item.items():
                                if isinstance(value, str) and value.strip():
                                    if add_special_tokens:
                                        texts.append(f"<{key}>{value}</{key}>")
                                    else:
                                        texts.append(value)
                                    break
            elif isinstance(data, dict):
                # Handle dictionary with text values
                for key, value in data.items():
                    if isinstance(value, str) and value.strip():
                        if add_special_tokens:
                            texts.append(f"<{key}>{value}</{key}>")
                        else:
                            texts.append(value)
            
            logger.info(f"Loaded {len(texts)} text samples from JSON")
        
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise
    
    # Handle text files
    elif format_type == "txt":
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Check if the text already contains XML tags
            has_tags = any(re.search(r'<[\w/]+>', line) for line in lines[:10])
            
            if has_tags:
                # Text already has tags
                texts = lines
            else:
                # Add <text> tags if no tags are present
                texts = [f"<text>{line}</text>" if add_special_tokens else line for line in lines]
            
            logger.info(f"Loaded {len(texts)} lines from text file")
        
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Preprocess texts
    processed_texts = []
    for text in texts:
        if text.strip():
            processed_texts.append(preprocess_text_with_tags(text))
    
    # Split into train and validation sets
    random.seed(42)  # For reproducibility
    random.shuffle(processed_texts)
    
    split_idx = int(0.9 * len(processed_texts))
    train_texts = processed_texts[:split_idx]
    val_texts = processed_texts[split_idx:]
    
    logger.info(f"Split into {len(train_texts)} training and {len(val_texts)} validation samples")
    
    return train_texts, val_texts

def prepare_datasets(
    train_texts: List[str], 
    val_texts: List[str], 
    tokenizer, 
    max_length: int = 512
) -> Tuple[Dataset, Dataset]:
    """
    Prepare datasets from text lists
    
    Args:
        train_texts: List of training texts
        val_texts: List of validation texts
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of train and validation datasets
    """
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    # Process dataset in smaller batches to reduce memory usage
    batch_size = min(16, len(train_dataset))
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length,
            padding="max_length"  # This helps with uniform sequence lengths
        )
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        desc="Tokenizing training dataset"
    )
    
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        desc="Tokenizing validation dataset"
    )
    
    # Clean up to free memory
    del train_dataset, val_dataset
    gc.collect()
    
    # Add labels for causal language modeling
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    train_dataset = tokenized_train.map(
        add_labels,
        batched=True,
        batch_size=batch_size,
        desc="Adding labels to train dataset"
    )
    
    val_dataset = tokenized_val.map(
        add_labels,
        batched=True,
        batch_size=batch_size,
        desc="Adding labels to validation dataset"
    )
    
    # Force release memory
    del tokenized_train, tokenized_val
    gc.collect()
    
    return train_dataset, val_dataset

def create_tokenizer_with_special_tokens(
    model_name_or_path: str,
    special_tokens: Optional[Dict[str, str]] = None
) -> Tuple[GPT2Tokenizer, bool]:
    """
    Create tokenizer with special tokens for XML tags
    
    Args:
        model_name_or_path: Pretrained model name or path
        special_tokens: Dictionary of special tokens to add
        
    Returns:
        Tuple of tokenizer and whether new tokens were added
    """
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    
    # Make sure there's a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Track if we added new tokens
    added_tokens = False
    
    # Add special tokens for XML tags
    if special_tokens:
        # Get all special tokens as a list
        special_tokens_list = list(special_tokens.values())
        
        # Filter out tokens that are already in the vocabulary
        new_tokens = [token for token in special_tokens_list if token not in tokenizer.get_vocab()]
        
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            logger.info(f"Added {len(new_tokens)} special tokens to the tokenizer vocabulary")
            added_tokens = True
        else:
            logger.info("No new special tokens added (all already in vocabulary)")
    
    return tokenizer, added_tokens


def fine_tune_gpt2(
    model_name: str = "gpt2",
    data_files: Optional[str] = None,
    output_dir: str = "./gpt2-finetuned",
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    warmup_steps: int = 500,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    fp16: bool = True,
    cpu_offload: bool = True,
    use_special_tokens: bool = True,
    save_steps: int = 1000,
    eval_steps: int = 500,
    huggingface_config: Optional[str] = None,
    huggingface_split: Optional[str] = None,
    continue_training: bool = False,
    continue_from: Optional[str] = None
    ):    
    
    """
    Memory-efficient fine-tuning of GPT-2
    
    Args:
        model_name: Pretrained model name (gpt2, gpt2-medium, etc.)
        data_files: Path to data files
        output_dir: Output directory for the fine-tuned model
        batch_size: Batch size for training
        gradient_accumulation_steps: Accumulate gradients over multiple steps
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_length: Maximum sequence length
        warmup_steps: Number of warmup steps
        warmup_ratio: Ratio of total training steps to use for warmup
        weight_decay: Weight decay for regularization
        fp16: Use mixed precision training
        cpu_offload: Use CPU offloading for memory efficiency
        use_special_tokens: Add special tokens for XML tags
        save_steps: Save checkpoint every X steps
        eval_steps: Evaluate every X steps
        huggingface_config: Config name for Hugging Face dataset
        huggingface_split: Split name for Hugging Face dataset
        continue_training: Whether to continue training from a previous checkpoint
        continue_from: Path to the previous model to continue training from
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
    
    # Determine the model path to use (original model or continue from checkpoint)
    model_path = continue_from if continue_training and continue_from else model_name
    logger.info(f"Using model from: {model_path}")
    
    # Create tokenizer with special tokens
    special_tokens = DEFAULT_SPECIAL_TOKENS if use_special_tokens else None
    tokenizer, added_tokens = create_tokenizer_with_special_tokens(model_path, special_tokens)
    
    # Memory-efficient model loading
    device_map = "auto" if cpu_offload else None
    
    if cpu_offload:
        logger.info("Using CPU offloading for memory efficiency")
        try:
            # Try to use accelerate for CPU offloading
            from accelerate import init_empty_weights
            from accelerate.utils import get_balanced_memory
            
            # If continuing from a checkpoint, load config from there
            if continue_training and continue_from:
                model_config = GPT2Config.from_pretrained(continue_from)
            else:
                model_config = GPT2Config.from_pretrained(model_name)
            
            # Calculate optimal device map
            max_memory = {0: "3GB", "cpu": "16GB"}  # Adjust based on your system
            device_map = get_balanced_memory(model_config, max_memory=max_memory)
            
            with init_empty_weights():
                model = GPT2LMHeadModel(model_config)
            
            model = GPT2LMHeadModel.from_pretrained(
                model_path,
                device_map=device_map,
                offload_folder="offload",
                offload_state_dict=True
            )
        except (ImportError, Exception) as e:
            logger.warning(f"CPU offloading with accelerate failed: {e}")
            logger.info("Falling back to standard model loading with smaller model size")
            model = GPT2LMHeadModel.from_pretrained(model_path)
            if added_tokens:
                model.resize_token_embeddings(len(tokenizer))
            if device == "cuda":
                model = model.to(device)
    else:
        # Standard model loading
        model = GPT2LMHeadModel.from_pretrained(model_path)
        if added_tokens:
            model.resize_token_embeddings(len(tokenizer))
        if device == "cuda":
            model = model.to(device)
    
    # Prepare datasets
    if data_files:
        # Load and prepare data
        train_texts, val_texts = load_and_prepare_data(
            data_path=data_files,
            huggingface_config=huggingface_config,
            huggingface_split=huggingface_split,
            add_special_tokens=use_special_tokens
        )
        
        # Prepare datasets
        train_dataset, val_dataset = prepare_datasets(
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Use causal language modeling
        )
        
        # For transformers 4.51.0, we need to make sure only one warmup parameter is used
        # The key issue is that when warmup_steps is 0, it's still being considered in validation
        
        # Create training arguments dictionary first
        training_args_dict = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": 100,
            "save_steps": save_steps,
            "save_total_limit": 3,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "fp16": fp16 and device == "cuda",
            "report_to": "none",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": True if device == "cuda" else False,
            "optim": "adamw_torch",
        }
        
        # Add either warmup_steps or warmup_ratio, but not both
        if warmup_steps > 0:
            training_args_dict["warmup_steps"] = warmup_steps
        else:
            # Only add warmup_ratio if warmup_steps is not being used
            training_args_dict["warmup_ratio"] = max(0.01, warmup_ratio)
        
        # Create TrainingArguments from dictionary
        training_args = TrainingArguments(**training_args_dict)
                                
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
    parser = argparse.ArgumentParser(description="Enhanced GPT-2 fine-tuning with XML tag support")
    parser.add_argument("--model", default="gpt2", help="Base model to use (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--data_files", required=True, help="Path to data files (text, JSON, or Hugging Face dataset)")
    parser.add_argument("--output_dir", default="./gpt2-finetuned", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (smaller values use less memory)")
    parser.add_argument("--gradient_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--no_cpu_offload", action="store_true", help="Disable CPU offloading")
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--no_special_tokens", action="store_true", help="Disable special tokens for XML tags")
    parser.add_argument("--hf_config", type=str, default=None, help="Hugging Face dataset config name")
    parser.add_argument("--hf_split", type=str, default=None, help="Hugging Face dataset split name")
    parser.add_argument("--continue_training", action="store_true", help="Continue training from a previous checkpoint")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to model to continue training from")
    
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    
    # Check for continue_training logic
    if args.continue_training and not args.continue_from:
        if os.path.exists(args.output_dir):
            logger.info(f"No continue_from specified, but output_dir exists. Using {args.output_dir} to continue training.")
            args.continue_from = args.output_dir
        else:
            logger.warning("--continue_training specified but neither --continue_from nor output_dir exists. Starting from scratch.")
            args.continue_training = False
    
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
        fp16=not args.no_fp16,
        use_special_tokens=not args.no_special_tokens,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        huggingface_config=args.hf_config,
        huggingface_split=args.hf_split,
        continue_training=args.continue_training,
        continue_from=args.continue_from
    )

if __name__ == "__main__":
    main()