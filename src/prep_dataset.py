#!/usr/bin/env python3
"""
Script to prepare a better dataset for GPT-2 fine-tuning
"""

import os
import json
import argparse
import re
import logging
from tqdm import tqdm
import random

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Strip extra whitespace
    text = text.strip()
    
    return text

def process_files(input_files, min_length=100, max_length=100000):
    """
    Process input files and extract text samples
    
    Args:
        input_files: List of input files or directories
        min_length: Minimum text length to keep
        max_length: Maximum text length to keep
        
    Returns:
        List of processed text samples
    """
    all_texts = []
    
    for input_path in input_files:
        if os.path.isdir(input_path):
            # Process all files in directory
            for root, _, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    texts = process_single_file(file_path, min_length, max_length)
                    all_texts.extend(texts)
        else:
            # Process single file
            texts = process_single_file(input_path, min_length, max_length)
            all_texts.extend(texts)
    
    logger.info(f"Collected a total of {len(all_texts)} text samples")
    return all_texts

def process_single_file(file_path, min_length, max_length):
    """Process a single file and extract text"""
    logger.info(f"Processing {file_path}")
    
    try:
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() in ['.txt', '.md', '.csv']:
            return process_text_file(file_path, min_length, max_length)
        elif ext.lower() == '.json':
            return process_json_file(file_path, min_length, max_length)
        else:
            logger.warning(f"Unsupported file type: {ext}. Skipping {file_path}")
            return []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def process_text_file(file_path, min_length, max_length):
    """Extract text from a text file"""
    texts = []
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    
    # Process each paragraph
    for para in paragraphs:
        clean_para = clean_text(para)
        if min_length <= len(clean_para) <= max_length:
            texts.append(clean_para)
    
    return texts

def process_json_file(file_path, min_length, max_length):
    """Extract text from a JSON file"""
    texts = []
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        # If it's a list of strings
        if data and isinstance(data[0], str):
            for item in data:
                clean_item = clean_text(item)
                if min_length <= len(clean_item) <= max_length:
                    texts.append(clean_item)
        
        # If it's a list of objects with a 'text' field
        elif data and isinstance(data[0], dict):
            text_fields = ["text", "content", "body", "description"]
            
            for item in data:
                for field in text_fields:
                    if field in item and isinstance(item[field], str):
                        clean_item = clean_text(item[field])
                        if min_length <= len(clean_item) <= max_length:
                            texts.append(clean_item)
                        break
    
    # If it's a dictionary with text fields
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                clean_value = clean_text(value)
                if min_length <= len(clean_value) <= max_length:
                    texts.append(clean_value)
    
    return texts

def save_dataset(texts, output_path, split_ratio=0.9, format='json'):
    """
    Save processed texts to a dataset file
    
    Args:
        texts: List of text samples
        output_path: Output file path
        split_ratio: Train/validation split ratio
        format: Output format ('json' or 'text')
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Shuffle texts
    random.shuffle(texts)
    
    # Split into train and validation sets
    split_idx = int(len(texts) * split_ratio)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    logger.info(f"Saving {len(train_texts)} training and {len(val_texts)} validation samples")
    
    # Save based on format
    if format == 'json':
        train_path = f"{os.path.splitext(output_path)[0]}_train.json"
        val_path = f"{os.path.splitext(output_path)[0]}_val.json"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_texts, f, ensure_ascii=False, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_texts, f, ensure_ascii=False, indent=2)
        
        # Also save combined dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
    
    elif format == 'text':
        train_path = f"{os.path.splitext(output_path)[0]}_train.txt"
        val_path = f"{os.path.splitext(output_path)[0]}_val.txt"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_texts))
        
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_texts))
        
        # Also save combined dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
    
    logger.info(f"Datasets saved: {output_path}, {train_path}, {val_path}")

def augment_dataset(texts, augmentation_factor=3):
    """
    Augment the dataset by creating variations of existing texts
    
    Args:
        texts: List of original text samples
        augmentation_factor: Number of variations to create for each text
        
    Returns:
        List of original and augmented texts
    """
    augmented_texts = list(texts)  # Start with original texts
    
    logger.info(f"Augmenting dataset from {len(texts)} to approximately {len(texts) * (augmentation_factor + 1)} samples")
    
    for text in tqdm(texts, desc="Augmenting dataset"):
        # Only augment texts that are long enough
        if len(text) < 200:
            continue
        
        for _ in range(augmentation_factor):
            augmented = augment_single_text(text)
            if augmented:
                augmented_texts.append(augmented)
    
    logger.info(f"Final dataset size after augmentation: {len(augmented_texts)} samples")
    return augmented_texts

def augment_single_text(text):
    """Create a variation of a single text"""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) < 3:
        return None
    
    augmentation_type = random.choice(['truncate', 'subset', 'shuffle'])
    
    if augmentation_type == 'truncate':
        # Use a random portion of the text
        cut_point = random.randint(len(sentences) // 3, len(sentences) - 1)
        return ' '.join(sentences[:cut_point])
    
    elif augmentation_type == 'subset':
        # Pick a random subset of sentences
        num_sentences = max(3, len(sentences) // 2)
        selected_indices = sorted(random.sample(range(len(sentences)), num_sentences))
        return ' '.join(sentences[i] for i in selected_indices)
    
    elif augmentation_type == 'shuffle':
        # Shuffle some sentences while maintaining some order
        shuffle_start = random.randint(1, len(sentences) // 3)
        shuffle_end = random.randint(len(sentences) * 2 // 3, len(sentences) - 1)
        
        shuffled_sentences = sentences[:shuffle_start]
        middle_sentences = sentences[shuffle_start:shuffle_end]
        random.shuffle(middle_sentences)
        shuffled_sentences.extend(middle_sentences)
        shuffled_sentences.extend(sentences[shuffle_end:])
        
        return ' '.join(shuffled_sentences)

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset for GPT-2 fine-tuning")
    parser.add_argument("--input", nargs='+', required=True, help="Input files or directories")
    parser.add_argument("--output", required=True, help="Output dataset path")
    parser.add_argument("--min_length", type=int, default=100, help="Minimum text length")
    parser.add_argument("--max_length", type=int, default=100000, help="Maximum text length")
    parser.add_argument("--format", choices=['json', 'text'], default='json', help="Output format")
    parser.add_argument("--augment", type=int, default=0, help="Augmentation factor (0 to disable)")
    
    args = parser.parse_args()
    
    # Process input files and extract texts
    texts = process_files(
        input_files=args.input,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    if not texts:
        logger.error("No valid text samples found!")
        return
    
    # Augment dataset if requested
    if args.augment > 0:
        texts = augment_dataset(texts, augmentation_factor=args.augment)
    
    # Save the dataset
    save_dataset(
        texts=texts,
        output_path=args.output,
        format=args.format
    )
    
    logger.info("Dataset preparation completed!")

if __name__ == "__main__":
    main()