import os
import logging
from datasets import load_dataset, Value, Sequence, Features, DatasetDict, get_dataset_config_names

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_text_fields(example, path=None, exclude_fields=None):
    """Recursively find all text fields in the dataset example"""
    if path is None:
        path = []
    if exclude_fields is None:
        exclude_fields = {"id", "title", "answer_start"}

    text_paths = []

    if isinstance(example, str):
        if path and path[-1] not in exclude_fields:
            return [path]
    elif isinstance(example, dict):
        for k, v in example.items():
            if k not in exclude_fields:
                new_paths = find_text_fields(v, path + [k], exclude_fields)
                text_paths.extend(new_paths)
    elif isinstance(example, list):
        if all(isinstance(i, str) for i in example):
            if path and path[-1] not in exclude_fields:
                return [path]
        elif example:
            # For lists, we search through all items, not just the first one
            for item in example:
                new_paths = find_text_fields(item, path, exclude_fields)
                text_paths.extend(new_paths)

    return text_paths

def extract_nested_value(example, path):
    """Extract values from nested structures following the given path"""
    current = example
    
    # Handle the path traversal differently for different data structures
    for i, p in enumerate(path):
        if isinstance(current, dict):
            if p in current:
                current = current[p]
            else:
                return []
        elif isinstance(current, list):
            # For lists, we need to process each item
            results = []
            for item in current:
                if isinstance(item, dict) and p in item:
                    # Continue traversing with remaining path
                    if i == len(path) - 1:  # Last element in path
                        val = item[p]
                        if isinstance(val, str):
                            results.append(val)
                        elif isinstance(val, list) and all(isinstance(x, str) for x in val):
                            results.extend(val)
                    else:
                        # Need to recursively traverse
                        temp_result = extract_nested_value(item, path[i:])
                        if temp_result:
                            results.extend(temp_result)
            return [r for r in results if isinstance(r, str) and r.strip()]
        else:
            return []
    
    # Handle the final value
    if isinstance(current, str):
        return [current] if current.strip() else []
    elif isinstance(current, list):
        if all(isinstance(x, str) for x in current):
            return [x for x in current if x.strip()]
        else:
            # Lists of complex objects
            results = []
            for item in current:
                if isinstance(item, str) and item.strip():
                    results.append(item)
            return results
    
    return []

def convert_example_to_xml(example, text_paths, exclude_fields=None, complex_tags=None):
    """Convert an example to XML format based on detected paths"""
    if exclude_fields is None:
        exclude_fields = set()
    if complex_tags is None:
        complex_tags = set()
        
    xml_parts = []
    grouped = {}

    # Special handling for common dataset structures
    if "question" in example and "answers" in example and "context" in example:
        # Handle QA datasets like SQuAD
        question = example.get("question", "")
        context = example.get("context", "")
        
        answers = []
        if isinstance(example["answers"], dict) and "text" in example["answers"]:
            answers = example["answers"]["text"]
        elif isinstance(example["answers"], list):
            for ans in example["answers"]:
                if isinstance(ans, dict) and "text" in ans:
                    answers.append(ans["text"])
        
        if question and answers and isinstance(answers, list) and len(answers) > 0:
            xml_parts.append(f"<question>{question}</question>")
            xml_parts.append(f"<answer>{answers[0]}</answer>")
            if context:
                xml_parts.append(f"<context>{context}</context>")
            return {"text": "".join(xml_parts)}
    
    # Handle complex web questions format
    if "question" in example and "answers" in example:
        question = example.get("question", "")
        
        answers = []
        if isinstance(example["answers"], list):
            answers = example["answers"]
        
        if question and answers:
            xml_parts.append(f"<question>{question}</question>")
            xml_parts.append(f"<answer>{' '.join(answers[:1])}</answer>")
            return {"text": "".join(xml_parts)}
    
    # Generic handling for other dataset structures
    for path in text_paths:
        if path and path[-1] in exclude_fields:
            continue
            
        values = extract_nested_value(example, path)
        logger.debug(f"Path: {path}, Values: {values}")
        
        if not values:
            continue
            
        # Use the last part of the path as the tag name
        tag = path[-1]
        top = path[0] if path else None
        
        if top and top in complex_tags:
            grouped.setdefault(top, []).extend((tag, v) for v in values)
        else:
            for value in values:
                if value.strip():
                    xml_parts.append(f"<{tag}>{value}</{tag}>")
    
    # Add complex tags
    for top, fields in grouped.items():
        if fields:
            xml_parts.append(f"<{top}>")
            for tag, text in fields:
                if text.strip():
                    xml_parts.append(f"<{tag}>{text}</{tag}>")
            xml_parts.append(f"</{top}>")

    return {"text": "".join(xml_parts)}

def format_dataset(dataset_name, config_name=None, exclude_fields=None, complex_tags=None, split="train", save_path="./tmp_datafiles/formatted_data.txt"):
    try:
        # Check for available configs
        configs = []
        try:
            configs = get_dataset_config_names(dataset_name)
            logger.info(f"Available configs for {dataset_name}: {configs}")
        except Exception as e:
            logger.warning(f"Couldn't get config names: {e}")
        
        # Load the dataset
        if configs and config_name is None:
            config_name = configs[0]
            logger.info(f"Using config: {config_name}")
            dataset = load_dataset(dataset_name, config_name)
        else:
            try:
                # Try loading with config name if provided
                if config_name:
                    dataset = load_dataset(dataset_name, config_name)
                else:
                    dataset = load_dataset(dataset_name)
            except ValueError as ve:
                if "Config name is missing" in str(ve) and configs:
                    # If error mentions missing config and we have configs available
                    config_name = configs[0]
                    logger.info(f"Using config: {config_name}")
                    dataset = load_dataset(dataset_name, config_name)
                else:
                    raise
        
        # Handle the dataset
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available splits: {available_splits}")
                split = available_splits[0]
                logger.info(f"Using '{split}' instead.")
            
            ds_split = dataset[split]
        else:
            ds_split = dataset
        
        logger.info(f"Loaded {dataset_name} with {len(ds_split)} {split} samples")

        # Get a sample for inspection
        sample_example = ds_split[0]
        logger.info("Sample example from dataset:")
        for k, v in sample_example.items():
            if isinstance(v, str):
                logger.info(f"  {k}: {v[:100]}...")
            else:
                logger.info(f"  {k}: {v}")
        
        # Find text fields in the sample
        text_paths = find_text_fields(sample_example, exclude_fields=set(exclude_fields or []))
        logger.info(f"Detected text paths: {text_paths}")
        
        if not text_paths:
            logger.warning("No text fields detected in the dataset!")
            return None

        # Process all examples
        formatted_examples = []
        for i, example in enumerate(ds_split):
            if i % 1000 == 0:
                logger.info(f"Processing example {i}/{len(ds_split)}")
                
            result = convert_example_to_xml(
                example,
                text_paths,
                exclude_fields=set(exclude_fields or []),
                complex_tags=set(complex_tags or [])
            )
            
            if result["text"].strip():
                formatted_examples.append(result["text"])
        
        # Save to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for ex in formatted_examples:
                f.write(ex.strip() + "\n")

        logger.info(f"Saved {len(formatted_examples)} formatted examples to {save_path}")
        
        # Print a sample of the formatted data
        if formatted_examples:
            logger.info(f"Sample formatted example: {formatted_examples[0]}")
        
        return save_path
    
    except Exception as e:
        logger.error(f"Error processing dataset: {e}", exc_info=True)
        return None

def main():
    # Get the dataset name
    hf_dataset_name = input("Enter the Hugging Face dataset name (e.g., 'squad'): ")
    
    # Check for available configs
    configs = []
    try:
        configs = get_dataset_config_names(hf_dataset_name)
        if configs:
            print(f"Available configs for {hf_dataset_name}:")
            for i, config in enumerate(configs):
                print(f"  {i+1}. {config}")
            
            config_choice = input(f"Select config (1-{len(configs)}) or press Enter for default: ")
            if config_choice.strip() and config_choice.isdigit() and 1 <= int(config_choice) <= len(configs):
                config_name = configs[int(config_choice) - 1]
            else:
                config_name = configs[0]
        else:
            config_name = None
    except Exception as e:
        print(f"Error checking configs: {e}")
        config_name = None
    
    # Define standard tag mappings for different dataset types
    dataset_mappings = {
        "squad": {
            "exclude_fields": ["id", "title", "answer_start"],
            "complex_tags": ["qa"]
        },
        "glue": {
            "exclude_fields": ["idx"],
            "complex_tags": []
        },
        "wiki": {
            "exclude_fields": ["id", "url", "title"],
            "complex_tags": ["section"]
        },
        "web_questions": {
            "exclude_fields": ["id", "url"],
            "complex_tags": ["qa"]
        },
        "default": {
            "exclude_fields": ["id", "title", "metadata"],
            "complex_tags": ["dialogue", "qa", "section"]
        }
    }
    
    # Choose the appropriate mapping based on the dataset name
    mapping = dataset_mappings.get("default")
    for key in dataset_mappings:
        if key in hf_dataset_name.lower():
            mapping = dataset_mappings[key]
            break
    
    # Load and format the dataset
    output_dir = "./datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_basename = hf_dataset_name.replace('/', '_')
    if config_name:
        dataset_basename += f"_{config_name}"
    
    formatted_dataset_path = format_dataset(
        dataset_name=hf_dataset_name,
        config_name=config_name,
        exclude_fields=mapping["exclude_fields"],
        complex_tags=mapping["complex_tags"],
        save_path=f"{output_dir}/{dataset_basename}_formatted.txt"
    )
    
    if formatted_dataset_path:
        # Display formatted dataset sample
        try:
            with open(formatted_dataset_path, "r", encoding="utf-8") as f:
                sample = f.readline()
                print(f"\nSample formatted dataset entry: {sample.strip()}")
            
            print(f"\nFormatted dataset saved to: {formatted_dataset_path}")
            print(f"Total examples processed: {sum(1 for _ in open(formatted_dataset_path, 'r', encoding='utf-8'))}")
        except Exception as e:
            print(f"Error reading output file: {e}")
    else:
        print("Failed to process dataset.")

if __name__ == "__main__":
    main()