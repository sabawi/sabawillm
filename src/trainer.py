# train_gpt2_custom_json.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import sys
# import transformers
# print(transformers.__version__) 

from datasets import Dataset
import torch
import json

# Configuration
MODEL_NAME = "gpt2"  # Can use "gpt2-medium", "gpt2-large", etc.
TRAIN_FILE = "../NeuralNetworkStarter/data/clean/alsbrain.json"  # Path to your JSON file
OUTPUT_DIR = "./gpt2-finetuned"
BLOCK_SIZE = 512  # Maximum sequence length
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# Load tokenizer and model
# When loading the model, try this:
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
config = GPT2Config.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel(config)

# Add special tokens if needed
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Load dataset from JSON file
def load_dataset(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = json.load(f)  # Load JSON array
        
        print(f"Loaded {len(texts)} texts from {file_path}")
        
        # Check if texts are strings
        if len(texts) > 0:
            if not isinstance(texts[0], str):
                # If texts are dictionaries with a 'text' field
                if isinstance(texts[0], dict) and 'text' in texts[0]:
                    texts = [item['text'] for item in texts]
                    print("Extracted text field from dictionary items")
                else:
                    print(f"Warning: Unexpected data format. First item type: {type(texts[0])}")
                    print(f"First item preview: {str(texts[0])[:100]}...")
        
        # Split into train and validation
        split = int(0.9 * len(texts))
        train_texts = texts[:split]
        val_texts = texts[split:]
        
        print(f"Split into {len(train_texts)} training and {len(val_texts)} validation texts")
        
        return train_texts, val_texts
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts and split into chunks of BLOCK_SIZE
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Print for debugging
    print(f"Total tokens before chunking: {total_length}")
    
    # If total_length is smaller than BLOCK_SIZE, we need to handle it differently
    if total_length < BLOCK_SIZE:
        # Just use what we have without truncation
        result = {k: [t] for k, t in concatenated_examples.items()}
    else:
        # Otherwise, proceed with chunking as before
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
    
    result["labels"] = result["input_ids"].copy()
    print(f"Created {len(result['input_ids'])} text chunks")
    return result

# Prepare datasets
train_texts, val_texts = load_dataset(TRAIN_FILE)

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)
tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

lm_train_dataset = tokenized_train.map(
    group_texts,
    batched=True,
)
lm_val_dataset = tokenized_val.map(
    group_texts,
    batched=True,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling
)

# After the mapping operations, add these checks
print(f"Train dataset size: {len(lm_train_dataset)}")
print(f"Validation dataset size: {len(lm_val_dataset)}")

# Also print a sample from the dataset to verify its structure
if len(lm_train_dataset) > 0:
    print("Sample from training dataset:", lm_train_dataset[0])
else:
    print("Training dataset is empty.")
    sys.exit(1)


# Training arguments
# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    learning_rate=LEARNING_RATE,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_val_dataset,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete! Model saved to {OUTPUT_DIR}")