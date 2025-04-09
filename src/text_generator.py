#!/usr/bin/env python3
"""
Text generator for fine-tuned GPT-2 model
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer
from threading import Thread
import time

class GPT2Generator:
    def __init__(self, model_path="./gpt2-finetuned"):
        """
        Initialize the GPT-2 text generator
        
        Args:
            model_path: Path to the fine-tuned model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Convert to absolute path if it's a relative path
        import os
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        print(f"Loading model from: {model_path}")
        
        # Check if path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
            
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def generate_text(self, prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1):
        """
        Generate text based on a prompt
        
        Args:
            prompt: The input text to continue
            max_length: Maximum length of the generated text (including prompt)
            temperature: Controls randomness (higher = more random)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token selection
            num_return_sequences: Number of text sequences to generate
            
        Returns:
            List of generated text sequences
        """
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and return the generated text
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts

    def generate_streaming_text(self, prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95, callback=None):
        """
        Generate text in a streaming fashion, yielding tokens as they're generated
        
        Args:
            prompt: The input text to continue
            max_length: Maximum length of the generated text (including prompt)
            temperature: Controls randomness (higher = more random)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token selection
            callback: Optional callback function to receive the tokens
            
        Returns:
            The full generated text
        """
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Initialize the streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generate text in a separate thread
        generation_kwargs = {
            "input_ids": input_ids,
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": 1.2,  # Add this parameter
            "no_repeat_ngram_size": 3,  # Add this parameter
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        generated_text = prompt
        first_token = True
        for token in streamer:
            generated_text += token
            if callback:
                callback(token)
            else:
                if first_token:
                    first_token = False
                    print("\nAI: "+prompt, end="", flush=True)
                    
                print(token, end="", flush=True)
        
        return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT-2 model")
    parser.add_argument("--model_path", default="./gpt2-finetuned", help="Path to the fine-tuned model")
    # parser.add_argument("--prompt", required=True, help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--stream", action="store_true", help="Stream the output token by token")
    args = parser.parse_args()
    
    # Initialize generator
    generator = GPT2Generator(args.model_path)
    
    while True:
        # Get user input for prompt
        # print(f"Prompt: {args.prompt}")
        prompt = input("\nEnter a prompt (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        
        if args.stream:
            # print("Streaming output:")
            generator.generate_streaming_text(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            # print("\n\nFull generated text:", generated_text)
        else:
            generated_texts = generator.generate_text(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            for i, text in enumerate(generated_texts):
                print(f"\nGenerated text {i+1}:")
                print(text)
        print("\n====================================================================================================\n\n")
if __name__ == "__main__":
    main()