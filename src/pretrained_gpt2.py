#!/usr/bin/env python3
"""
Using pretrained GPT-2 with effective prompting
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer
from threading import Thread

class PretrainedGPT2:
    def __init__(self, model_size="gpt2-medium"):
        """
        Initialize the GPT-2 text generator with a pretrained model
        
        Args:
            model_size: Size of the model to use ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading pretrained model: {model_size}")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        self.model = GPT2LMHeadModel.from_pretrained(model_size)
        self.model.to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def generate_streaming_text(self, prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95):
        """
        Generate text in a streaming fashion, yielding tokens as they're generated
        
        Args:
            prompt: The input text to continue
            max_length: Maximum length of the generated text (including prompt)
            temperature: Controls randomness (higher = more random)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token selection
            
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
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        generated_text = prompt
        for token in streamer:
            generated_text += token
            print(token, end="", flush=True)
        
        return generated_text

# Example usage
if __name__ == "__main__":
    # Initialize with desired model size
    # Options: "gpt2" (small), "gpt2-medium", "gpt2-large", "gpt2-xl"
    generator = PretrainedGPT2(model_size="gpt2-medium")
    
    # Example prompt about the brain (to match your interests)
    prompt = "The brain processes information through neural networks, which"
    
    print("\n=== Example 1: Standard Generation ===")
    print(f"Prompt: {prompt}")
    
    texts = generator.generate_text(
        prompt=prompt,
        max_length=200,
        temperature=0.7,
        num_return_sequences=1
    )
    
    print("\nGenerated text:")
    print(texts[0])
    
    # Example with streaming
    prompt = "Neural networks can be used to solve complex problems such as"
    
    print("\n\n=== Example 2: Streaming Generation ===")
    print(f"Prompt: {prompt}")
    print("Generated text (streaming):")
    
    full_text = generator.generate_streaming_text(
        prompt=prompt,
        max_length=200,
        temperature=0.7
    )
    
    print("\n\nFull generated text:")
    print(full_text)