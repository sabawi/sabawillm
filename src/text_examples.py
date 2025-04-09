#!/usr/bin/env python3
"""
Example usage of the GPT-2 text generator
"""

from text_generator import GPT2Generator
import time

def main():
    # Initialize the generator with absolute path
    import os
    model_path = os.path.abspath("./gpt2-finetuned")
    
    # You can also use a relative path from the script's location
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(script_dir, "../gpt2-finetuned")
    
    generator = GPT2Generator(model_path=model_path)
    
    # Example 1: Basic text generation
    prompt = "The brain is responsible for"
    print(f"\n=== Example 1: Basic Text Generation ===")
    print(f"Prompt: {prompt}")
    
    generated_texts = generator.generate_text(
        prompt=prompt,
        max_length=150,
        temperature=0.4,
        top_k=30,
        top_p=0.95,
        num_return_sequences=2
    )
    
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated text {i+1}:")
        print(text)
    
    # Example 2: Streaming text generation with callback
    prompt = "Neural networks can be used to"
    print(f"\n\n=== Example 2: Streaming Text Generation ===")
    print(f"Prompt: {prompt}")
    print("Generated text (streaming):")
    
    # Custom callback function
    def token_callback(token):
        print(token, end="", flush=True)
        time.sleep(0.05)  # Add slight delay to simulate typing
    
    # Generate with streaming
    full_text = generator.generate_streaming_text(
        prompt=prompt,
        max_length=150,
        temperature=0.4,
        top_k=30,
        top_p=0.95,
        callback=token_callback
    )
    
    print("\n\nFull generated text:")
    print(full_text)

if __name__ == "__main__":
    main()