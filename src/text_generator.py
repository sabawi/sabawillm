#!/usr/bin/env python3
"""
Enhanced text generator for fine-tuned GPT-2 model with system prompts,
context management, multiple response options, and post-processing
"""

import argparse
import torch
import re
import os
import json
import traceback
from datetime import datetime
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
        
        # Default context
        self.conversation_history = []
        self.selected_responses = []
        self.max_context_length = 1024  # Maximum tokens for context window
        
        # Debug mode
        self.debug = False

    def _truncate_input_for_context(self, text, context_length):
        """
        Truncate the input text to fit within context_length
        
        Args:
            text: Input text to truncate
            context_length: Maximum token length
            
        Returns:
            Truncated text
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= context_length:
            return text
        
        # Truncate tokens to fit context_length
        truncated_tokens = tokens[:context_length]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        # Try to end at a reasonable punctuation
        last_period = truncated_text.rfind('.')
        last_question = truncated_text.rfind('?')
        last_exclamation = truncated_text.rfind('!')
        
        # Find the latest punctuation mark
        last_punct = max(last_period, last_question, last_exclamation)
        
        # If we found a punctuation mark that's not too far from the end, truncate there
        if last_punct > 0 and len(truncated_text) - last_punct < 50:
            return truncated_text[:last_punct+1]
        
        return truncated_text

    def _prepare_input_with_context(self, system_prompt, context, prompt, context_length=512):
        """
        Prepare the input text with system prompt, context, and user prompt
        with current prompt prioritized over history
        
        Args:
            system_prompt: Instructions for the model
            context: Previous conversation or additional context
            prompt: The current user prompt
            context_length: Maximum number of tokens to include in context
            
        Returns:
            Combined input text
        """
        # First, tokenize the essential parts to know how much space they'll take
        current_prompt = f"User: {prompt}\nAI:"
        current_prompt_tokens = len(self.tokenizer.encode(current_prompt))
        
        system_prompt_tokens = 0
        if system_prompt:
            system_prompt_text = f"System: {system_prompt}"
            system_prompt_tokens = len(self.tokenizer.encode(system_prompt_text))
        
        # Calculate how much space is left for history
        essential_tokens = current_prompt_tokens + system_prompt_tokens
        history_budget = max(0, context_length - essential_tokens)
        
        # Start building the final input
        components = []
        
        # Add system prompt if provided
        if system_prompt:
            components.append(system_prompt_text)
        
        # Add explicit context if provided and there's room
        if context and history_budget > 0:
            context_tokens = self.tokenizer.encode(f"Context: {context}")
            if len(context_tokens) <= history_budget:
                components.append(f"Context: {context}")
                history_budget -= len(context_tokens)
            elif history_budget > 20:  # Only add truncated context if there's reasonable space
                truncated_context = self._truncate_input_for_context(f"Context: {context}", history_budget)
                components.append(truncated_context)
                history_budget = 0
        
        # Prepare conversation history with sliding window if needed
        if self.conversation_history and history_budget > 0:
            # We'll add conversation history items until we approach the limit
            history_items = []
            history_header = "Previous conversation:"
            history_header_tokens = len(self.tokenizer.encode(history_header))
            
            # Adjust budget for the header
            if history_budget > history_header_tokens:
                history_budget -= history_header_tokens
            else:
                history_budget = 0
            
            # Start from most recent and work backwards, but SKIP the current prompt if it's there
            skip_count = 0
            
            # Check if the most recent items in history match the current prompt
            if len(self.conversation_history) >= 1 and prompt in self.conversation_history[-1]:
                skip_count = 1
            
            for entry in reversed(self.conversation_history[:-skip_count if skip_count > 0 else None]):
                entry_tokens = self.tokenizer.encode(entry)
                
                # If adding this entry would exceed our budget, stop
                if len(entry_tokens) > history_budget:
                    break
                
                # Otherwise, add it to our history
                history_items.insert(0, entry)
                history_budget -= len(entry_tokens)
            
            if history_items:
                components.append(history_header)
                components.extend(history_items)
        
        # Always add the current prompt at the end
        components.append(current_prompt)
        
        # Join all components with newlines between sections
        formatted_input = "\n\n".join(components)
        
        # Check the token length
        token_length = len(self.tokenizer.encode(formatted_input))
        if token_length > context_length:
            print(f"Warning: Input exceeds context length ({token_length} > {context_length} tokens). Some context has been dropped to preserve the current prompt.")
        
        if self.debug:
            # Print first and last parts of input to help with debugging
            print("\n--- INPUT PREVIEW ---")
            print("FIRST 100 CHARS:", formatted_input[:100].replace('\n', '\\n'))
            print("LAST 100 CHARS:", formatted_input[-100:].replace('\n', '\\n'))
            print("-------------------\n")
        
        return formatted_input

    def _post_process_response(self, text):
        """
        Clean up and improve the generated response
        
        Args:
            text: Raw generated text
            
        Returns:
            Processed text
        """
        # Extract just the AI's response if the format includes prompt
        if "AI:" in text:
            response_parts = text.split("AI:", 1)
            if len(response_parts) > 1:
                text = response_parts[1].strip()
        
        # Remove any trailing incomplete sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences and not any(sentences[-1].endswith(end) for end in ['.', '!', '?']):
            sentences = sentences[:-1]
        
        # Remove repetitive content (same sentence appearing multiple times)
        unique_sentences = []
        for sentence in sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
            else:
                # If we have more than 3 sentences and find repetition, stop processing
                if len(unique_sentences) > 3:
                    break
        
        # Find a clean section break if appropriate
        final_text = " ".join(unique_sentences)
        section_breaks = [
            r'###\s+\d+\.\s+\*\*.*?\*\*',  # Markdown section like "### 1. **Title**"
            r'\d+\.\s+\*\*.*?\*\*',         # Numbered section like "1. **Title**"
            r'\*\*\d+\.\s+.*?\*\*',         # Bold numbered section like "**1. Title**"
            r'\n\s*\n'                      # Double line break
        ]
        
        # Try to find a good breaking point if the text is long
        if len(final_text) > 300:
            for pattern in section_breaks:
                matches = list(re.finditer(pattern, final_text))
                if matches and len(matches) > 1:
                    # Find the latest complete section
                    last_complete_section = matches[-2].start()
                    # Only break if we've got a substantial amount of text already
                    if last_complete_section > 250:
                        final_text = final_text[:last_complete_section]
                        break
        
        return final_text.strip()

    def _safe_generate(self, input_ids, **generation_kwargs):
        """
        Safely attempt to generate text with error handling for CUDA issues
        
        Args:
            input_ids: Encoded input tokens
            generation_kwargs: Parameters for generation
            
        Returns:
            Generated output or None on error
        """
        try:
            # Important: When input_ids length > max_length, we need to use max_new_tokens instead
            input_length = input_ids.shape[1]
            
            # Check if we need to switch to max_new_tokens
            if 'max_length' in generation_kwargs and input_length >= generation_kwargs['max_length']:
                # Calculate max_new_tokens instead
                max_new_tokens = 100  # Default fallback
                if 'max_length' in generation_kwargs:
                    # Use the difference, or a minimum of 20 tokens
                    max_new_tokens = max(20, generation_kwargs['max_length'] - input_length)
                
                # Replace max_length with max_new_tokens
                del generation_kwargs['max_length']
                generation_kwargs['max_new_tokens'] = max_new_tokens
                print(f"Switched to max_new_tokens={max_new_tokens} because input_length={input_length}")
            
            # Try with the updated parameters
            return self.model.generate(input_ids, **generation_kwargs)
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg or "out of memory" in error_msg:
                print("\nCUDA error detected. Trying with reduced parameters...")
                # Try again with reduced parameters
                new_kwargs = generation_kwargs.copy()
                
                # Remove max_length and set a small max_new_tokens
                if 'max_length' in new_kwargs:
                    del new_kwargs['max_length']
                new_kwargs['max_new_tokens'] = 50
                
                # Move to CPU if needed
                input_ids_cpu = input_ids.to('cpu')
                model_cpu = self.model.to('cpu')
                
                try:
                    outputs = model_cpu.generate(input_ids_cpu, **new_kwargs)
                    # Move model back to original device
                    self.model.to(self.device)
                    return outputs
                except Exception as e2:
                    print(f"Second attempt failed: {e2}")
                    # Move model back to original device
                    self.model.to(self.device)
                    return None
            else:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                return None

    def generate_multiple_responses(self, prompt, system_prompt=None, context=None, num_responses=3, 
                                   context_length=512, max_length=200, temperature=0.1, top_k=50, top_p=0.95):
        """
        Generate multiple responses for the same prompt
        
        Args:
            prompt: The input text to continue
            system_prompt: Instructions for the model
            context: Additional context or history
            num_responses: Number of different responses to generate
            context_length: Length of context to consider
            max_length: Maximum length of the generated text
            temperature: Controls randomness (higher = more random)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token selection
            
        Returns:
            List of generated text sequences
        """
        responses = []
        
        # Ensure context_length isn't too large
        context_length = min(context_length, self.max_context_length)
        
        # Prepare the combined input
        combined_input = self._prepare_input_with_context(
            system_prompt, context, prompt, context_length
        )
        
        # Print token count for debugging
        input_tokens = self.tokenizer.encode(combined_input)
        print(f"Input token count: {len(input_tokens)}")
        
        for i in range(num_responses):
            # Use slightly different parameters for each response to ensure variety
            temp_adjust = temperature * (0.8 + (i * 0.4))  # Vary temperature a bit
            top_p_adjust = min(top_p * (0.9 + (i * 0.1)), 0.99)
            
            # Encode the prompt
            input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.device)
            input_length = len(input_ids[0])
            
            # Limit max_length to avoid CUDA errors, but ensure it's larger than input length
            adjusted_max_length = max(input_length + 1, min(input_length + max_length, self.max_context_length))
            
            # Generate text with error handling
            with torch.no_grad():
                outputs = self._safe_generate(
                    input_ids,
                    max_length=adjusted_max_length,
                    temperature=temp_adjust,
                    top_k=top_k,
                    top_p=top_p_adjust,
                    do_sample=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # If generation failed, add a placeholder
            if outputs is None:
                responses.append("[Generation failed due to resource constraints]")
                continue
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the model's response
            response = generated_text[len(combined_input):].strip()
            
            # Post-process the response
            clean_response = self._post_process_response(response)
            responses.append(clean_response)
        
        return responses

    def generate_text(self, prompt, system_prompt=None, context=None, context_length=512,
                     max_length=200, temperature=0.1, top_k=50, top_p=0.95, num_return_sequences=1):
        """
        Generate text based on a prompt
        
        Args:
            prompt: The input text to continue
            system_prompt: Instructions for the model
            context: Additional context or history
            context_length: Length of context to consider
            max_length: Maximum length of the generated text
            temperature: Controls randomness (higher = more random)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token selection
            num_return_sequences: Number of text sequences to generate
            
        Returns:
            List of generated text sequences
        """
        # Ensure context_length isn't too large
        context_length = min(context_length, self.max_context_length)
        
        # Prepare the combined input
        combined_input = self._prepare_input_with_context(
            system_prompt, context, prompt, context_length
        )
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.device)
        input_length = len(input_ids[0])
        
        # Limit max_length to avoid CUDA errors, but ensure it's larger than input length
        adjusted_max_length = max(input_length + 1, min(input_length + max_length, self.max_context_length))
        
        # Generate text with error handling
        with torch.no_grad():
            outputs = self._safe_generate(
                input_ids,
                max_length=adjusted_max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Handle generation failure
        if outputs is None:
            return ["[Generation failed due to resource constraints]"] * num_return_sequences
        
        # Decode and process the generated text
        generated_texts = []
        for output in outputs:
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)
            response = full_text[len(combined_input):].strip()
            clean_response = self._post_process_response(response)
            generated_texts.append(clean_response)
        
        return generated_texts

    def generate_streaming_text(self, prompt, system_prompt=None, context=None, context_length=512,
                               max_length=200, temperature=0.1, top_k=50, top_p=0.95, callback=None):
        """
        Generate text in a streaming fashion, yielding tokens as they're generated
        
        Args:
            prompt: The input text to continue
            system_prompt: Instructions for the model
            context: Additional context or history
            context_length: Length of context to consider
            max_length: Maximum length of the generated text
            temperature: Controls randomness (higher = more random)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token selection
            callback: Optional callback function to receive the tokens
            
        Returns:
            The full generated text
        """
        # Ensure context_length isn't too large
        context_length = min(context_length, self.max_context_length)
        
        # Prepare the combined input
        combined_input = self._prepare_input_with_context(
            system_prompt, context, prompt, context_length
        )
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.device)
        input_length = len(input_ids[0])
        
        # Check if we need to use max_new_tokens instead of max_length
        generation_kwargs = {}
        if input_length >= max_length:
            # Use max_new_tokens instead
            generation_kwargs["max_new_tokens"] = 100
            print(f"Using max_new_tokens={generation_kwargs['max_new_tokens']} because input_length={input_length} >= max_length={max_length}")
        else:
            # Limit max_length to avoid CUDA errors
            adjusted_max_length = max(input_length + 1, min(input_length + max_length, self.max_context_length))
            generation_kwargs["max_length"] = adjusted_max_length
        
        # Initialize the streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Add common generation parameters
        generation_kwargs.update({
            "input_ids": input_ids,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id
        })
        
        # Use a try-except block to catch errors in the thread
        def generate_with_error_handling():
            try:
                self.model.generate(**generation_kwargs)
            except Exception as e:
                print(f"\nError in generation thread: {e}")
                # Signal error to streamer
                if hasattr(streamer, 'end'):
                    streamer.end()
        
        thread = Thread(target=generate_with_error_handling)
        thread.start()
        
        # Stream the output
        generated_text = ""
        first_token = True
        try:
            for token in streamer:
                generated_text += token
                if callback:
                    callback(token)
                else:
                    if first_token:
                        first_token = False
                        print("\nAI: ", end="", flush=True)
                        
                    print(token, end="", flush=True)
        except Exception as e:
            print(f"\nError in streaming: {e}")
            generated_text += " [Generation interrupted due to an error]"
        
        # Post-process the streamed response
        clean_response = self._post_process_response(generated_text)
        
        # Update conversation history
        self.conversation_history.append(f"User: {prompt}")
        self.conversation_history.append(f"AI: {clean_response}")
        
        return combined_input + clean_response

    def update_conversation_history(self, user_input, model_response):
        """Add a conversation exchange to the history"""
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"AI: {model_response}")
        
        # Check if we need to trim history to prevent it from growing too large
        self._trim_conversation_history()
    
    def _trim_conversation_history(self, max_entries=10):
        """Trim conversation history to prevent it from growing too large"""
        if len(self.conversation_history) > max_entries:
            # Keep only the most recent entries
            self.conversation_history = self.conversation_history[-max_entries:]
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        self.selected_responses = []

    def enable_debug(self, enabled=True):
        """Enable or disable debug mode"""
        self.debug = enabled
        return self


def interactive_with_options(generator, args):
    """Interactive mode with multiple response options"""
    print(f"Interactive Mode - Generating {args.num_responses} response options")
    print("Type 'exit' to quit, 'clear' to start a new conversation, 'debug' to toggle debug mode")
    
    context = args.context
    system_prompt = args.system_prompt
    
    conversation_log = []
    
    while True:
        user_input = input("\nEnter your prompt: ")
        if user_input.lower() == "exit":
            # Save conversation log if there was a conversation
            if conversation_log and args.save_conversations:
                log_filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_filename, 'w') as f:
                    json.dump(conversation_log, f, indent=2)
                print(f"Conversation saved to {log_filename}")
            break
        
        if user_input.lower() == "clear":
            generator.clear_conversation_history()
            context = args.context
            print("Conversation history cleared.")
            conversation_log = []
            continue
        
        if user_input.lower() == "debug":
            generator.debug = not generator.debug
            print(f"Debug mode {'enabled' if generator.debug else 'disabled'}")
            continue
        
        # Generate multiple responses
        print("\nGenerating options...\n")
        responses = generator.generate_multiple_responses(
            prompt=user_input,
            system_prompt=system_prompt,
            context=context,
            num_responses=args.num_responses,
            context_length=args.context_length,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Display options to user
        for i, resp in enumerate(responses):
            print(f"\n--- Option {i+1} ---\n")
            print(resp)
            print("\n" + "-" * 30)
        
        # Let user choose
        while True:
            choice = input(f"\nSelect your preferred option (1-{args.num_responses}) or 'n' for new options: ")
            if choice.lower() == 'n':
                print("\nGenerating new options...\n")
                responses = generator.generate_multiple_responses(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    context=context,
                    num_responses=args.num_responses,
                    context_length=args.context_length,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                for i, resp in enumerate(responses):
                    print(f"\n--- Option {i+1} ---\n")
                    print(resp)
                    print("\n" + "-" * 30)
            elif choice.isdigit() and 1 <= int(choice) <= args.num_responses:
                selected = int(choice) - 1
                selected_response = responses[selected]
                
                # Update conversation history
                generator.update_conversation_history(user_input, selected_response)
                
                # Update context from conversation history
                history_text = ""
                for entry in generator.conversation_history[-6:]:  # Last 3 exchanges
                    history_text += entry + "\n"
                context = history_text.strip()
                
                # Add to conversation log
                conversation_log.append({
                    "user": user_input,
                    "model_responses": responses,
                    "selected_response": selected,
                    "timestamp": datetime.now().isoformat()
                })
                
                print(f"\nYou selected option {int(choice)}. Response added to conversation history.")
                break
            else:
                print(f"Please enter a number between 1 and {args.num_responses}, or 'n'.")

def interactive_streaming(generator, args):
    """Interactive mode with streaming responses"""
    print("Interactive Streaming Mode")
    print("Type 'exit' to quit, 'clear' to start a new conversation, 'debug' to toggle debug mode")
    
    context = args.context
    system_prompt = args.system_prompt
    
    while True:
        user_input = input("\nEnter a prompt (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "clear":
            generator.clear_conversation_history()
            context = args.context
            print("Conversation history cleared.")
            continue
        
        if user_input.lower() == "debug":
            generator.debug = not generator.debug
            print(f"Debug mode {'enabled' if generator.debug else 'disabled'}")
            continue
        
        # Generate streaming response
        generator.generate_streaming_text(
            prompt=user_input,
            system_prompt=system_prompt,
            context=context,
            context_length=args.context_length,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Update context from conversation history
        history_text = ""
        for entry in generator.conversation_history[-6:]:  # Last 3 exchanges
            history_text += entry + "\n"
        context = history_text.strip()
        
        print("\n" + "=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Enhanced text generator using a fine-tuned GPT-2 model")
    parser.add_argument("--model_path", default="./gpt2-finetuned", help="Path to the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling (lower = more focused)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--stream", action="store_true", help="Stream the output token by token")
    parser.add_argument("--system_prompt", type=str, 
                        default="You are a helpful, knowledgeable assistant that provides accurate and detailed information without repetition.",
                        help="System instructions for the model")
    parser.add_argument("--context_length", type=int, default=512, help="Length of context to consider")
    parser.add_argument("--context", type=str, default=None, help="Additional context to provide to the model")
    parser.add_argument("--num_responses", type=int, default=3, help="Number of response options to generate")
    parser.add_argument("--save_conversations", action="store_true", help="Save conversations to JSON files")
    parser.add_argument("--options", action="store_true", help="Use multiple options mode even with streaming")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to show input previews")
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    # Initialize generator
    generator = GPT2Generator(args.model_path)
    if args.debug:
        generator.enable_debug(True)
    
    # Choose interaction mode
    if args.stream and not args.options and args.num_responses <= 1:
        interactive_streaming(generator, args)
    else:
        interactive_with_options(generator, args)
        

if __name__ == "__main__":
    main()