#!/usr/bin/env python3
"""
Enhanced text generator for fine-tuned GPT-2 model with system prompts,
context management, multiple response options, post-processing, and special token handling
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
    "think_opne": "<think>",
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
        
        # Check and ensure all special tokens are properly loaded
        self._ensure_special_tokens()
        
        # Default context
        self.conversation_history = []
        self.selected_responses = []
        self.max_context_length = 1024  # Maximum tokens for context window
        
        # Debug mode
        self.debug = False

    def _ensure_special_tokens(self):
        """
        Ensure all special tokens are in the tokenizer
        If they were properly added during fine-tuning, they should already be there
        """
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Check if all XML tags are in the tokenizer's vocabulary
        special_tokens_list = list(DEFAULT_SPECIAL_TOKENS.values())
        missing_tokens = [token for token in special_tokens_list 
                          if token not in self.tokenizer.get_vocab()]
        
        # Add any missing special tokens
        if missing_tokens:
            print(f"Adding {len(missing_tokens)} missing special tokens to the tokenizer")
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            print("All special tokens are already in the tokenizer vocabulary")

    def _preprocess_text_with_tags(self, text):
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
        with improved contextual relevance and topic change detection
        
        Args:
            system_prompt: Instructions for the model
            context: Previous conversation or additional context
            prompt: The current user prompt
            context_length: Maximum number of tokens to include in context
            
        Returns:
            Combined input text
        """
        # Check if this is a topic change by comparing the current prompt to the previous context
        is_topic_change = False
        if self.conversation_history and len(self.conversation_history) >= 2:
            # Get the last question from history
            last_question = None
            for entry in reversed(self.conversation_history):
                if entry.startswith("<question>"):
                    last_question = entry.replace("<question>", "").replace("</question>", "").strip()
                    break
            
            if last_question:
                # Compare topics using simple keyword overlap or question type
                prompt_words = set(re.sub(r'[^\w\s]', '', prompt.lower()).split())
                last_words = set(re.sub(r'[^\w\s]', '', last_question.lower()).split())
                
                # Check for common words (excluding stopwords)
                stopwords = {'the', 'a', 'an', 'and', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                            'to', 'for', 'of', 'with', 'by', 'about', 'like', 'from', 'but', 'or', 
                            'as', 'what', 'when', 'where', 'who', 'why', 'how', 'which', 'than', 'if'}
                
                meaningful_prompt_words = prompt_words - stopwords
                meaningful_last_words = last_words - stopwords
                
                # Calculate word overlap
                if len(meaningful_prompt_words) > 0 and len(meaningful_last_words) > 0:
                    overlap = len(meaningful_prompt_words.intersection(meaningful_last_words))
                    # If less than 20% word overlap, consider it a topic change
                    if overlap / len(meaningful_prompt_words) < 0.2 and overlap / len(meaningful_last_words) < 0.2:
                        is_topic_change = True
                
                # Also check for question type changes (what vs where vs how etc.)
                prompt_question_words = {'what', 'where', 'when', 'why', 'how', 'who', 'which'}.intersection(prompt_words)
                last_question_words = {'what', 'where', 'when', 'why', 'how', 'who', 'which'}.intersection(last_words)
                
                if prompt_question_words and last_question_words:
                    if not prompt_question_words.intersection(last_question_words):
                        # Different question types suggest topic change
                        is_topic_change = True
        
        # First, tokenize the essential parts to know how much space they'll take
        current_prompt = f"<question>{prompt}</question>\n<answer>"
        current_prompt_tokens = len(self.tokenizer.encode(current_prompt))
        
        system_prompt_tokens = 0
        if system_prompt:
            system_prompt_text = f"<instruction>{system_prompt}</instruction>"
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
        if context and history_budget > 0 and not is_topic_change:
            context_tokens = self.tokenizer.encode(f"<context>{context}</context>")
            if len(context_tokens) <= history_budget:
                components.append(f"<context>{context}</context>")
                history_budget -= len(context_tokens)
            elif history_budget > 20:  # Only add truncated context if there's reasonable space
                truncated_context = self._truncate_input_for_context(f"<context>{context}</context>", history_budget)
                components.append(truncated_context)
                history_budget = 0
        
        # Prepare conversation history with sliding window if needed
        if self.conversation_history and history_budget > 0 and not is_topic_change:
            # We'll add conversation history items until we approach the limit
            history_items = []
            history_header = "<dialog>"
            history_header_tokens = len(self.tokenizer.encode(history_header))
            history_footer = "</dialog>"
            history_footer_tokens = len(self.tokenizer.encode(history_footer))
            
            # Adjust budget for the header and footer
            if history_budget > (history_header_tokens + history_footer_tokens):
                history_budget -= (history_header_tokens + history_footer_tokens)
            else:
                history_budget = 0
            
            # Start from most recent and work backwards, but SKIP the current prompt if it's there
            skip_count = 0
            
            # Check if the most recent items in history match the current prompt
            if len(self.conversation_history) >= 1 and prompt in self.conversation_history[-1]:
                skip_count = 1
            
            # Focus on recent and relevant conversation
            # Limit how far back we go based on relevance
            max_history_pairs = 2  # Just use the most recent Q&A exchange by default
            
            for entry in reversed(self.conversation_history[:-skip_count if skip_count > 0 else None]):
                entry_tokens = self.tokenizer.encode(entry)
                
                # If adding this entry would exceed our budget, stop
                if len(entry_tokens) > history_budget:
                    break
                
                # Otherwise, add it to our history
                history_items.insert(0, entry)
                history_budget -= len(entry_tokens)
                
                # Only add a limited amount of history
                if len(history_items) >= max_history_pairs * 2:  # 2 items per Q&A pair
                    break
            
            if history_items:
                components.append(history_header)
                components.extend(history_items)
                components.append(history_footer)
        
        # Topic change indicator - explicitly inform the model that topic has changed
        if is_topic_change:
            components.append("<context>New topic: The previous conversation is not relevant to this new question.</context>")
        
        # Always add the current prompt at the end
        components.append(current_prompt)
        
        # Join all components with newlines between sections
        formatted_input = "\n\n".join(components)
        
        # Preprocess to ensure XML tags are properly formatted
        formatted_input = self._preprocess_text_with_tags(formatted_input)
        
        # Check the token length
        token_length = len(self.tokenizer.encode(formatted_input))
        if token_length > context_length:
            print(f"Warning: Input exceeds context length ({token_length} > {context_length} tokens). Some context may be truncated.")
        
        if self.debug:
            # Print first and last parts of input to help with debugging
            print("\n--- INPUT PREVIEW ---")
            print("FORMATTED INPUT:", formatted_input.replace('\n', '\\n'))
            print("INPUT TOKEN COUNT:", token_length)
            print("TOPIC CHANGE DETECTED:", is_topic_change)
            print("-------------------\n")
        
        return formatted_input


    # Add a clear_context method to allow explicit context clearing
    def clear_context(self):
        """Clear just the context while keeping the system prompt"""
        self.conversation_history = []
        return "Context cleared. The model will not reference previous conversation."


    # 2. Modified _post_process_response to debug and better handle empty responses
    def _post_process_response(self, text):
        """
        Clean up and improve the generated response with better handling of XML tags
        
        Args:
            text: Raw generated text
            
        Returns:
            Processed text
        """
        # Debug raw output if in debug mode
        if self.debug:
            print("\n--- RAW RESPONSE ---")
            print(text.replace('\n', '\\n'))
            print("--------------------")
        
        # If empty text, return a placeholder
        if not text or text.strip() == "":
            return "[Model generated empty response]"
        
        # First, try to extract content from within answer tags
        if "<answer>" in text:
            try:
                # Extract text between <answer> and the next special tag
                start_idx = text.find("<answer>") + len("<answer>")
                
                # Find the next opening tag after <answer>
                next_tags = ["<dialog>", "<question>", "<section>", "<context>", 
                            "<instruction>", "<document>", "<text>", "<story>", 
                            "<fiction>", "<title>"]
                
                end_positions = []
                for tag in next_tags:
                    pos = text.find(tag, start_idx)
                    if pos != -1:
                        end_positions.append(pos)
                
                # Find </answer> tag position
                answer_end = text.find("</answer>", start_idx)
                if answer_end != -1:
                    end_positions.append(answer_end)
                
                # If we found any ending position, use the earliest one
                if end_positions:
                    end_idx = min(end_positions)
                    extracted_text = text[start_idx:end_idx].strip()
                else:
                    # If no ending tag found, use the rest of the text
                    extracted_text = text[start_idx:].strip()
                
                if extracted_text:
                    text = extracted_text
                    if self.debug:
                        print("Extracted content from answer tags")
            except Exception as e:
                if self.debug:
                    print(f"Error extracting answer tag content: {e}")
        
        # Clean up any remaining XML tags
        text = re.sub(r'</?[a-z_]+>', '', text).strip()
        
        # Remove any trailing incomplete sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences and not any(sentences[-1].endswith(end) for end in ['.', '!', '?']):
            sentences = sentences[:-1]
            if self.debug:
                print("Trimmed incomplete sentence")
        
        # If no complete sentences were found, return the original text
        if not sentences:
            return text
        
        # Remove repetitive content
        unique_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
            else:
                # If we have more than 3 sentences and find repetition, stop processing
                if len(unique_sentences) > 3 and self.debug:
                    print("Detected repetition, stopping early")
                if len(unique_sentences) > 3:
                    break
        
        # Join the sentences back together
        final_text = " ".join(unique_sentences)
        
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

    # 4. Modified generate_multiple_responses to better handle debugging and empty responses
    def generate_multiple_responses(self, prompt, system_prompt=None, context=None, num_responses=3, 
                                context_length=512, max_length=200, temperature=0.1, top_k=50, top_p=0.95, quiet=False):
        """
        Generate multiple responses with quiet mode support
        """
        responses = []
        
        # Enable debug temporarily for the first generation only if not in quiet mode
        old_debug = self.debug
        if num_responses > 0 and not self.debug and not quiet:
            self.debug = True
            print("Enabling debug mode for first generation to diagnose issues...")
        
        # If in quiet mode, completely disable debug
        if quiet:
            self.debug = False
        
        # Ensure context_length isn't too large
        context_length = min(context_length, self.max_context_length)
        
        # Prepare the combined input
        combined_input = self._prepare_input_with_context(
            system_prompt, context, prompt, context_length
        )
        
        # Print token count only if in debug mode
        input_tokens = self.tokenizer.encode(combined_input)
        if self.debug:
            print(f"Input token count: {len(input_tokens)}")
        
        for i in range(num_responses):
            # Use slightly different parameters for each response to ensure variety
            temp_adjust = temperature * (0.8 + (i * 0.4))  # Vary temperature a bit
            top_p_adjust = min(top_p * (0.9 + (i * 0.1)), 0.99)
            
            if i > 0:
                # Return to original debug setting after first generation
                self.debug = old_debug
            
            # Encode the prompt
            input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)  # Explicit attention mask
            input_length = len(input_ids[0])
            
            # Limit max_length to avoid CUDA errors, but ensure it's larger than input length
            adjusted_max_length = max(input_length + 1, min(input_length + max_length, self.max_context_length))
            
            # Generate text with error handling
            with torch.no_grad():
                try:
                    outputs = self._safe_generate(
                        input_ids,
                        attention_mask=attention_mask,  # Explicitly pass attention mask
                        max_length=adjusted_max_length,
                        temperature=temp_adjust,
                        top_k=top_k,
                        top_p=top_p_adjust,
                        do_sample=True,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # If generation failed, add a placeholder
                    if outputs is None:
                        responses.append("[Generation failed due to resource constraints]")
                        continue
                    
                    # Decode the generated text - don't skip special tokens
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                    
                    # For debugging the first response
                    if i == 0 and self.debug:
                        print("\n--- FULL GENERATED TEXT (First 100 chars) ---")
                        print(generated_text[:100].replace('\n', '\\n'))
                        print("--- LAST 50 CHARS ---")
                        print(generated_text[-50:].replace('\n', '\\n'))
                        
                    # Extract just the model's response
                    response = generated_text[len(combined_input):].strip()
                    
                    # Post-process the response
                    clean_response = self._post_process_response(response)
                    
                    # If we got an empty response, try to use more of the generated text
                    if not clean_response or clean_response.startswith("[Model generated"):
                        if self.debug:
                            print("Got empty response, trying alternate extraction...")
                        # Try using the full generated text and clean it
                        clean_response = self._post_process_response(generated_text)
                    
                    responses.append(clean_response)
                    
                except Exception as e:
                    if not quiet:
                        print(f"Error in generate_multiple_responses: {e}")
                    responses.append(f"[Error during generation: {str(e)[:100]}...]")
        
        # Restore original debug setting
        self.debug = old_debug
        
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
            full_text = self.tokenizer.decode(output, skip_special_tokens=False)
            response = full_text[len(combined_input):].strip()
            clean_response = self._post_process_response(response)
            generated_texts.append(clean_response)
        
        return generated_texts

    def generate_streaming_text(self, prompt, system_prompt=None, context=None, context_length=512,
                            max_length=200, temperature=0.1, top_k=50, top_p=0.95, callback=None):
        """
        Generate text in a streaming fashion with proper sentence boundary detection
        
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
        attention_mask = torch.ones_like(input_ids).to(self.device)
        input_length = len(input_ids[0])
        
        # Adjust streaming length based on user-provided max_length
        streaming_max_length = min(max_length, max(150, max_length // 2))  
        
        # Check if we need to use max_new_tokens instead of max_length
        generation_kwargs = {}
        if input_length >= streaming_max_length:
            # Use max_new_tokens instead with appropriate value
            generation_kwargs["max_new_tokens"] = min(100, max_length // 4)
            if self.debug:
                print(f"Using max_new_tokens={generation_kwargs['max_new_tokens']} because input_length={input_length} >= max_length={streaming_max_length}")
        else:
            # Limit max_length to avoid CUDA errors
            adjusted_max_length = max(input_length + 1, min(input_length + streaming_max_length, self.max_context_length))
            generation_kwargs["max_length"] = adjusted_max_length
        
        # Initialize the streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Add common generation parameters
        generation_kwargs.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 4,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
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
        
        # State tracking
        generated_text = ""
        first_token = True
        accumulated_tokens = ""
        special_tags = ["</answer>", "</dialog>", "</question>", "</section>", 
                    "</context>", "</instruction>", "</document>", "</text>", 
                    "</story>", "</fiction>", "</title>"]
        
        # Track sentences and completeness
        sentence_count = 0
        max_sentences = min(6, max(3, max_length // 75))
        sentence_buffer = ""
        complete_sentences = []
        stream_ended = False
        
        try:
            for token in streamer:
                # Add token to buffer for processing
                buffer = accumulated_tokens + token
                
                # Check for special tags
                tag_found = False
                for tag in special_tags:
                    if tag in buffer:
                        tag_pos = buffer.find(tag)
                        text_before_tag = buffer[:tag_pos]
                        text_before_tag = re.sub(r'<[^/][^>]*>', '', text_before_tag)
                        
                        if text_before_tag.strip():
                            # Add to sentence buffer
                            sentence_buffer += text_before_tag
                            
                            # Display to user
                            if callback:
                                callback(text_before_tag)
                            else:
                                if first_token:
                                    first_token = False
                                    print("\nAI: ", end="", flush=True)
                                print(text_before_tag, end="", flush=True)
                        
                        accumulated_tokens = ""
                        tag_found = True
                        stream_ended = True
                        break
                
                if stream_ended:
                    break
                    
                if not tag_found:
                    # Check for opening tags
                    if '<' in buffer and '>' in buffer:
                        # Clean opening tags
                        clean_buffer = re.sub(r'<[^/][^>]*>', '', buffer)
                        
                        if clean_buffer != buffer:
                            if clean_buffer.strip():
                                # Add to sentence buffer
                                sentence_buffer += clean_buffer
                                
                                # Display to user
                                if callback:
                                    callback(clean_buffer)
                                else:
                                    if first_token:
                                        first_token = False
                                        print("\nAI: ", end="", flush=True)
                                    print(clean_buffer, end="", flush=True)
                            accumulated_tokens = ""
                        else:
                            accumulated_tokens = buffer
                    else:
                        # Add to sentence buffer and display
                        sentence_buffer += token
                        
                        if callback:
                            callback(token)
                        else:
                            if first_token:
                                first_token = False
                                print("\nAI: ", end="", flush=True)
                            print(token, end="", flush=True)
                        
                        accumulated_tokens = ""
                        
                        # Check if we've completed a sentence
                        # Look for sentence-ending punctuation followed by space or end of buffer
                        if (re.search(r'[.!?](\s|$)', sentence_buffer)):
                            # We have a complete sentence
                            complete_sentences.append(sentence_buffer.strip())
                            generated_text += sentence_buffer
                            sentence_buffer = ""
                            sentence_count += 1
                            
                            # Check if we've reached the maximum number of sentences
                            if sentence_count >= max_sentences:
                                stream_ended = True
                                break
                
                    # Also check for repetition
                    if len(generated_text) > 100 and sentence_count >= 2:
                        # Simple check for repetition - look for the same sentence appearing twice
                        if len(complete_sentences) >= 2:
                            # Check if the last two sentences are very similar
                            if complete_sentences[-1] and complete_sentences[-2]:
                                similarity = _compute_similarity(complete_sentences[-1], complete_sentences[-2])
                                if similarity > 0.7:  # High similarity threshold
                                    stream_ended = True
                                    break
                
        except Exception as e:
            print(f"\nError in streaming: {e}")
        
        # Add any remaining sentence buffer to the generated text
        if sentence_buffer:
            generated_text += sentence_buffer
        
        # Final cleanup of any tags
        clean_response = re.sub(r'</?[a-z_]+>', '', generated_text).strip()
        
        # Ensure complete sentences - check if we end with punctuation
        if not re.search(r'[.!?]$', clean_response) and sentence_buffer:
            # Add a period to the end if needed
            clean_response += "."
        
        # Update conversation history
        question_tag = f"<question>{prompt}</question>"
        answer_tag = f"<answer>{clean_response}</answer>"
        self.conversation_history.append(question_tag)
        self.conversation_history.append(answer_tag)
        
        return combined_input + clean_response





    def update_conversation_history(self, user_input, model_response):
        """Add a conversation exchange to the history with XML tags"""
        question_tag = f"<question>{user_input}</question>"
        answer_tag = f"<answer>{model_response}</answer>"
        self.conversation_history.append(question_tag)
        self.conversation_history.append(answer_tag)
        
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

    # 1. Fix the tokenizer padding issue in __init__
    def _fix_init(self):
        # Make sure pad_token is different from eos_token
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.pad_token = '[PAD]'
            # Make sure the model knows about the pad token too
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"Set pad_token to '{self.tokenizer.pad_token}' (id: {self.tokenizer.pad_token_id})")



# Helper function to compute sentence similarity
def _compute_similarity(sentence1, sentence2):
    """
    Compute a simple similarity score between two sentences.
    Returns a value between 0 (completely different) and 1 (identical).
    """
    # Convert to lowercase and remove punctuation
    s1 = re.sub(r'[^\w\s]', '', sentence1.lower())
    s2 = re.sub(r'[^\w\s]', '', sentence2.lower())
    
    # Get word sets
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

# Modify the interactive_with_options function to add a 'reset' command
def interactive_with_options(generator, args):
    """Interactive mode with multiple response options and improved context handling"""
    print(f"Interactive Mode - Generating {args.num_responses} response options")
    print("Type 'exit' to quit, 'clear' to start a new conversation, 'reset' to clear context but keep history, 'debug' to toggle debug mode, 'verbose' to toggle verbose mode")
    
    context = args.context
    system_prompt = args.system_prompt
    quiet_mode = args.quiet
    
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
        
        if user_input.lower() == "reset":
            # Clear just the context but keep the conversation history for reference
            context = None
            print(generator.clear_context())
            continue
        
        if user_input.lower() == "debug":
            generator.debug = not generator.debug
            print(f"Debug mode {'enabled' if generator.debug else 'disabled'}")
            continue
            
        if user_input.lower() == "verbose":
            quiet_mode = not quiet_mode
            print(f"Verbose mode {'disabled' if quiet_mode else 'enabled'}")
            continue
        
        # Generate multiple responses
        if not quiet_mode:
            print("\nGenerating options...\n")
        else:
            print("\nGenerating...", end="", flush=True)
            
        responses = generator.generate_multiple_responses(
            prompt=user_input,
            system_prompt=system_prompt,
            context=context,
            num_responses=args.num_responses,
            context_length=args.context_length,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            quiet=quiet_mode
        )
        
        if quiet_mode:
            print("\r" + " " * 12 + "\r", end="")  # Clear "Generating..." text
        
        # Display options to user
        for i, resp in enumerate(responses):
            print(f"\n--- Option {i+1} ---\n")
            print(resp)
            print("\n" + "-" * 30)
        
        # Let user choose
        while True:
            choice = input(f"\nSelect your preferred option (1-{args.num_responses}), 'n' for new options, 's' to skip, or 'r' to reset context: ")
            if choice.lower() == 'n':
                if not quiet_mode:
                    print("\nGenerating new options...\n")
                else:
                    print("\nGenerating...", end="", flush=True)
                    
                responses = generator.generate_multiple_responses(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    context=context,
                    num_responses=args.num_responses,
                    context_length=args.context_length,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    quiet=quiet_mode
                )
                
                if quiet_mode:
                    print("\r" + " " * 12 + "\r", end="")  # Clear "Generating..." text
                    
                for i, resp in enumerate(responses):
                    print(f"\n--- Option {i+1} ---\n")
                    print(resp)
                    print("\n" + "-" * 30)
            elif choice.lower() == 's':
                print("Skipping this exchange - not added to conversation history.")
                break
            elif choice.lower() == 'r':
                # Reset context mid-exchange
                context = None
                print(generator.clear_context())
                print("Regenerating response with cleared context...")
                
                responses = generator.generate_multiple_responses(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    context=None,  # Explicitly pass None to clear context
                    num_responses=args.num_responses,
                    context_length=args.context_length,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    quiet=quiet_mode
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
                
                # Update context from conversation history - but only use the most recent exchange
                # to prevent context overload and topic drift
                latest_exchange = ""
                if len(generator.conversation_history) >= 2:
                    latest_exchange = "\n".join(generator.conversation_history[-2:])
                context = latest_exchange
                
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
                print(f"Please enter a number between 1 and {args.num_responses}, 'n', 's', or 'r'.")


# Also update the streaming interactive mode
def interactive_streaming(generator, args):
    """Interactive streaming mode with improved context handling"""
    print("Interactive Streaming Mode - Direct Response")
    print("Type 'exit' to quit, 'clear' to start a new conversation, 'reset' to clear context but keep history, 'debug' to toggle debug mode")
    
    context = args.context
    system_prompt = args.system_prompt
    quiet_mode = args.quiet
    
    # Force debug off if quiet mode is on
    if quiet_mode:
        generator.debug = False
    
    while True:
        user_input = input("\nEnter a prompt (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "clear":
            generator.clear_conversation_history()
            context = args.context
            print("Conversation history cleared.")
            continue
        
        if user_input.lower() == "reset":
            # Clear just the context but keep the conversation history for reference
            context = None
            print(generator.clear_context())
            continue
        
        if user_input.lower() == "debug":
            generator.debug = not generator.debug
            print(f"Debug mode {'enabled' if generator.debug else 'disabled'}")
            continue

        if user_input.lower() == "verbose":
            quiet_mode = not quiet_mode
            generator.debug = not quiet_mode  # Link debug and quiet modes
            print(f"Verbose mode {'disabled' if quiet_mode else 'enabled'}")
            continue
        
        # Generate streaming response - pass through the user's max_length parameter
        if not quiet_mode:
            print("\nGenerating streaming response...")
        
        response = generator.generate_streaming_text(
            prompt=user_input,
            system_prompt=system_prompt,
            context=context,
            context_length=args.context_length,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Update context but only use the most recent exchange to prevent context drift
        latest_exchange = ""
        if len(generator.conversation_history) >= 2:
            latest_exchange = "\n".join(generator.conversation_history[-2:])
        context = latest_exchange
        
        print("\n" + "=" * 50 + "\n")
        


def main():
    parser = argparse.ArgumentParser(description="Enhanced text generator using a fine-tuned GPT-2 model")
    parser.add_argument("--model_path", default="./gpt2-finetuned", help="Path to the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling (lower = more focused)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--stream", action="store_true", help="Stream the output token by token")
    parser.add_argument("--system_prompt", type=str, 
                    default="You are a factual assistant who provides accurate information. Stick to verified facts and clearly separate facts from opinions or speculation and without repetition.", help="System instructions for the model")
    parser.add_argument("--context_length", type=int, default=512, help="Length of context to consider")
    parser.add_argument("--context", type=str, default=None, help="Additional context to provide to the model")
    parser.add_argument("--num_responses", type=int, default=3, help="Number of response options to generate (ignored in streaming mode)")
    parser.add_argument("--save_conversations", action="store_true", help="Save conversations to JSON files")
    parser.add_argument("--options", action="store_true", help="Use multiple options mode even with streaming")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to show input previews")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output mode")
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    # Initialize generator
    generator = GPT2Generator(args.model_path)
    if args.debug and not args.quiet:  # Don't enable debug if quiet mode
        generator.enable_debug(True)
    
    # Choose interaction mode - always use direct streaming if --stream is specified
    # unless --options is explicitly set
    if args.stream and not args.options:
        interactive_streaming(generator, args)
    else:
        interactive_with_options(generator, args)        

if __name__ == "__main__":
    main()