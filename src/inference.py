"""
Inference script for SQL LoRA fine-tuned Llama-2-7B model
Provides interactive interface and batch inference capabilities
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json
from typing import Optional, List, Dict
import sys


class SQLGenerator:
    def __init__(
        self,
        base_model_name: str,
        lora_checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_local_model: bool = False
    ):
        """
        Initialize SQL Generator
        
        Args:
            base_model_name: HuggingFace model name or local path
            lora_checkpoint_path: Path to LoRA checkpoint (optional)
            device: Device to run inference on
            use_local_model: If True, load from local cache only
        """
        self.device = device
        self.use_local_model = use_local_model
        print(f"Loading model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=use_local_model
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=use_local_model
        )
        
        # Load LoRA weights if provided
        if lora_checkpoint_path:
            print(f"Loading LoRA weights from {lora_checkpoint_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint_path)
        
        self.model.eval()
        print("Model loaded successfully!\n")
    
    def format_prompt(self, question: str, context: str) -> str:
        """Format the input prompt for CodeLlama Instruct model"""
        prompt = f"""[INST] You are a helpful SQL expert. Generate a SQL query to answer the question based on the provided database schema.

Question: {question}

Database Schema:
{context}

Generate only the SQL query without explanation. [/INST]"""
        return prompt
    
    def generate(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate SQL query for given question and database context
        
        Args:
            question: Natural language question
            context: Database schema/context
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated SQL queries
        """
        prompt = self.format_prompt(question, context)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode generated sequences
        generated_texts = []
        for output in outputs:
            # Decode only the generated part (exclude prompt)
            generated_text = self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    def interactive_mode(self):
        """Run interactive inference mode"""
        print("="*70)
        print("SQL QUERY GENERATOR - Interactive Mode")
        print("="*70)
        print("Enter your question and database schema to generate SQL queries.")
        print("Type 'quit' or 'exit' to exit.\n")
        
        while True:
            try:
                # Get question
                print("\n" + "-"*70)
                question = input("Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive mode...")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                # Get context
                print("\nDatabase Schema (press Enter twice when done):")
                context_lines = []
                while True:
                    line = input()
                    if line == "":
                        if context_lines and context_lines[-1] == "":
                            context_lines.pop()
                            break
                        context_lines.append(line)
                    else:
                        context_lines.append(line)
                
                context = "\n".join(context_lines).strip()
                
                if not context:
                    print("Please enter a database schema.")
                    continue
                
                # Generate SQL
                print("\nGenerating SQL query...")
                generated_sqls = self.generate(
                    question=question,
                    context=context,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1
                )
                
                # Display result
                print("\n" + "="*70)
                print("GENERATED SQL QUERY:")
                print("="*70)
                print(generated_sqls[0])
                print("="*70)
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
    
    def batch_inference(
        self,
        input_file: str,
        output_file: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ):
        """
        Run batch inference on a JSON file
        
        Args:
            input_file: Path to input JSON file with format:
                        [{"question": "...", "context": "..."}, ...]
            output_file: Path to output JSON file
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        print(f"Loading input from {input_file}...")
        with open(input_file, 'r') as f:
            inputs = json.load(f)
        
        results = []
        print(f"Processing {len(inputs)} queries...")
        
        for i, item in enumerate(inputs):
            print(f"Processing query {i+1}/{len(inputs)}...")
            
            question = item.get('question', '')
            context = item.get('context', '')
            
            if not question or not context:
                print(f"Skipping item {i+1}: missing question or context")
                continue
            
            try:
                generated_sqls = self.generate(
                    question=question,
                    context=context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1
                )
                
                results.append({
                    'question': question,
                    'context': context,
                    'generated_sql': generated_sqls[0],
                    'reference_sql': item.get('answer', None)
                })
                
            except Exception as e:
                print(f"Error processing item {i+1}: {str(e)}")
                results.append({
                    'question': question,
                    'context': context,
                    'generated_sql': None,
                    'error': str(e)
                })
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch inference complete! Processed {len(results)} queries.")


def main():
    parser = argparse.ArgumentParser(
        description="SQL Query Generator - Inference Script"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to LoRA checkpoint (optional)"
    )
    parser.add_argument(
        "--use_local",
        action="store_true",
        help="Use local cached model only (no download)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['interactive', 'batch'],
        default='interactive',
        help="Inference mode: interactive or batch"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input JSON file for batch inference"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.json",
        help="Output JSON file for batch inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question for quick inference"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Database context for quick inference"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SQLGenerator(
        base_model_name=args.base_model,
        lora_checkpoint_path=args.lora_checkpoint,
        use_local_model=args.use_local
    )
    
    # Run inference based on mode
    if args.question and args.context:
        # Quick single inference
        print("Question:", args.question)
        print("Context:", args.context)
        print("\nGenerating SQL query...\n")
        
        results = generator.generate(
            question=args.question,
            context=args.context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        print("="*70)
        print("GENERATED SQL QUERY:")
        print("="*70)
        print(results[0])
        print("="*70)
        
    elif args.mode == 'interactive':
        generator.interactive_mode()
        
    elif args.mode == 'batch':
        if not args.input_file:
            print("Error: --input_file is required for batch mode")
            sys.exit(1)
        
        generator.batch_inference(
            input_file=args.input_file,
            output_file=args.output_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )


if __name__ == "__main__":
    main()