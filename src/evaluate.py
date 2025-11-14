"""
Evaluation script for SQL LoRA fine-tuned Llama-2-7B model
Evaluates the model on validation/test dataset with metrics
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
import argparse
from typing import Dict, List
import re


def calculate_exact_match(predicted: str, reference: str) -> bool:
    """Calculate exact match between predicted and reference SQL"""
    # Normalize whitespace and case
    pred_normalized = ' '.join(predicted.lower().split())
    ref_normalized = ' '.join(reference.lower().split())
    return pred_normalized == ref_normalized


def calculate_sql_accuracy(predicted: str, reference: str) -> float:
    """Calculate SQL token accuracy"""
    pred_tokens = predicted.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(ref_tokens) == 0:
        return 0.0
    
    # Calculate token overlap
    matches = sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
    return matches / len(ref_tokens)


def extract_sql_from_response(response: str) -> str:
    """Extract SQL query from model response"""
    # Try to find SQL between common delimiters
    sql_patterns = [
        r"```sql\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"SELECT.*?;",
        r"INSERT.*?;",
        r"UPDATE.*?;",
        r"DELETE.*?;",
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return response.strip()


def format_prompt(question: str, context: str) -> str:
    """Format input prompt for CodeLlama Instruct model"""
    prompt = f"""[INST] You are a helpful SQL expert. Generate a SQL query to answer the question based on the provided database schema.

Question: {question}

Database Schema:
{context}

Generate only the SQL query without explanation. [/INST]"""
    return prompt


class SQLEvaluator:
    def __init__(
        self,
        base_model_name: str,
        lora_checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        print(f"Loading model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load LoRA weights
        print(f"Loading LoRA weights from {lora_checkpoint_path}...")
        self.model = PeftModel.from_pretrained(self.model, lora_checkpoint_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_sql(self, question: str, context: str, max_new_tokens: int = 256) -> str:
        """Generate SQL query for given question and context"""
        prompt = format_prompt(question, context)
        
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
                do_sample=False,  # Greedy decoding for evaluation
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (exclude prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return extract_sql_from_response(generated_text)
    
    def evaluate_dataset(
        self,
        dataset_name: str = "b-mc2/sql-create-context",
        split: str = "train",
        num_samples: int = None,
        batch_size: int = 1
    ) -> Dict:
        """Evaluate model on dataset"""
        print(f"Loading dataset: {dataset_name}, split: {split}")
        dataset = load_dataset(dataset_name, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        metrics = defaultdict(list)
        predictions = []
        
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            question = example['question']
            context = example['context']
            reference_sql = example['answer']
            
            try:
                # Generate prediction
                predicted_sql = self.generate_sql(question, context)
                
                # Calculate metrics
                exact_match = calculate_exact_match(predicted_sql, reference_sql)
                sql_accuracy = calculate_sql_accuracy(predicted_sql, reference_sql)
                
                metrics['exact_match'].append(exact_match)
                metrics['sql_accuracy'].append(sql_accuracy)
                
                predictions.append({
                    'index': i,
                    'question': question,
                    'context': context,
                    'reference': reference_sql,
                    'prediction': predicted_sql,
                    'exact_match': exact_match,
                    'sql_accuracy': sql_accuracy
                })
                
            except Exception as e:
                print(f"\nError processing example {i}: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        results = {
            'num_samples': len(predictions),
            'exact_match_rate': np.mean(metrics['exact_match']) * 100,
            'average_sql_accuracy': np.mean(metrics['sql_accuracy']) * 100,
            'predictions': predictions
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL LoRA model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="b-mc2/sql-create-context",
        help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SQLEvaluator(
        base_model_name=args.base_model,
        lora_checkpoint_path=args.lora_checkpoint
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Exact Match Rate: {results['exact_match_rate']:.2f}%")
    print(f"Average SQL Accuracy: {results['average_sql_accuracy']:.2f}%")
    print("="*50)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {args.output_file}")
    
    # Show some examples
    print("\nSample Predictions:")
    for i, pred in enumerate(results['predictions'][:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {pred['question']}")
        print(f"Reference: {pred['reference']}")
        print(f"Prediction: {pred['prediction']}")
        print(f"Exact Match: {pred['exact_match']}")
        print(f"SQL Accuracy: {pred['sql_accuracy']:.2f}")


if __name__ == "__main__":
    main()