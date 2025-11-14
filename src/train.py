"""
Training module for SQL generation with LoRA fine-tuning.
Implements QLoRA training with configurable hyperparameters.
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk, DatasetDict
import json
from typing import Optional, Dict
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

assert torch.cuda.is_available(), "CUDA not available! Install PyTorch with CUDA support"
print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

class SQLLoRATrainer:
    """Handles LoRA fine-tuning for SQL generation."""
    
    def __init__(self,
                 model_name: str = "codellama/CodeLlama-7b-hf",
                 use_4bit: bool = True,
                 use_8bit: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model_name: Base model from HuggingFace or local path
            use_4bit: Use 4-bit quantization (QLoRA)
            use_8bit: Use 8-bit quantization
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load model with quantization and tokenizer."""
        print(f"\n{'='*60}")
        print(f"Loading model: {self.model_name}")
        print(f"{'='*60}\n")
        
        # DIAGNOSTIC: Check CUDA availability
        print("CUDA Diagnostics:")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"  - GPU name: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  ⚠️ WARNING: CUDA not available! Training will be VERY slow on CPU!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        print()
        
        try:
            # Check if local path exists
            local_model = os.path.exists(self.model_name)
            if local_model:
                print(f"✓ Found local model at: {self.model_name}")
            else:
                print(f"⚠ Loading from HuggingFace Hub: {self.model_name}")
            
            # Configure quantization
            bnb_config = None
            if self.use_4bit:
                print("Configuring 4-bit quantization (QLoRA)...")
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not configure 4-bit quantization: {e}")
                    print("Continuing without quantization...")
                    self.use_4bit = False
                    bnb_config = None
            elif self.use_8bit:
                print("Configuring 8-bit quantization...")
                try:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                except Exception as e:
                    print(f"⚠ Warning: Could not configure 8-bit quantization: {e}")
                    print("Continuing without quantization...")
                    self.use_8bit = False
                    bnb_config = None
            else:
                print("Loading model without quantization (full precision with LoRA)...")
            
            # Load model
            print("Loading model weights (this may take a few minutes)...")
            
            if bnb_config is not None:
                # Quantized model loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                )            
            else:
                # Non-quantized loading
                if local_model:
                    # Local folder → use accelerate-safe load
                    print("✓ Loading local model with accelerate...")
                    try:
                        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
                        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                        with init_empty_weights():
                            empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                        self.model = load_checkpoint_and_dispatch(
                            empty_model,
                            checkpoint=self.model_name,
                            device_map="auto",
                            dtype=torch.bfloat16
                        )
                    except ImportError:
                        print("⚠ accelerate not available, using standard loading...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            device_map="auto",
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True
                        )
                else:
                    # Remote model → normal transformers load
                    print("⚠ Loading model from Hugging Face Hub...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True
                    )

            print("✓ Model loaded successfully!")
            
            # DIAGNOSTIC: Check where model is loaded
            print(f"\nModel device placement:")
            if hasattr(self.model, 'hf_device_map'):
                print(f"  - Device map: {self.model.hf_device_map}")
            for name, param in list(self.model.named_parameters())[:3]:
                print(f"  - {name}: {param.device}")
            print()

            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("✓ Tokenizer loaded successfully!")

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                print("Setting pad token to eos token...")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Prepare model for k-bit training only if using quantization
            if self.use_4bit or self.use_8bit:
                print("Preparing model for k-bit training...")
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=True
                )
            else:
                # For non-quantized training, enable gradient checkpointing manually
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    print("Enabling gradient checkpointing...")
                    self.model.gradient_checkpointing_enable()
            
            # Ensure model is in training mode
            self.model.train()
            
            # Disable cache for training (silences warning)
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False

            print(f"\n{'='*60}")
            print("✓ Model and tokenizer loaded successfully!")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ ERROR loading model: {e}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    def configure_lora(self,
                      r: int = 16,
                      lora_alpha: int = 32,
                      lora_dropout: float = 0.05,
                      target_modules: Optional[list] = None) -> LoraConfig:
        """
        Configure LoRA parameters.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout probability
            target_modules: Which modules to apply LoRA to
            
        Returns:
            LoraConfig object
        """
        if target_modules is None:
            # Target modules for LLaMA/CodeLlama models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        print(f"\nConfiguring LoRA:")
        print(f"  - Rank (r): {r}")
        print(f"  - Alpha: {lora_alpha}")
        print(f"  - Dropout: {lora_dropout}")
        print(f"  - Target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        return lora_config
    
    def apply_lora(self, lora_config: LoraConfig):
        """Apply LoRA to the model."""
        print("\n" + "="*60)
        print("Applying LoRA configuration...")
        print("="*60)
        
        try:
            self.model = get_peft_model(self.model, lora_config)
            
            # Ensure model is in training mode after PEFT
            self.model.train()
            
            print("\n✓ LoRA applied successfully!")
            print("\nTrainable parameters:")
            self.model.print_trainable_parameters()
            print()
        except Exception as e:
            print(f"\n❌ ERROR applying LoRA: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def tokenize_dataset(self, dataset: DatasetDict, max_length: int = 512) -> DatasetDict:
        """
        Tokenize the dataset.
        
        Args:
            dataset: Dataset to tokenize
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        print(f"\n{'='*60}")
        print("Tokenizing dataset...")
        print(f"Max length: {max_length}")
        print(f"{'='*60}\n")
        
        def tokenize_function(examples):
            # Tokenize - let DataCollator handle labels
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
        
        try:
            tokenized = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset['train'].column_names,
                desc="Tokenizing"
            )
            
            print(f"\n✓ Tokenization complete!")
            print(f"  - Train samples: {len(tokenized['train'])}")
            print(f"  - Validation samples: {len(tokenized['validation'])}")
            print()
            
            return tokenized
            
        except Exception as e:
            print(f"\n❌ ERROR during tokenization: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def train(self,
              train_dataset,
              eval_dataset,
              output_dir: str = "./models/sql-lora",
              num_epochs: int = 3,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100,
              logging_steps: int = 10,
              save_steps: int = 500,
              eval_steps: int = 500,
              resume_from_checkpoint: Optional[str] = None):
        """
        Train the model with LoRA.
        """
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        print(f"\nTraining configuration:")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine compute dtype based on available hardware
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16
        
        print(f"Compute settings:")
        print(f"  - BF16: {use_bf16}")
        print(f"  - FP16: {use_fp16}")
        print(f"  - Using quantization: {self.use_4bit or self.use_8bit}")
        print()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",  # Changed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,
            bf16=use_bf16,
            optim="paged_adamw_8bit" if (self.use_4bit or self.use_8bit) else "adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",
            save_total_limit=3,
            remove_unused_columns=False,
            logging_first_step=True,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False if torch.cuda.device_count() <= 1 else None
        )
        
        # Data collator with dynamic padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Helps with performance on GPUs
        )
        
        try:
            # Initialize trainer
            print("Initializing Trainer...")
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator
            )
            print("✓ Trainer initialized!\n")
            
            # Train
            print("Starting training loop...\n")
            
            # Check for resume
            if resume_from_checkpoint:
                if resume_from_checkpoint == "True" or resume_from_checkpoint == "true":
                    # Auto-detect latest checkpoint
                    print("Looking for latest checkpoint to resume from...")
                    resume_from_checkpoint = True
                else:
                    print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save final model
            print(f"\nSaving final model to {output_dir}...")
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"\n{'='*60}")
            print(f"✓ Training completed! Model saved to {output_dir}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ ERROR during training: {e}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def save_training_config(self, output_dir: str, config: Dict):
        """Save training configuration to JSON."""
        try:
            config_path = os.path.join(output_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Training config saved to {config_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not save config: {e}")


def main():
    """Main training script."""
    print("\n" + "="*60)
    print("SQL LoRA Training Script")
    print("="*60 + "\n")
    
    parser = argparse.ArgumentParser(description="Train SQL generation model with LoRA")
    parser.add_argument("--model_name", type=str, 
                       default="meta-llama/Llama-2-7b-hf",
                       help="Base model name or local path")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Directory with preprocessed data")
    parser.add_argument("--output_dir", type=str, default="./models/sql-lora",
                       help="Output directory for model")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--use_4bit", action="store_true", default=False,
                       help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true", default=False,
                       help="Use 8-bit quantization")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from (or 'True' for latest)")
    
    args = parser.parse_args()
    
    # Convert to absolute path if local path
    if not args.model_name.startswith("http") and "/" not in args.model_name:
        abs_model_path = os.path.abspath(args.model_name)
        if os.path.exists(abs_model_path):
            args.model_name = abs_model_path
            print(f"✓ Using local model at: {abs_model_path}")
        else:
            print(f"⚠ WARNING: Local path not found: {abs_model_path}")
            print(f"Will try to download from HuggingFace: {args.model_name}")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    
    print("\nConfiguration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ ERROR: Data directory not found: {args.data_dir}")
        print("Please run data preprocessing first!")
        sys.exit(1)
    
    try:
        # Load preprocessed data
        print(f"Loading data from {args.data_dir}...")
        dataset = load_from_disk(args.data_dir)
        print(f"✓ Dataset loaded!")
        print(f"  - Train samples: {len(dataset['train'])}")
        print(f"  - Validation samples: {len(dataset['validation'])}")
        print()
        
        # Check dataset structure
        print("Dataset columns:", dataset['train'].column_names)
        if 'text' not in dataset['train'].column_names:
            print("❌ ERROR: Dataset must have a 'text' column!")
            print("Available columns:", dataset['train'].column_names)
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize trainer
    trainer = SQLLoRATrainer(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Configure and apply LoRA
    lora_config = trainer.configure_lora(
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    trainer.apply_lora(lora_config)
    
    # Tokenize dataset
    tokenized_dataset = trainer.tokenize_dataset(dataset, max_length=args.max_length)
    
    # Train
    trainer.train(
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Save config
    config = vars(args)
    trainer.save_training_config(args.output_dir, config)
    
    print("\n" + "="*60)
    print("✓ All done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)