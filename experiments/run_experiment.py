#!/usr/bin/env python3
"""
Main experiment runner for quantized LoRA fine-tuning
Tests theoretical predictions across different bit-widths and ranks
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import load_dataset
import bitsandbytes as bnb
from accelerate import Accelerator

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ExperimentConfig:
    """Configuration for quantization experiments"""
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    
    # Quantization settings
    bits: int = 16
    use_quantization: bool = True
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data settings
    dataset_name: str = "daily_dialog"
    max_train_samples: int = 1000
    max_eval_samples: int = 200
    
    # Experiment settings
    seed: int = 42
    output_dir: str = "results"
    save_model: bool = False
    track_gradients: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj", "c_fc"]


class DialogDataset(Dataset):
    """Custom dataset for dialog fine-tuning"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


def setup_quantization(model, bits: int = 8):
    """Setup quantization configuration"""
    if bits == 16:
        return model  # No quantization
    
    print(f"Setting up {bits}-bit quantization...")
    
    if bits == 8:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
        )
    elif bits == 4:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise ValueError(f"Unsupported quantization: {bits} bits")
    
    # Reload model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model.config.name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model


def setup_lora(model, config: ExperimentConfig):
    """Setup LoRA configuration"""
    print(f"Setting up LoRA with rank {config.lora_rank}...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model


def load_dataset_for_dialog(config: ExperimentConfig) -> Tuple[List[str], List[str]]:
    """Load and preprocess dialog dataset"""
    print(f"Loading {config.dataset_name} dataset...")
    
    # Load DailyDialog dataset
    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    
    # Process dialogs into conversation format
    train_texts = []
    eval_texts = []
    
    for split, texts in [("train", train_texts), ("validation", eval_texts)]:
        data = dataset[split]
        max_samples = config.max_train_samples if split == "train" else config.max_eval_samples
        
        for i, dialog in enumerate(data["dialog"][:max_samples]):
            # Convert dialog to single conversation string
            conversation = ""
            for turn in dialog:
                if turn.strip():  # Skip empty turns
                    conversation += turn.strip() + " "
            
            if conversation.strip():
                texts.append(conversation.strip())
    
    print(f"Loaded {len(train_texts)} training and {len(eval_texts)} evaluation samples")
    return train_texts, eval_texts


class GradientTracker:
    """Track gradient statistics during training"""
    
    def __init__(self):
        self.gradient_norms = []
        self.gradient_variances = []
        self.step_count = 0
    
    def track_gradients(self, model):
        """Track gradients for current step"""
        total_norm = 0.0
        gradient_values = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                gradient_values.extend(param.grad.data.cpu().numpy().flatten())
        
        total_norm = total_norm ** 0.5
        gradient_variance = np.var(gradient_values) if gradient_values else 0.0
        
        self.gradient_norms.append(total_norm)
        self.gradient_variances.append(gradient_variance)
        self.step_count += 1
    
    def get_statistics(self) -> Dict:
        """Get gradient statistics"""
        if not self.gradient_norms:
            return {}
        
        return {
            "avg_gradient_norm": np.mean(self.gradient_norms),
            "std_gradient_norm": np.std(self.gradient_norms),
            "avg_gradient_variance": np.mean(self.gradient_variances),
            "std_gradient_variance": np.std(self.gradient_variances),
            "total_steps": self.step_count
        }


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run single quantization experiment"""
    print(f"\n=== Running Experiment: {config.bits}-bit, rank {config.lora_rank}, seed {config.seed} ===")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize tracking
    gradient_tracker = GradientTracker()
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU/MPS compatibility
        device_map="auto"
    )
    
    # Setup quantization
    if config.use_quantization and config.bits < 16:
        model = setup_quantization(model, config.bits)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Load dataset
    train_texts, eval_texts = load_dataset_for_dialog(config)
    
    # Create datasets
    train_dataset = DialogDataset(train_texts, tokenizer, config.max_length)
    eval_dataset = DialogDataset(eval_texts, tokenizer, config.max_length)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/temp_{config.bits}bit_r{config.lora_rank}_s{config.seed}",
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        fp16=False,  # Disable fp16 for CPU/MPS compatibility
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=[],  # Disable wandb completely
    )
    
    # Custom trainer with gradient tracking
    class CustomTrainer(Trainer):
        def __init__(self, *args, gradient_tracker=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.gradient_tracker = gradient_tracker
        
        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Track gradients if enabled
            if self.gradient_tracker and config.track_gradients:
                self.gradient_tracker.track_gradients(model)
            
            return loss
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        gradient_tracker=gradient_tracker,
    )
    
    # Training
    start_time = time.time()
    print("Starting training...")
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("Running final evaluation...")
    eval_result = trainer.evaluate()
    
    # Collect results (convert numpy types to native Python types for JSON serialization)
    results = {
        "config": asdict(config),
        "training_time": float(training_time),
        "train_loss": float(train_result.training_loss),
        "eval_loss": float(eval_result["eval_loss"]),
        "eval_perplexity": float(np.exp(eval_result["eval_loss"])),
        "gradient_stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in gradient_tracker.get_statistics().items()},
        "model_info": {
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        }
    }
    
    # Save results
    output_file = f"{config.output_dir}/results_{config.bits}bit_r{config.lora_rank}_s{config.seed}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    print(f"✓ Training completed in {training_time:.1f}s")
    print(f"✓ Final eval loss: {eval_result['eval_loss']:.4f}")
    print(f"✓ Final perplexity: {np.exp(eval_result['eval_loss']):.2f}")
    
    return results


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="Run quantized LoRA fine-tuning experiment")
    parser.add_argument("--bits", type=int, default=16, choices=[4, 8, 16],
                       help="Quantization bit-width")
    parser.add_argument("--rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum training samples")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory")
    parser.add_argument("--no-quantization", action="store_true",
                       help="Disable quantization")
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        bits=args.bits,
        lora_rank=args.rank,
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_samples,
        output_dir=args.output_dir,
        use_quantization=not args.no_quantization
    )
    
    # Run experiment
    results = run_experiment(config)
    
    print("\n=== Experiment Complete ===")
    print(f"Results saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 