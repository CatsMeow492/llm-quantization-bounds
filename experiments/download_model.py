#!/usr/bin/env python3
"""
Download DialoGPT-medium model for quantization experiments
Ensures reproducible model access across experimental runs
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from huggingface_hub import hf_hub_download
import psutil


def check_system_requirements():
    """Check if system meets minimum requirements for experiments"""
    print("=== System Requirements Check ===")
    
    # Check available RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Available RAM: {ram_gb:.1f} GB")
    if ram_gb < 16:
        print("⚠️  Warning: Less than 16GB RAM available. Model loading may be slow.")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs available: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  No GPU available. Experiments will run on CPU (very slow).")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    print(f"Available disk space: {free_gb:.1f} GB")
    if free_gb < 10:
        print("⚠️  Warning: Less than 10GB disk space. Model download may fail.")
    
    print()


def download_dialogpt_model(model_name: str = "microsoft/DialoGPT-medium", 
                           cache_dir: Optional[str] = None) -> tuple:
    """
    Download DialoGPT model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Directory to cache model files
        
    Returns:
        tuple: (model, tokenizer) objects
    """
    print(f"=== Downloading {model_name} ===")
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "data", "models")
    
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side='left'
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
        
        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Print model info
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model loaded:")
        print(f"  Total parameters: {param_count:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{param_count * 2 / 1e9:.1f} GB (fp16)")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)


def test_model_generation(model, tokenizer, test_prompt: str = "Hello, how are you?"):
    """Test that model can generate text properly"""
    print(f"=== Testing Model Generation ===")
    print(f"Test prompt: '{test_prompt}'")
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(test_prompt, return_tensors='pt')
        
        # Move to same device as model
        if hasattr(model, 'device'):
            inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 20,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated response: '{response}'")
        print("✓ Model generation test passed")
        
    except Exception as e:
        print(f"❌ Model generation test failed: {e}")
        return False
    
    return True


def save_model_info(model, tokenizer, output_path: str):
    """Save model configuration and metadata"""
    print(f"=== Saving Model Info ===")
    
    info = {
        "model_name": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "microsoft/DialoGPT-medium",
        "model_type": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "tokenizer_vocab_size": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Model info saved to {output_path}")


def main():
    """Main function to download and verify DialoGPT model"""
    parser = argparse.ArgumentParser(description="Download DialoGPT model for experiments")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", 
                       help="HuggingFace model identifier")
    parser.add_argument("--cache-dir", default=None,
                       help="Directory to cache model files")
    parser.add_argument("--test-generation", action="store_true",
                       help="Test model generation after download")
    parser.add_argument("--save-info", default="data/model_info.json",
                       help="Path to save model information")
    
    args = parser.parse_args()
    
    # Check system requirements
    check_system_requirements()
    
    # Download model
    model, tokenizer = download_dialogpt_model(args.model, args.cache_dir)
    
    # Test generation if requested
    if args.test_generation:
        test_model_generation(model, tokenizer)
    
    # Save model info
    if args.save_info:
        save_model_info(model, tokenizer, args.save_info)
    
    print("\n=== Download Complete ===")
    print("✓ DialoGPT model ready for experiments")
    print("✓ Run experiments with: python experiments/run_experiment.py")


if __name__ == "__main__":
    main() 