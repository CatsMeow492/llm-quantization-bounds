#!/usr/bin/env python3
"""
Setup script for LLM Quantization Bounds Research
Creates reproducible environment for DialoGPT fine-tuning experiments
"""

from setuptools import setup, find_packages

# Core dependencies with current versions for compatibility
REQUIREMENTS = [
    # Deep Learning Framework
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "torchaudio>=2.0.0",
    
    # Transformers and Model Hub
    "transformers>=4.30.0",
    "tokenizers>=0.13.0",
    "datasets>=2.10.0",
    "huggingface-hub>=0.15.0",
    
    # Quantization Libraries
    "bitsandbytes>=0.40.0",
    "accelerate>=0.20.0",
    "peft>=0.5.0",  # For LoRA implementation
    
    # Scientific Computing
    "numpy>=1.20.0",
    "scipy>=1.8.0",
    "pandas>=1.5.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    
    # Jupyter and Analysis
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
    "sympy>=1.10.0",
    
    # Utilities
    "tqdm>=4.60.0",
    "wandb>=0.13.0",  # For experiment tracking
    "psutil>=5.8.0",   # For system monitoring
    "GPUtil>=1.4.0",   # For GPU monitoring
    
    # Development
    "pytest>=7.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
]

# Optional dependencies for extended functionality
EXTRAS = {
    "dev": [
        "pre-commit==3.5.0",
        "flake8==6.1.0",
        "mypy==1.7.0",
    ],
    "paper": [
        "latexcodec==2.0.1",
        "bibtexparser==1.4.0",
    ]
}

setup(
    name="llm-quantization-bounds",
    version="1.0.0",
    description="Theoretical bounds for quantization noise in LoRA fine-tuning",
    author="Research Team",
    author_email="research@example.com",
    url="https://github.com/CatsMeow492/llm-quantization-bounds",
    
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS,
    
    # Entry points for experiment scripts
    entry_points={
        "console_scripts": [
            "run-experiment=experiments.run_experiment:main",
            "analyze-results=experiments.analyze_results:main",
        ],
    },
    
    # Package data
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    
    # Long description
    long_description="""
    This package implements theoretical bounds for quantization noise in LoRA fine-tuning
    of large language models. It provides:
    
    - Theoretical analysis of quantization-LoRA interaction
    - Experimental validation on DialoGPT fine-tuning
    - Optimal bit-width selection algorithms
    - Comprehensive benchmarking tools
    
    The research establishes fundamental limits linking quantization precision to 
    fine-tuning performance, enabling principled precision-performance trade-offs.
    """,
    long_description_content_type="text/plain",
) 