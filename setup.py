#!/usr/bin/env python3
"""
Setup script for LLM Quantization Bounds Research
Creates reproducible environment for DialoGPT fine-tuning experiments
"""

from setuptools import setup, find_packages

# Core dependencies with pinned versions for reproducibility
REQUIREMENTS = [
    # Deep Learning Framework
    "torch==2.1.0",
    "torchvision==0.16.0", 
    "torchaudio==2.1.0",
    
    # Transformers and Model Hub
    "transformers==4.35.2",
    "tokenizers==0.15.0",
    "datasets==2.14.6",
    "huggingface-hub==0.19.4",
    
    # Quantization Libraries
    "bitsandbytes==0.41.3",
    "accelerate==0.24.1",
    "peft==0.6.2",  # For LoRA implementation
    
    # Scientific Computing
    "numpy==1.24.3",
    "scipy==1.11.4",
    "pandas==2.0.3",
    "matplotlib==3.7.2",
    "seaborn==0.12.2",
    
    # Jupyter and Analysis
    "jupyter==1.0.0",
    "ipykernel==6.25.2",
    "sympy==1.12",
    
    # Utilities
    "tqdm==4.66.1",
    "wandb==0.15.12",  # For experiment tracking
    "psutil==5.9.6",   # For system monitoring
    "GPUtil==1.4.0",   # For GPU monitoring
    
    # Development
    "pytest==7.4.3",
    "black==23.10.1",
    "isort==5.12.0",
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