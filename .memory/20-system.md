# System Architecture

## System Overview
The project comprises experiment scripts, analysis utilities, and LaTeX paper files orchestrated to derive, validate, and present a bit-width error bound for low-precision fine-tuning.

```
Experiment Scripts  --->  Results (JSON)  --->  Analysis/Plots  --->  Paper Figures
                        |                                   |
                        |-- Config & Logs ------------------|
```

## Component Breakdown
1. **setup.py** – Installs dependencies and downloads models.
2. **run_ft.py** – Runs fine-tuning at specified precision and seed.
3. **analyze.py** – Aggregates metrics and generates plots.
4. **paper/** – LaTeX source, figures, and bibliography.

## Design Patterns
- Modular CLI utilities.
- Reproducible environment via pinned dependencies.

## Data Flow
`run_ft.py` → saves metrics to `results/metrics_{bits}_{seed}.json` → `analyze.py` merges into DataFrame → plots exported to `paper/figures/`.

## Integration Points
- bitsandbytes for quantization.
- PEFT library for LoRA adapters.

## Architectural Decisions
Targeting DialoGPT-medium for resource efficiency; scalable to larger models if resources permit.

## Non-Functional Requirements
- Reproducibility.
- Scripted automation (no manual clicks).
- Runs on CUDA-enabled GPUs; fallback CPU support (slow).

*Last updated: Initial memory initialization* 