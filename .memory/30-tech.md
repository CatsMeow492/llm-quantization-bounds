# Technology Landscape

## Technology Stack
- **Language**: Python ≥ 3.10
- **Frameworks/Libraries**: PyTorch ≥ 2.1, Transformers, PEFT, bitsandbytes, pandas, matplotlib, LaTeX (latexmk).

## Development Environment
- OS: macOS / Linux
- Version control: Git
- Recommended: `conda` or `venv` with requirements.txt / `setup.py`.

## Dependencies
- GPU: NVIDIA with CUDA 11.x (if available)
- CPU-only mode supported (slow)

## Build & Deployment
- `python setup.py install` sets up environment.
- Experiments invoked via CLI.
- Paper compiled with `latexmk -pdf`.

## Environment Configuration
- Environment variables for dataset/model cache paths.

## Tool Chain
- Jupyter Notebook for derivations.
- VSCode / IDE for code editing.
- GitHub Actions (future) for CI on tests and lint.

*Last updated: Initial memory initialization* 