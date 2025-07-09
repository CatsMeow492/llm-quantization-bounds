# Domain & Project Knowledge

## Domain Concepts
- **Quantization Noise**: Error introduced when representing weights with limited bit-width.
- **LoRA (Low-Rank Adaptation)**: Technique for fine-tuning LLMs by training low-rank updates.
- **Bit-Width (b)**: Number of bits used to represent each weight.

## Relationship Map
Quantization Noise ∝ 2^{-b} → Affects Gradient Variance → Influences Fine-Tuning Error Bound.

## Key Resources
- QLoRA paper (Dettmers et al., 2023)
- bitsandbytes documentation
- PEFT library

## Project Best Practices
- Pin package versions for reproducibility.
- Log all experiment metadata.

## FAQ
**Q:** Why choose 4-bit precision?
**A:** It offers maximum efficiency while remaining empirically viable; the project seeks to bound its theoretical limits.

## Implicit Knowledge
Gradient variance amplification due to quantization can often be mitigated by low-rank adapters.

*Last updated: Initial memory initialization* 