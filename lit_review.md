# Literature Review: Quantization Noise Regularization and LoRA Theory

## 1. IR-QLoRA: Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
**Citation:** Qin, H., Ma, X., Zheng, X., Li, X., Zhang, Y., Liu, S., Luo, J., Liu, X., & Magno, M. (2024). Accurate LoRA-Finetuning Quantization of LLMs via Information Retention. *arXiv preprint arXiv:2402.05445*.

**Key Equation/Claim:** Proposes Information Calibration Quantization and Information Elastic Connection for improved 4-bit LoRA fine-tuning.

**Relevance:** Directly addresses quantization-LoRA interaction, achieving 1.4% improvement on MMLU with 4-bit LLaMA-7B while maintaining efficiency.

## 2. QR-Adaptor: Efficient Fine-Tuning of Quantized Models via Adaptive Rank and Bitwidth
**Citation:** Zhou, C., Han, S., Zhang, S., Zhou, Y., Zhang, W., & Jin, C. (2025). Efficient Fine-Tuning of Quantized Models via Adaptive Rank and Bitwidth. *arXiv preprint arXiv:2505.03802*.

**Key Equation/Claim:** Joint optimization of quantization precision and LoRA rank per layer, treating it as a discrete optimization problem guided by downstream performance.

**Relevance:** Provides theoretical framework for precision-rank allocation with 4.89% accuracy improvement on GSM8K, sometimes outperforming 16-bit models.

## 3. QReg: On Regularization Effects of Quantization
**Citation:** AskariHemmat, M. H., et al. (2022). QReg: On Regularization Effects of Quantization. *arXiv preprint arXiv:2206.12372*.

**Key Equation/Claim:** Models quantization as additive noise: $w_q = w + \epsilon$ where $\epsilon \sim U(-\Delta/2,\Delta/2)$ and derives regularization bound linking bit-width to gradient variance.

**Relevance:** Foundational work establishing quantization as regularization with theoretical analysis showing 8-bit provides reliable regularization across vision tasks.

## 4. Can Post-Training Quantization Benefit from an Additional QLoRA Integration?
**Citation:** Zhu, X., Khasanova, E., & Chen, C. (2025). Can Post-Training Quantization Benefit from an Additional QLoRA Integration? *NAACL 2025*.

**Key Equation/Claim:** Integration of 4-bit PTQ with QLoRA outperforms standard PTQ and sometimes 16-bit full-parameter fine-tuning.

**Relevance:** Demonstrates practical benefits of PTQ-QLoRA integration for resource-constrained deployment without performance compromise.

## 5. QGen: On the Ability to Generalize in Quantization Aware Training
**Citation:** AskariHemmat, M. H., et al. (2024). QGen: On the Ability to Generalize in Quantization Aware Training. *arXiv preprint arXiv:2404.11769*.

**Key Equation/Claim:** Derives generalization bound for quantized models conditioned on quantization noise amount, connecting loss landscape sharpness to generalization.

**Relevance:** Provides theoretical foundation linking quantization noise to generalization with validation on 2000+ models across CIFAR and ImageNet.

## 6. Sharp Generalization Bounds for Foundation Models with Asymmetric Randomized Low-Rank Adapters
**Citation:** Kratsios, A., Cheng, T. S., Lucchi, A., & Borde, H. S. de O. (2025). Sharp Generalization Bounds for Foundation Models with Asymmetric Randomized Low-Rank Adapters. *arXiv preprint arXiv:2506.14530*.

**Key Equation/Claim:** Sample complexity bound of $\tilde{\mathcal{O}}(\sqrt{r}/\sqrt{N})$ for rank-r LoRA with matching lower bound $\mathcal{O}(1/\sqrt{N})$.

**Relevance:** Provides fundamental theoretical limits for LoRA generalization, crucial for understanding precision-rank trade-offs.

## 7. A Statistical Framework for Low-bitwidth Training of Deep Neural Networks
**Citation:** Chen, J., Gai, Y., Yao, Z., Mahoney, M. W., & Gonzalez, J. E. (2020). A Statistical Framework for Low-bitwidth Training of Deep Neural Networks. *arXiv preprint arXiv:2010.14298*.

**Key Equation/Claim:** FQT gradient is unbiased estimator of QAT gradient with variance decomposition showing quantization variance scales as $O(2^{-2b})$.

**Relevance:** Establishes theoretical foundation for low-precision training with variance analysis directly applicable to our bit-width error bounds.

## 8. SC-LoRA: Balancing Efficient Fine-tuning and Knowledge Preservation via Subspace-Constrained LoRA
**Citation:** Luo, M., Kuang, F., Wang, Y., Liu, Z., & He, T. (2025). SC-LoRA: Balancing Efficient Fine-tuning and Knowledge Preservation via Subspace-Constrained LoRA. *arXiv preprint arXiv:2505.23724*.

**Key Equation/Claim:** Constrains LoRA adapter output in low-rank subspace to balance fine-tuning efficiency and knowledge preservation.

**Relevance:** Addresses knowledge forgetting in LoRA adaptation, relevant to understanding how quantization noise affects learned representations.

## 9. GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning
**Citation:** Zhou, S., Wang, S., Yuan, Z., Shi, M., Shang, Y., & Yang, D. (2025). GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning. *arXiv preprint arXiv:2502.12913*.

**Key Equation/Claim:** Group-Shared Exponents Integer format eliminates floating-point operations in both inference and training while maintaining accuracy comparable to BF16.

**Relevance:** Demonstrates extreme quantization (integer-only) for on-device fine-tuning with 1.85x memory reduction and 5x power savings vs FP8.

## 10. QuIP: 2-Bit Quantization of Large Language Models With Guarantees
**Citation:** Chee, J., Cai, Y., Kuleshov, V., & De Sa, C. (2024). QuIP: 2-Bit Quantization of Large Language Models With Guarantees. *arXiv preprint arXiv:2307.13304*.

**Key Equation/Claim:** Quantization with incoherence processing using random orthogonal matrices to ensure weight and Hessian incoherence for stable 2-bit quantization.

**Relevance:** Provides theoretical analysis for extreme quantization with guarantees, showing viable 2-bit results and establishing theoretical framework applicable to our bounds.

## 11. How to Parameterize Asymmetric Quantization Ranges for Quantization-Aware Training
**Citation:** You, J., Park, M., Lee, K., An, S., Patel, C., & Nage, M. (2024). How to Parameterize Asymmetric Quantization Ranges for Quantization-Aware Training. *arXiv preprint arXiv:2404.16898*.

**Key Equation/Claim:** Comparative analysis of three asymmetric quantization parameterizations (scale-offset, min-max, beta-gamma) with focus on stability and convergence.

**Relevance:** Provides practical insights into quantization parameterization affecting training dynamics, relevant to our theoretical bounds derivation.

## Summary of Key Insights

1. **Quantization as Regularization:** Multiple works (QReg, QGen) establish quantization noise as effective regularization with theoretical bounds.

2. **LoRA-Quantization Synergy:** Recent works (IR-QLoRA, QR-Adaptor) show joint optimization of precision and rank yields superior results.

3. **Theoretical Foundations:** Statistical frameworks (Chen et al.) and generalization bounds (Kratsios et al.) provide mathematical basis for our derivations.

4. **Practical Validation:** Extensive empirical evidence across vision and NLP tasks supports theoretical predictions about bit-width–error relationships.

5. **Extreme Quantization:** Works like QuIP and GSQ-Tuning demonstrate viability of 2-4 bit training with proper theoretical foundations. 