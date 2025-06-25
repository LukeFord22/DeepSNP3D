# DNABERT-2: Potential Upgrades and Modernizations (as of 2025)

This document lists recent advances in transformer architectures and sequence modeling that could be integrated into DNABERT-2 for improved performance, efficiency, and flexibility.

## 2. Rotary Position Embeddings (RoPE)
- **What:** Replace ALiBi with RoPE, as used in Llama and other recent models.
- **Benefit:** Often improves performance on sequence tasks, especially for long-range dependencies.
- **Difficulty:** Requires changes to attention code and careful tensor handling.

## 3. LoRA (Low-Rank Adaptation)
- **What:** Add support for parameter-efficient fine-tuning by injecting LoRA adapters into attention/MLP layers.
- **Benefit:** Enables fast, low-resource fine-tuning for new tasks or datasets.
- **Difficulty:** Requires module wrapping and training logic changes.

## 4. HuggingFace Transformers API Integration
- **What:** Refactor model to fully support the latest HuggingFace Trainer and AutoModel APIs.
- **Benefit:** Easier training, evaluation, deployment, and sharing.
- **Difficulty:** May require significant refactoring for custom features (unpadding, custom attention).

## 5. Other Modernizations
- RMSNorm, SwiGLU, and other normalization/activation improvements.
- Support for long-context models (Longformer, BigBird, Mamba, Hyena).
- Improved regularization and training tricks from recent LLMs.

---

**Note:** Each upgrade brings both benefits and integration challenges. See the main README or consult the codebase for implementation details and compatibility notes.
