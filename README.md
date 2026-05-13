<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/tlmtc-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/tlmtc-bright.png">
    <img
      src="assets/tlmtc-bright.png"
      alt="tlmtc: Transfer Learning for Multi-label Text Classification"
      style="width: 100%; max-width: 900px;"
    >
  </picture>
</p>

<hr>

**tlmtc** is a Python package that provides an out-of-the-box, end-to-end pipeline for fine-tuning pretrained encoder-only transformer models for robust multi-label text classification.

In real applications, text classification is rarely a simple single-label or mutually exclusive multiclass problem. In healthcare, a single clinical note may map to several ICD codes at once. In fintech products, assets may need multiple attributes and in-app user queries can express several intents. In the legal domain, one contract often bundles many clause types, and litigation documents can raise several legal issues. In defense, technical incident reports can span several failure modes, while threat-intelligence reports are tagged with multiple threat dimensions. Even in highly specialized domains like optics and photonics, defect reports are often labeled with multiple defect mechanisms. And in everyday customer support, tickets routinely span several issue categories that matter for routing, prioritization, and analytics.

**tlmtc** turns these use cases into a repeatable workflow. With a single function call or CLI command, it runs data preparation, hyperparameter tuning, model fine-tuning, and evaluation end to end.

**Key features**

- Parameter-efficient fine-tuning via LoRA (PEFT), with optional full fine-tuning
- Customizable Optuna-based hyperparameter tuning on a smaller proxy model
- Carry-over from proxy to larger target model with automatic learning-rate scaling
- Iterative stratified data splitting for multi-label datasets
- Custom class-weighted loss for handling label imbalance
- Global and label-specific threshold optimization for calibrated multi-label decisions
- Comprehensive evaluation suite with multi-label metrics
- Publication-ready reporting through tables and graphs
- Automatic persistence and reuse of data splits and Optuna studies
- Highly configurable via a Python API and CLI arguments
- PyPI package, CLI entrypoints, and Docker image for modular pipeline execution
- CPU and multi-GPU training support
