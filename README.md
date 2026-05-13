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

**tlmtc** (**T**ransfer **L**earning for **M**ulti-label **T**ext **C**lassification) is an opinionated Python package that provides production-ready, end-to-end workflows for fine-tuning pretrained encoder-only transformer models for robust multi-label text classification.

In applied settings, text classification is rarely a simple single-label task or a mutually exclusive multiclass problem. A clinical note may map to several ICD codes at once. A customer-support ticket can span multiple issue categories for routing, prioritization, and analytics. A contract may contain several clause types, and a litigation document can raise multiple legal issues. In RAG and LLM evaluation, a single answer may need several concurrent quality labels. A threat-intelligence report can mention multiple tactics, techniques, and vulnerabilities in the same document. Across domains, useful text labels often overlap, co-occur, and vary in prevalence.

**tlmtc** turns these use cases into repeatable training and prediction workflows. It prepares multi-label text data, fine-tunes transformer classifiers, tunes hyperparameters and decision thresholds, evaluates model performance, writes reports, and applies trained models to new data  — all exposed through a small workflow-oriented API.

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
