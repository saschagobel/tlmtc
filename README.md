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


<p align="center">
  <a href="https://github.com/saschagobel/tlmtc/blob/main/LICENSE.md">
    <img alt="License" src="https://img.shields.io/github/license/saschagobel/tlmtc">
  </a>
  <a href="https://github.com/saschagobel/tlmtc/actions">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/saschagobel/tlmtc/ci.yml?label=ci">
  </a>
</p>

**tlmtc** (**T**ransfer **L**earning for **M**ulti-label **T**ext **C**lassification) is an opinionated Python package that provides production-ready, end-to-end workflows for fine-tuning pretrained encoder-only transformer models for robust multi-label text classification.

In applied settings, text classification is rarely a simple single-label task or a mutually exclusive multiclass problem. A clinical note may map to several ICD codes at once. A customer-support ticket can span multiple issue categories for routing, prioritization, and analytics. A contract may contain several clause types, and a litigation document can raise multiple legal issues. In RAG and LLM evaluation, a single answer may need several concurrent quality labels. A threat-intelligence report can mention multiple tactics, techniques, and vulnerabilities in the same document. Across domains, useful text labels often overlap, co-occur, and vary in prevalence.

**tlmtc** turns these use cases into repeatable training and prediction workflows. It prepares multi-label text data, fine-tunes transformer classifiers, tunes hyperparameters and decision thresholds, evaluates model performance, writes reports, and applies trained models to new data  — all exposed through a small workflow-oriented API.

## Key features

- Parameter-efficient fine-tuning via LoRA (PEFT), with optional full fine-tuning
- Automatic detection of single-text and paired-text inputs for encoder and cross-encoder-style classification
- Customizable Optuna-based hyperparameter tuning, optionally on a smaller proxy model for efficiency
- Carry-over from proxy to larger target model with optional automatic learning-rate scaling
- Global and label-specific threshold optimization for calibrated multi-label decisions
- Iterative stratified data splitting for multi-label datasets
- Custom class-weighted loss for handling label imbalance
- Comprehensive evaluation suite with global and label-specific multi-label metrics
- Publication-ready reporting through tables and graphs
- End-to-end prediction workflow that reloads trained models, metadata, labels, and thresholds automatically
- Automatic persistence and reuse of data splits, Optuna studies, trained models, thresholds, and run metadata
- Highly configurable through a small workflow-oriented Python API and CLI
- CPU and multi-GPU training and prediction support
