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
  <a href="https://github.com/saschagobel/tlmtc/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/saschagobel/tlmtc"></a>
  <a href="https://github.com/saschagobel/tlmtc/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/saschagobel/tlmtc/ci.yml?label=ci"></a>
  <a href="https://pypi.org/project/tlmtc/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tlmtc?label=pypi"></a>
  <a href="https://pypi.org/project/tlmtc/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/tlmtc"></a>
</p>

**tlmtc** (**T**ransfer **L**earning for **M**ulti-label **T**ext **C**lassification) is an opinionated Python package that provides production-ready, end-to-end workflows for fine-tuning pretrained encoder-only transformer models for robust multi-label text classification.

In applied settings, text classification is rarely a simple single-label task or a mutually exclusive multiclass problem. A clinical note may map to several ICD codes at once. A customer-support ticket can span multiple issue categories for routing, prioritization, and analytics. A contract may contain several clause types, and a litigation document can raise multiple legal issues. In RAG and LLM evaluation, a single answer may need several concurrent quality labels. A threat-intelligence report can mention multiple tactics, techniques, and vulnerabilities in the same document. Across domains, useful text labels often overlap, co-occur, and vary in prevalence.

**tlmtc** turns these use cases into repeatable training and prediction workflows. It prepares multi-label text data, fine-tunes transformer classifiers, tunes hyperparameters and decision thresholds, evaluates model performance, writes reports, and applies trained models to new data — all exposed through a small workflow-oriented API.

## Key features

- Parameter-efficient fine-tuning via LoRA (PEFT), with optional full fine-tuning
- Automatic detection of single-text and paired-text inputs for encoder and cross-encoder-style classification
- Customizable Optuna-based hyperparameter tuning, optionally on a smaller proxy model for efficiency
- Carry-over from proxy to larger target model with optional automatic learning-rate scaling
- Global and label-specific threshold optimization for calibrated multi-label decisions
- Iterative stratified data splitting for multi-label datasets, with automatic group-aware splitting
- Custom class-weighted loss for handling label imbalance
- Comprehensive evaluation suite with global and label-specific multi-label metrics
- Publication-ready reporting through tables and graphs
- End-to-end prediction workflow that reloads trained models, metadata, labels, and thresholds automatically
- Optional ONNX export artifacts for production inference deployment
- Automatic persistence and reuse of data splits, Optuna studies, trained models, thresholds, and run metadata
- Highly configurable through a small workflow-oriented Python API and CLI
- CPU and DDP-safe multi-GPU training and prediction support

## Installation

Install **tlmtc** into your active Python environment with the `train` extra:

```bash
pip install "tlmtc[train]"
```

The `train` extra installs the dependencies required for the full training workflow and includes the Torch-backed prediction runtime.

<details>
<summary><strong>Using uv?</strong></summary>

To install into an existing uv-managed environment, use:

```bash
uv pip install "tlmtc[train]"
```

If you are adding **tlmtc** as a dependency to a uv-managed Python project, use:

```bash
uv add "tlmtc[train]"
```

</details>

<details>
<summary><strong>Installing from source?</strong></summary>

Install the latest development version directly from GitHub:

```bash
pip install "tlmtc[train] @ git+https://github.com/saschagobel/tlmtc.git"
```

With uv, install into the active environment with:

```bash
uv pip install "tlmtc[train] @ git+https://github.com/saschagobel/tlmtc.git"
```

Or add it to a uv-managed project:

```bash
uv add "tlmtc[train] @ git+https://github.com/saschagobel/tlmtc.git"
```

</details>

## Quickstart

This quickstart uses a small synthetic paired-text dataset included in the repository. It mimics a requirements-engineering setting, where requirement records are paired with validation, commissioning, configuration, or field-service evidence and labeled for multiple concurrent issues. You will fine-tune a multi-label classifier and then apply the trained model to unlabeled examples.

Create a working directory and download the example data:

```bash
mkdir tlmtc-quickstart
cd tlmtc-quickstart

curl -L -o paired_example.csv \
  https://raw.githubusercontent.com/saschagobel/tlmtc/main/examples/paired_example.csv

curl -L -o paired_example_unlabeled.csv \
  https://raw.githubusercontent.com/saschagobel/tlmtc/main/examples/paired_example_unlabeled.csv
```

The code below assumes that both CSV files are in your current working directory. If you save them somewhere else, adjust the file paths accordingly.

<details>
<summary><strong>Using Windows PowerShell?</strong></summary>

```powershell
New-Item -ItemType Directory -Force -Path tlmtc-quickstart
Set-Location tlmtc-quickstart

Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/saschagobel/tlmtc/main/examples/paired_example.csv" `
  -OutFile "paired_example.csv"

Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/saschagobel/tlmtc/main/examples/paired_example_unlabeled.csv" `
  -OutFile "paired_example_unlabeled.csv"
```

</details>

Fine-tune a model:

```python
from tlmtc import train_tlmtc

train_tlmtc(
    labeled_data="paired_example.csv",
    target_name="Requirements Evidence Alignment",
    checkpoint="google/bert_uncased_L-2_H-128_A-2",
    tuning_trials=5,
    use_cpu=True,
)
```

This quickstart intentionally uses a tiny model and only five HPO trials to keep the demo lightweight. 

Use your fine-tuned model to run prediction on unlabeled data:

```python
from tlmtc import predict_tlmtc

predict_tlmtc(
    "paired_example_unlabeled.csv",
    use_cpu=True,
)
```

**tlmtc** writes training artifacts to `train_outputs/` and prediction artifacts to `prediction_outputs/`. Evaluation reports are written to `train_outputs/<run_id>/evaluation/`.

<details>
<summary><strong>Prefer the CLI?</strong></summary>

```bash
tlmtc train \
  --labeled-data paired_example.csv \
  --target-name "Requirements Evidence Alignment" \
  --checkpoint google/bert_uncased_L-2_H-128_A-2 \
  --tuning-trials 5 \
  --use-cpu

tlmtc predict \
  --unlabeled-data paired_example_unlabeled.csv \
  --use-cpu
```

</details>

<details>
<summary><strong>Try your own data</strong></summary>

Your labeled data must include:

- a `text` column
- at least two binary `label_`-prefixed columns

Add a `text_pair` column for paired-input classification. **tlmtc** detects and handles this automatically.

Add a `split_group` column when semantically related rows must not cross train, validation, and test splits.

For prediction, provide unlabeled data with the same input columns used during training. For a paired-text model, this means both `text` and `text_pair`.

</details>

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.
