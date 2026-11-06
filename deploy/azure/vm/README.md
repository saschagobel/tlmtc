# Azure VM deployments

Terraform configurations and deployment guides for running tlmtc workflows on Azure VMs, using [shared Azure storage](../shared/) for pipeline inputs and outputs.

- [`hpo-gpu/`](hpo-gpu/): disposable GPU VMs for hyperparameter optimization.
- [`train-gpu/`](train-gpu/): disposable GPU VMs for fine-tuning and ONNX export, with or without a prior HPO run.
- [`predict-cpu/`](predict-cpu/): disposable CPU VMs for ONNX batch prediction from a published training run.
- [`shared/`](shared/): GPU host bootstrap shared by the HPO and training deployments.

Each deployment guide covers provisioning, workload execution, and VM lifecycle commands.
