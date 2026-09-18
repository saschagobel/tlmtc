# Azure VM deployments

Terraform configurations and deployment guides for running tlmtc workflows on Azure VMs, using [shared Azure storage](../shared/README.md) for pipeline inputs and outputs.

- [`hpo-gpu/`](hpo-gpu/README.md): disposable GPU VMs for hyperparameter optimization.
- [`train-gpu/`](train-gpu/README.md): disposable GPU VMs for fine-tuning and ONNX export, with or without a prior HPO run.
- [`shared/`](shared/README.md): GPU host bootstrap shared by the HPO and training deployments.

Each deployment guide covers provisioning, workload execution, and VM lifecycle commands.
