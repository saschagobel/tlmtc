# Azure deployments

Terraform configurations and deployment guides for running tlmtc workflows on Azure.

- [`shared/`](shared/README.md): persistent Blob Storage for pipeline inputs and outputs.
- [`vm/`](vm/README.md): disposable GPU VM runners for HPO and training.

Deploy shared storage first, then follow the [VM deployment guides](vm/README.md). Shared storage persists independently of compute resources.
