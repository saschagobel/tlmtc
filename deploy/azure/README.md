# Azure deployments

Terraform configurations and deployment guides for running tlmtc workflows on Azure.

- [`shared/`](shared/): persistent Blob Storage for pipeline inputs and outputs.
- [`vm/`](vm/): Azure VM deployments for tlmtc workflows.

Deploy shared storage first, then follow the [VM deployment guides](vm/). Shared storage persists independently of compute resources.
