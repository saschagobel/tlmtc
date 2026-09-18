# Shared GPU runner components

This directory contains the host setup shared by the disposable HPO and training GPU runners. It is consumed by their Terraform and cloud-init configurations.

## GPU host bootstrap

[`scripts/bootstrap-gpu-host.sh`](scripts/bootstrap-gpu-host.sh) prepares an Ubuntu GPU host for tlmtc workloads. It:

- partitions and mounts the attached data disk at `/mnt/tlmtc`
- creates the input, work, cache, and container-runtime directories
- installs the pinned NVIDIA driver, Docker, NVIDIA Container Toolkit, and AzCopy versions
- configures Docker and containerd to use the data disk
- schedules a reboot to activate the NVIDIA driver

Both `hpo-gpu` and `train-gpu` embed and run this script through cloud-init. Changes to it therefore affect both runner configurations.
