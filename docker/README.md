# tlmtc container images

This directory contains the cloud-neutral container definitions for tlmtc batch
workflows.

## GPU HPO

`hpo-gpu.Dockerfile` provides `tlmtc[train]` with CUDA-enabled PyTorch for
single-process HPO on one visible GPU. The `linux/amd64` image requires the
NVIDIA Container Toolkit, uses the ordinary `tlmtc train` entrypoint, and does
not include ONNX export or ONNX Runtime dependencies.

Images are published as `ghcr.io/saschagobel/tlmtc-hpo-gpu`. Commit-SHA tags
identify validation images; releases additionally receive the exact project
version tag.

## GPU dependency policy

The image uses Python 3.12 and pins
`pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` by digest, providing PyTorch
2.13.0, CUDA 12.6, and cuDNN 9.
