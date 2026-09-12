# tlmtc container images

This directory contains the cloud-neutral container definitions for tlmtc batch
workflows.

## GPU HPO

`hpo-gpu.Dockerfile` provides `tlmtc[train]` with CUDA-enabled PyTorch for
single-process HPO on one visible GPU. The `linux/amd64` image requires the
NVIDIA Container Toolkit, uses the ordinary `tlmtc train` entrypoint, and does
not include ONNX export or ONNX Runtime dependencies.

Images are published as `ghcr.io/saschagobel/tlmtc-hpo-gpu`. Candidate images
use `<project-version>-candidate-<short-sha>-<run-id>-<attempt>` tags; releases
use the exact project version tag.

## GPU Training

`train-gpu.Dockerfile` provides `tlmtc[train,onnx-export]` with CUDA-enabled
PyTorch for final fine-tuning and ONNX export. The `linux/amd64` image requires
the NVIDIA Container Toolkit and supports single-GPU training and single-node
multi-GPU DDP through an external launcher. It uses the ordinary `tlmtc train`
entrypoint and preserves the application's workflow defaults.

The intended image repository is `ghcr.io/saschagobel/tlmtc-train-gpu`, using
the same candidate and release tag scheme as the HPO image.

## GPU dependency policy

Both GPU images use Python 3.12 and pin
`pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` by digest, providing PyTorch
2.13.0, CUDA 12.6, and cuDNN 9.
