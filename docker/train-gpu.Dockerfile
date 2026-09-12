# syntax=docker/dockerfile:1

# Keep the GPU base and the Torch/CUDA checks below in sync.
ARG UV_BUILDER_IMAGE="ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58"
ARG GPU_BASE_IMAGE="pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"

FROM ${UV_BUILDER_IMAGE} AS package-builder

ENV UV_NO_PROGRESS=1

WORKDIR /build

COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY src ./src

# Use CUDA-enabled Torch from the runtime base, not the lock's CPU-only source.
RUN mkdir -p /dist \
    && uv export \
        --locked \
        --extra train \
        --extra onnx-export \
        --no-emit-project \
        --no-emit-package torch \
        --no-header \
        --no-annotate \
        --output-file /dist/requirements-train-onnx.txt \
    && uv build --wheel --out-dir /dist

FROM ${GPU_BASE_IMAGE} AS runtime

ARG IMAGE_VERSION=dev
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="tlmtc Train GPU" \
      org.opencontainers.image.description="Final GPU fine-tuning for tlmtc with single-node DDP support and ONNX export" \
      org.opencontainers.image.source="https://github.com/saschagobel/tlmtc" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="MIT"

ENV HOME=/home/tlmtc \
    HF_HOME=/home/tlmtc/.cache/huggingface \
    MPLCONFIGDIR=/home/tlmtc/.cache/matplotlib \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY --from=package-builder /dist /tmp/tlmtc-dist

# The base marks Python as externally managed and includes spin, which conflicts
# with the locked Click version.
RUN python -m pip uninstall \
        --yes \
        --break-system-packages \
        spin \
    && python -m pip install \
        --break-system-packages \
        --no-cache-dir \
        --require-hashes \
        --requirement /tmp/tlmtc-dist/requirements-train-onnx.txt \
    && python -m pip install \
        --break-system-packages \
        --no-cache-dir \
        --no-deps \
        /tmp/tlmtc-dist/tlmtc-*.whl \
    && python -m pip check \
    && rm -rf /tmp/tlmtc-dist

# Verify the GPU training stack and ONNX export dependencies.
RUN python - <<'PY'
import importlib.metadata

import accelerate
import datasets
import olive
import onnx
import onnxruntime
import optuna
import peft
import sklearn
import tlmtc
import torch
import transformers
from olive.cli.api import optimize

assert importlib.metadata.version("torch") == "2.13.0+cu126"
assert torch.version.cuda == "12.6"
assert not torch.__version__.endswith("+cpu")
assert callable(optimize)

print(
    {
        "tlmtc": tlmtc.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "optuna": optuna.__version__,
        "scikit_learn": sklearn.__version__,
        "olive-ai": importlib.metadata.version("olive-ai"),
        "onnx": importlib.metadata.version("onnx"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
    }
)
PY

RUN tlmtc --version \
    && tlmtc train --help >/dev/null

RUN groupadd --gid 10001 tlmtc \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --create-home \
        --home-dir /home/tlmtc \
        --shell /usr/sbin/nologin \
        tlmtc \
    && mkdir -p \
        /workspace \
        /home/tlmtc/.cache/huggingface \
        /home/tlmtc/.cache/matplotlib \
    && chown -R 10001:10001 /workspace /home/tlmtc

WORKDIR /workspace
USER 10001:10001

ENTRYPOINT ["tlmtc", "train"]
CMD ["--help"]
