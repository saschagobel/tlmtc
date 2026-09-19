# syntax=docker/dockerfile:1

ARG UV_BUILDER_IMAGE="ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58"
ARG CPU_BASE_IMAGE="python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254"

FROM ${UV_BUILDER_IMAGE} AS package-builder

ENV UV_NO_PROGRESS=1

WORKDIR /build

COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY src ./src

RUN mkdir -p /dist \
    && uv export \
        --locked \
        --extra onnx-runtime \
        --no-emit-project \
        --no-header \
        --no-annotate \
        --output-file /dist/requirements-onnx-runtime.txt \
    && uv build --wheel --out-dir /dist

FROM ${CPU_BASE_IMAGE} AS runtime

ARG IMAGE_VERSION=dev
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="tlmtc Predict CPU" \
      org.opencontainers.image.description="CPU batch prediction for tlmtc with ONNX Runtime" \
      org.opencontainers.image.source="https://github.com/saschagobel/tlmtc" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="MIT"

ENV HOME=/home/tlmtc \
    HF_HOME=/home/tlmtc/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Mount build artifacts and uv only while installing, keeping them out of runtime layers.
RUN --mount=from=package-builder,source=/dist,target=/tmp/tlmtc-dist \
    --mount=from=package-builder,source=/usr/local/bin/uv,target=/usr/local/bin/uv \
    uv pip install \
        --system \
        --no-cache \
        --require-hashes \
        --requirement /tmp/tlmtc-dist/requirements-onnx-runtime.txt \
    && uv pip install \
        --system \
        --no-cache \
        --no-deps \
        /tmp/tlmtc-dist/tlmtc-*.whl \
    && uv pip check --system

# Verify that the prediction API imports without a training or export stack.
RUN python - <<'PY'
import importlib.util

import datasets
import onnxruntime
import pandera
import tlmtc
import transformers
from tlmtc.api import predict_tlmtc

assert callable(predict_tlmtc)
assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
assert "CUDAExecutionProvider" not in onnxruntime.get_available_providers()
for package in ("torch", "peft", "accelerate", "optuna", "olive", "onnx", "sklearn", "matplotlib"):
    assert importlib.util.find_spec(package) is None, f"Unexpected training/export dependency: {package}"

print(
    {
        "tlmtc": tlmtc.__version__,
        "onnxruntime": onnxruntime.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "pandera": pandera.__version__,
    }
)
PY

RUN groupadd --gid 10001 tlmtc \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --create-home \
        --home-dir /home/tlmtc \
        --shell /usr/sbin/nologin \
        tlmtc \
    && mkdir -p /workspace /home/tlmtc/.cache/huggingface \
    && chown -R 10001:10001 /workspace /home/tlmtc

WORKDIR /workspace
USER 10001:10001

RUN tlmtc --version \
    && tlmtc predict --help >/dev/null

ENTRYPOINT ["tlmtc", "predict"]
CMD ["--help"]
