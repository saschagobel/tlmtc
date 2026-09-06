## Project

`tlmtc` is a Python package for production-ready, end-to-end training and prediction workflows for transformer-based multi-label text classification, exposed through public Python APIs and a CLI.

Key architectural concerns are single- and paired-text data, configurable training and evaluation stages, persisted run artifacts, and Torch and ONNX inference backends.

## Development toolchain

- Use Python 3.12 for repository development; do not require features introduced in later Python versions.
- Use `uv` exclusively for repository development. Do not use `pip` directly or create ad hoc virtual environments.
- Use `uv sync` to install dependencies and `uv run` to execute Python and project tools.
- Use `uv add` and `uv remove` for dependency changes. Keep `pyproject.toml` and `uv.lock` synchronized.

## Architectural ownership

- Keep `src/tlmtc/api/` as the public orchestration layer. It may resolve settings, compose workflow stages, coordinate runtime output, and return public results; put domain and implementation logic in the corresponding modules.
- Keep `src/tlmtc/cli.py` as a thin adapter over the public API. Limit it to command definitions, CLI-specific parsing and presentation, and API invocation.
- Put user-configurable defaults and configuration validation in the corresponding Pydantic models in `src/tlmtc/settings.py`.
- Put output layout, filesystem path names, path-component validation, and path resolution in `src/tlmtc/paths.py`. Consume the resolved path objects instead of constructing output paths elsewhere.
- Treat the public Python APIs, CLI options, configuration models, and persisted artifact formats as compatibility-sensitive interfaces. Preserve existing behavior unless the task explicitly changes it.

## Implementation discipline

- Make the smallest clear change that fully satisfies the requested behavior, respects the architectural ownership above, and preserves public contracts unless the task explicitly changes them.
- Reuse existing project helpers and Python's standard library before adding custom code. Use already-declared dependencies only within the runtime extra or development dependency group that provides them.
- Introduce abstractions, configuration, fallbacks, or compatibility layers only when required by current behavior, a supported backend or dependency version, or the explicit task.
- Prefer readable, idiomatic Python over compressed or unnecessarily clever code.
- Inspect the relevant execution flow and its affected callers and contracts; avoid unrelated exploration.
- Run the narrowest tests and checks that cover the change, following the Verification section below and `CONTRIBUTING.md` when preparing a pull request.
- Report the outcome, verification, and material risks or limitations concisely.

## Verification

- Prefer the narrowest existing tests that cover the change over running the entire test suite.
- Run integration tests only when a change may affect end-to-end training or prediction behavior.

## Authoritative references

Consult these only when relevant to the task:

- `README.md` for public behavior, installation, usage, and examples.
- `CONTRIBUTING.md` for contribution and pull-request expectations.
- `pyproject.toml` for supported Python versions, declared dependencies, and tool configuration; `uv.lock` for the resolved dependency environment.
- `.github/workflows/ci.yml` for CI-specific setup and checks.
