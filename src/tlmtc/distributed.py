"""Distributed process coordination for tlmtc workflows."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from accelerate import PartialState

R = TypeVar("R")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LocalContext:
    """Single-process policy for local prediction workflows."""

    is_main_process: bool = True

    def run_on_main(
        self,
        fn: Callable[..., R],
        *args: Any,
        sync: bool = False,
        **kwargs: Any,
    ) -> R:
        """Run a callable immediately in local execution."""
        return fn(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Distributed process policy for tlmtc orchestration."""

    state: PartialState

    @classmethod
    def create(
        cls,
        *,
        use_cpu: bool,
    ) -> Self:
        """Create a distributed context for the current workflow."""
        from accelerate import PartialState

        return cls(state=PartialState(cpu=use_cpu))

    @property
    def is_distributed(self) -> bool:
        """Whether the workflow runs with more than one process."""
        return self.state.num_processes > 1

    @property
    def is_main_process(self) -> bool:
        """Whether the current process is the main process."""
        return self.state.is_main_process

    @property
    def process_index(self) -> int:
        """Current process index."""
        return self.state.process_index

    @property
    def num_processes(self) -> int:
        """Number of active processes."""
        return self.state.num_processes

    def wait_for_everyone(self) -> None:
        """Synchronize all processes."""
        self.state.wait_for_everyone()

    def main_process_first(self) -> AbstractContextManager[None]:
        """Run a block on the main process before other processes enter it."""
        return self.state.main_process_first()

    def broadcast_value(
        self,
        value: T | None,
    ) -> T:
        """Broadcast a Python value from the main process."""
        if not self.is_distributed:
            if value is None:
                raise RuntimeError("Cannot broadcast a missing value outside distributed execution.")
            return value

        objects: list[T | None] = [value if self.is_main_process else None]

        try:
            from accelerate.utils import broadcast_object_list
        except ImportError:
            broadcast_object_list = None

        if broadcast_object_list is not None:
            broadcast_object_list(objects, from_process=0)
        else:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("torch.distributed is not initialized; cannot broadcast distributed value.")

            dist.broadcast_object_list(objects, src=0)

        result = objects[0]
        if result is None:
            raise RuntimeError("Distributed broadcast returned no value.")

        return result

    def resolve_run_id(
        self,
        run_id: str | None,
    ) -> str:
        """Resolve a single run identifier across all processes."""
        if not self.is_distributed:
            return run_id or uuid4().hex

        local_value = run_id
        main_value = (local_value or uuid4().hex) if self.is_main_process else None
        resolved = self.broadcast_value(main_value)

        if local_value is not None and local_value != resolved:
            raise RuntimeError(
                "run_id differs across distributed ranks. "
                f"Rank {self.process_index} received {local_value!r}, "
                f"but rank 0 resolved {resolved!r}."
            )

        return resolved

    def run_on_main(
        self,
        fn: Callable[..., R],
        *args: Any,
        sync: bool = False,
        **kwargs: Any,
    ) -> R | None:
        """Run a callable only on the main process."""
        result = fn(*args, **kwargs) if self.is_main_process else None

        if sync:
            self.wait_for_everyone()

        return result

    def warn_once(
        self,
        message: str,
    ) -> None:
        """Emit a warning only on the main process."""
        if self.is_main_process:
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    def warn_if_multi_gpu_without_launcher(
        self,
        *,
        use_cpu: bool,
    ) -> None:
        """Warn when multiple CUDA devices are visible without distributed launch."""
        import torch

        if use_cpu or self.is_distributed or not torch.cuda.is_available():
            return

        if torch.cuda.device_count() > 1:
            self.warn_once(
                "Multiple CUDA devices are visible, but tlmtc does not detect a distributed launcher. "
                "For modern multi-GPU execution, launch tlmtc with torchrun or accelerate launch."
            )


def create_prediction_context(
    *,
    inference_backend: Literal["torch", "onnx"],
    use_cpu: bool,
) -> LocalContext | DistributedContext:
    """Create the process context for a prediction backend."""
    if inference_backend == "onnx":
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("ONNX prediction backend does not support distributed launch.")
        return LocalContext()

    return DistributedContext.create(use_cpu=use_cpu)
