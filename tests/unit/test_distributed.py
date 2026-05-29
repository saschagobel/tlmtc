"""Tests for distributed process coordination."""

import warnings
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import pytest

import tlmtc.distributed as distributed_module
from tlmtc.distributed import DistributedContext


@dataclass(slots=True)
class FakeState:
    """Small stand-in for Accelerate PartialState."""

    is_main_process: bool = True
    num_processes: int = 1
    process_index: int = 0
    wait_calls: int = 0
    main_process_first_calls: int = 0

    def wait_for_everyone(self) -> None:
        """Record a synchronization call."""
        self.wait_calls += 1

    def main_process_first(self) -> AbstractContextManager[None]:
        """Return a context manager and record that it was requested."""
        self.main_process_first_calls += 1
        return nullcontext()


@dataclass(frozen=True, slots=True)
class FakeUuid:
    """Deterministic uuid4 stand-in."""

    hex: str = "generated-run-id"


def make_context(
    *,
    is_main_process: bool = True,
    num_processes: int = 1,
    process_index: int = 0,
) -> tuple[DistributedContext, FakeState]:
    """Create a DistributedContext backed by a fake state object."""
    state = FakeState(
        is_main_process=is_main_process,
        num_processes=num_processes,
        process_index=process_index,
    )
    return DistributedContext(state=state), state


def test_create_initializes_partial_state_inside_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePartialState(FakeState):
        def __init__(self, *, cpu: bool) -> None:
            super().__init__()
            self.cpu = cpu

    monkeypatch.setattr(distributed_module, "PartialState", FakePartialState)

    context = DistributedContext.create(use_cpu=True)

    assert isinstance(context.state, FakePartialState)
    assert context.state.cpu is True


def test_exposes_process_state() -> None:
    context, _ = make_context(
        is_main_process=False,
        num_processes=4,
        process_index=2,
    )

    assert context.is_distributed is True
    assert context.is_main_process is False
    assert context.num_processes == 4
    assert context.process_index == 2


def test_wait_for_everyone_delegates_to_state() -> None:
    context, state = make_context()

    context.wait_for_everyone()

    assert state.wait_calls == 1


def test_main_process_first_returns_state_context_manager() -> None:
    context, state = make_context()

    with context.main_process_first():
        pass

    assert state.main_process_first_calls == 1


def test_broadcast_value_returns_local_value_when_not_distributed() -> None:
    context, _ = make_context(num_processes=1)

    assert context.broadcast_value("run-id") == "run-id"


def test_broadcast_value_rejects_missing_single_process_value() -> None:
    context, _ = make_context(num_processes=1)

    with pytest.raises(RuntimeError, match="Cannot broadcast a missing value"):
        context.broadcast_value(None)


def test_broadcast_value_uses_accelerate_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    seen_values: list[str | None] = []

    def fake_broadcast_object_list(
        objects: list[str | None],
        *,
        from_process: int,
    ) -> None:
        calls.append(from_process)
        seen_values.append(objects[0])
        objects[0] = "broadcast-run-id"

    monkeypatch.setattr(
        distributed_module,
        "_accelerate_broadcast_object_list",
        fake_broadcast_object_list,
    )
    context, _ = make_context(
        is_main_process=False,
        num_processes=2,
        process_index=1,
    )

    result = context.broadcast_value("ignored-local-value")

    assert result == "broadcast-run-id"
    assert calls == [0]
    assert seen_values == [None]


def test_broadcast_value_raises_when_broadcast_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_broadcast_object_list(
        objects: list[str | None],
        *,
        from_process: int,
    ) -> None:
        return None

    monkeypatch.setattr(
        distributed_module,
        "_accelerate_broadcast_object_list",
        fake_broadcast_object_list,
    )
    context, _ = make_context(
        is_main_process=False,
        num_processes=2,
        process_index=1,
    )

    with pytest.raises(RuntimeError, match="Distributed broadcast returned no value"):
        context.broadcast_value("ignored-local-value")


def test_broadcast_value_has_clear_torch_fallback_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distributed_module, "_accelerate_broadcast_object_list", None)
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: False)

    context, _ = make_context(
        is_main_process=True,
        num_processes=2,
        process_index=0,
    )

    with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
        context.broadcast_value("run-id")


def test_broadcast_value_uses_torch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    seen_values: list[str | None] = []

    def fake_broadcast_object_list(
        objects: list[str | None],
        *,
        src: int,
    ) -> None:
        calls.append(src)
        seen_values.append(objects[0])
        objects[0] = "fallback-run-id"

    monkeypatch.setattr(distributed_module, "_accelerate_broadcast_object_list", None)
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "broadcast_object_list", fake_broadcast_object_list)

    context, _ = make_context(
        is_main_process=False,
        num_processes=2,
        process_index=1,
    )

    result = context.broadcast_value("ignored-local-value")

    assert result == "fallback-run-id"
    assert calls == [0]
    assert seen_values == [None]


def test_resolve_run_id_preserves_explicit_single_process_value() -> None:
    context, _ = make_context(num_processes=1)

    assert context.resolve_run_id("explicit-run") == "explicit-run"


def test_resolve_run_id_generates_single_process_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distributed_module, "uuid4", FakeUuid)

    context, _ = make_context(num_processes=1)

    assert context.resolve_run_id(None) == "generated-run-id"


def test_resolve_run_id_broadcasts_generated_main_process_value(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, str | None]] = []

    def fake_broadcast_object_list(
        objects: list[str | None],
        *,
        from_process: int,
    ) -> None:
        calls.append((from_process, objects[0]))

    monkeypatch.setattr(distributed_module, "uuid4", FakeUuid)
    monkeypatch.setattr(
        distributed_module,
        "_accelerate_broadcast_object_list",
        fake_broadcast_object_list,
    )

    context, _ = make_context(
        is_main_process=True,
        num_processes=2,
        process_index=0,
    )

    assert context.resolve_run_id(None) == "generated-run-id"
    assert calls == [(0, "generated-run-id")]


def test_resolve_run_id_rejects_mismatched_explicit_distributed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_broadcast_object_list(
        objects: list[str | None],
        *,
        from_process: int,
    ) -> None:
        objects[0] = "rank-zero-run"

    monkeypatch.setattr(
        distributed_module,
        "_accelerate_broadcast_object_list",
        fake_broadcast_object_list,
    )

    context, _ = make_context(
        is_main_process=False,
        num_processes=2,
        process_index=1,
    )

    with pytest.raises(RuntimeError, match="run_id differs across distributed ranks"):
        context.resolve_run_id("rank-one-run")


def test_run_on_main_executes_callable_on_main_process() -> None:
    context, state = make_context(is_main_process=True)

    result = context.run_on_main(lambda value: value + 1, 2)

    assert result == 3
    assert state.wait_calls == 0


def test_run_on_main_skips_callable_on_non_main_process() -> None:
    context, state = make_context(is_main_process=False)
    calls: list[str] = []

    result = context.run_on_main(calls.append, "called")

    assert result is None
    assert calls == []
    assert state.wait_calls == 0


def test_run_on_main_synchronizes_only_when_requested() -> None:
    context, state = make_context(is_main_process=False)

    context.run_on_main(lambda: None, sync=True)

    assert state.wait_calls == 1


def test_warn_once_emits_on_main_process() -> None:
    context, _ = make_context(is_main_process=True)

    with pytest.warns(RuntimeWarning, match="main warning"):
        context.warn_once("main warning")


def test_warn_once_skips_non_main_process() -> None:
    context, _ = make_context(is_main_process=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        context.warn_once("non-main warning")


def test_warn_if_multi_gpu_without_launcher_emits_on_main_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.torch.cuda, "device_count", lambda: 2)

    context, _ = make_context(
        is_main_process=True,
        num_processes=1,
        process_index=0,
    )

    with pytest.warns(RuntimeWarning, match="Multiple CUDA devices are visible"):
        context.warn_if_multi_gpu_without_launcher(use_cpu=False)


@pytest.mark.parametrize(
    ("use_cpu", "num_processes", "cuda_available", "device_count"),
    [
        (True, 1, True, 2),
        (False, 2, True, 2),
        (False, 1, False, 2),
        (False, 1, True, 1),
    ],
)
def test_warn_if_multi_gpu_without_launcher_skips_when_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
    use_cpu: bool,
    num_processes: int,
    cuda_available: bool,
    device_count: int,
) -> None:
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(distributed_module.torch.cuda, "device_count", lambda: device_count)

    context, _ = make_context(
        is_main_process=True,
        num_processes=num_processes,
        process_index=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        context.warn_if_multi_gpu_without_launcher(use_cpu=use_cpu)
