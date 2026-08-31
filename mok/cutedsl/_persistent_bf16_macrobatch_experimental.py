"""Private macrobatch sweep seam for the accepted persistent BF16 forward.

The public ``persistent_bf16`` specialization stays frozen at B=32768.  This
module changes only that ring extent for a bounded experiment and reuses the
accepted device kernel verbatim.  Executors are cached; state, pointer wrappers,
and the caller stream are rebound for every invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Final

from . import persistent_bf16 as _accepted


MACROBATCH_CANDIDATES: Final = (4096, 8192, 16384, 32768)
MINIBATCH_ROWS: Final = 4096
NUM_COMM_SMS: Final = 40
_CACHE_EPOCH: Final = "r53-private-macrobatch-screen-v1"


@dataclass(frozen=True)
class ExperimentalMacrobatchPlan:
    macrobatch_size: int
    minibatch_size: int = MINIBATCH_ROWS
    num_comm_sms: int = NUM_COMM_SMS

    def validate(self) -> None:
        if (
            type(self.macrobatch_size) is not int
            or self.macrobatch_size not in MACROBATCH_CANDIDATES
        ):
            raise NotImplementedError(
                "experimental persistent BF16 macro must be 4096, 8192, "
                "16384, or 32768"
            )
        if self.minibatch_size != MINIBATCH_ROWS:
            raise NotImplementedError("experimental persistent BF16 mini is fixed at 4096")
        if self.num_comm_sms != NUM_COMM_SMS:
            raise NotImplementedError("experimental persistent BF16 comm SMs are fixed at 40")


_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()


def _output_shapes(
    num_local_tokens: int,
    macrobatch_size: int,
) -> tuple[tuple[int, ...], ...]:
    if (
        type(num_local_tokens) is not int
        or num_local_tokens < 2 * _accepted.ROW_ALIGNMENT
        or num_local_tokens % _accepted.ROW_ALIGNMENT
    ):
        raise ValueError("num_local_tokens must be a multiple of 256 at least 512")
    ExperimentalMacrobatchPlan(macrobatch_size).validate()
    return (
        (macrobatch_size, _accepted.HIDDEN_SIZE),
        (num_local_tokens, _accepted.INTERMEDIATE_SIZE),
        (macrobatch_size, _accepted.INTERMEDIATE_SIZE),
        (num_local_tokens, _accepted.INTERMEDIATE_SIZE),
        (macrobatch_size, _accepted.INTERMEDIATE_SIZE),
        (num_local_tokens, _accepted.INTERMEDIATE_SIZE),
        (macrobatch_size, _accepted.INTERMEDIATE_SIZE),
        (num_local_tokens, _accepted.HIDDEN_SIZE),
        (macrobatch_size, _accepted.HIDDEN_SIZE),
    )


def _prepare_state(*, num_local_tokens: int, schedule_capacity: int, device, plan):
    import torch

    outputs = tuple(
        torch.empty(shape, dtype=torch.bfloat16, device=device)
        for shape in _output_shapes(num_local_tokens, plan.macrobatch_size)
    )
    lengths = _accepted.persistent_bf16_counter_lengths(
        num_local_tokens,
        schedule_capacity,
    )
    counters = tuple(
        torch.zeros(length, dtype=torch.int32, device=device)
        for length in (
            lengths.gate_up_tile_ready,
            lengths.hidden_row_block_ready,
            lengths.x_routed_ready,
            lengths.y_routed_ready,
            lengths.y_routed_done,
        )
    )
    return _accepted.PersistentBf16State(*outputs, *counters)


def _validate_call(workspace, schedule, weights, state, plan) -> None:
    """Reject unsupported metadata before importing CUTLASS or compiling."""

    import torch

    plan.validate()
    required_workspace = (
        "device",
        "ep_size",
        "ep_rank",
        "num_local_tokens",
        "hidden_size",
        "topk",
        "schedule_capacity",
        "x_buffer",
        "x_buffer_ptrs",
        "combine_buffer",
        "combine_buffer_ptrs",
    )
    missing = tuple(name for name in required_workspace if not hasattr(workspace, name))
    if missing:
        raise TypeError(f"workspace is missing fields: {missing}")
    if (
        workspace.ep_size != _accepted.EP_SIZE
        or workspace.hidden_size != _accepted.HIDDEN_SIZE
        or workspace.topk != _accepted.TOPK
    ):
        raise NotImplementedError("experimental persistent BF16 requires EP8 H4096 topk10")
    if torch.cuda.get_device_capability(workspace.device) != (10, 3):
        raise NotImplementedError("experimental persistent BF16 requires B300/SM103")
    if not 0 <= workspace.ep_rank < _accepted.EP_SIZE:
        raise ValueError("workspace.ep_rank is outside EP8")
    for name in ("x_buffer_ptrs", "combine_buffer_ptrs"):
        pointers = getattr(workspace, name)
        if (
            not isinstance(pointers, list)
            or len(pointers) != _accepted.EP_SIZE
            or any(
                type(pointer) is not int or pointer <= 0 or pointer % 16
                for pointer in pointers
            )
        ):
            raise ValueError(f"workspace.{name} must contain eight aligned addresses")
    if workspace.x_buffer_ptrs[workspace.ep_rank] != workspace.x_buffer.data_ptr():
        raise ValueError("workspace.x_buffer_ptrs does not own the local x buffer")
    if (
        workspace.combine_buffer_ptrs[workspace.ep_rank]
        != workspace.combine_buffer.data_ptr()
    ):
        raise ValueError("workspace.combine_buffer_ptrs does not own the local buffer")

    tensor_specs = (
        (
            "workspace.x_buffer",
            workspace.x_buffer,
            (workspace.num_local_tokens, _accepted.HIDDEN_SIZE),
            torch.bfloat16,
        ),
        (
            "workspace.combine_buffer",
            workspace.combine_buffer,
            (workspace.num_local_tokens * _accepted.TOPK, _accepted.HIDDEN_SIZE),
            torch.bfloat16,
        ),
        (
            "shared_gate_weights",
            weights[0],
            (_accepted.INTERMEDIATE_SIZE, _accepted.HIDDEN_SIZE),
            torch.bfloat16,
        ),
        (
            "routed_gate_weights",
            weights[1],
            (
                _accepted.NUM_LOCAL_EXPERTS,
                _accepted.INTERMEDIATE_SIZE,
                _accepted.HIDDEN_SIZE,
            ),
            torch.bfloat16,
        ),
        (
            "shared_up_weights",
            weights[2],
            (_accepted.INTERMEDIATE_SIZE, _accepted.HIDDEN_SIZE),
            torch.bfloat16,
        ),
        (
            "routed_up_weights",
            weights[3],
            (
                _accepted.NUM_LOCAL_EXPERTS,
                _accepted.INTERMEDIATE_SIZE,
                _accepted.HIDDEN_SIZE,
            ),
            torch.bfloat16,
        ),
        (
            "shared_down_weights",
            weights[4],
            (_accepted.HIDDEN_SIZE, _accepted.INTERMEDIATE_SIZE),
            torch.bfloat16,
        ),
        (
            "routed_down_weights",
            weights[5],
            (
                _accepted.NUM_LOCAL_EXPERTS,
                _accepted.HIDDEN_SIZE,
                _accepted.INTERMEDIATE_SIZE,
            ),
            torch.bfloat16,
        ),
        (
            "schedule.peer_rank",
            schedule.peer_rank,
            (workspace.schedule_capacity,),
            torch.int32,
        ),
        (
            "schedule.peer_token_idx",
            schedule.peer_token_idx,
            (workspace.schedule_capacity,),
            torch.int32,
        ),
        ("schedule.num_tokens", schedule.num_tokens, (1,), torch.int32),
        (
            "schedule.tokens_per_expert",
            schedule.tokens_per_expert,
            (_accepted.NUM_LOCAL_EXPERTS,),
            torch.int32,
        ),
    )
    for name, tensor, shape, dtype in tensor_specs:
        _accepted._validate_tensor(
            name,
            tensor,
            shape=shape,
            dtype=dtype,
            device=workspace.device,
        )

    expected_shapes = _output_shapes(
        workspace.num_local_tokens,
        plan.macrobatch_size,
    )
    if not isinstance(state, _accepted.PersistentBf16State):
        raise TypeError("state must be a PersistentBf16State")
    for index, (tensor, shape) in enumerate(zip(state.abi_outputs, expected_shapes)):
        _accepted._validate_tensor(
            f"state.output.{index}",
            tensor,
            shape=shape,
            dtype=torch.bfloat16,
            device=workspace.device,
        )
    lengths = _accepted.persistent_bf16_counter_lengths(
        workspace.num_local_tokens,
        workspace.schedule_capacity,
    )
    for index, (tensor, length) in enumerate(
        zip(
            state.counters,
            (
                lengths.gate_up_tile_ready,
                lengths.hidden_row_block_ready,
                lengths.x_routed_ready,
                lengths.y_routed_ready,
                lengths.y_routed_done,
            ),
        )
    ):
        _accepted._validate_tensor(
            f"state.counter.{index}",
            tensor,
            shape=(length,),
            dtype=torch.int32,
            device=workspace.device,
        )


def _executor_key(
    workspace,
    schedule,
    weights,
    state,
    plan,
    *,
    runtime_environment,
    device_index: int,
    context_key: int,
) -> tuple[object, ...]:
    tensors = (
        workspace.x_buffer,
        workspace.combine_buffer,
        schedule.peer_rank,
        schedule.peer_token_idx,
        schedule.num_tokens,
        schedule.tokens_per_expert,
        *weights,
        *state.abi_outputs,
        *state.counters,
    )
    return (
        "mok-private-persistent-bf16-macrobatch",
        _CACHE_EPOCH,
        ("runtime", runtime_environment),
        ("device", device_index, context_key, (10, 3)),
        (
            "geometry",
            workspace.num_local_tokens,
            workspace.schedule_capacity,
            plan.macrobatch_size,
            plan.minibatch_size,
            plan.num_comm_sms,
        ),
        (
            "abi",
            tuple(
                _accepted._tensor_abi_descriptor(str(index), tensor)
                for index, tensor in enumerate(tensors)
            ),
        ),
    )


def _cached_launch(key, device_index, positional_args, keyword_args):
    make_args, prepare, runtime_prefix = _accepted._load_mega_runtime()
    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            executor, runtime_args = prepare(*positional_args, **keyword_args)
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device_index)
            return executor, runtime_args
    cute_args = make_args(*positional_args, **keyword_args)
    return executor, cute_args[:runtime_prefix] + cute_args[-1:]


def clear_experimental_macrobatch_cache(*, synchronize: bool = True) -> None:
    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()


def forward_bf16(
    workspace,
    schedule,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    *,
    macrobatch_size: int,
    minibatch_size: int,
    swiglu_limit: float | None,
    num_comm_sms: int = NUM_COMM_SMS,
) -> tuple[object, ...]:
    """Run one private variable-B specialization of the accepted device body."""

    plan = ExperimentalMacrobatchPlan(
        macrobatch_size,
        minibatch_size,
        num_comm_sms,
    )
    plan.validate()
    if swiglu_limit is not None:
        raise NotImplementedError("experimental persistent BF16 supports unclamped SwiGLU only")

    import torch

    device_index = (
        workspace.device.index
        if workspace.device.index is not None
        else torch.cuda.current_device()
    )
    if torch.cuda.current_device() != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    runtime_environment = _accepted._prepare_public_runtime_environment()
    stream = torch.cuda.current_stream(workspace.device)
    state = _prepare_state(
        num_local_tokens=workspace.num_local_tokens,
        schedule_capacity=workspace.schedule_capacity,
        device=workspace.device,
        plan=plan,
    )
    weights = (
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
    )
    _validate_call(workspace, schedule, weights, state, plan)
    key = _executor_key(
        workspace,
        schedule,
        weights,
        state,
        plan,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_accepted._current_cuda_context_key(),
    )
    positional_args, keyword_args = _accepted._mega_call_arguments(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=0.0,
        is_clamped=False,
        stream=stream,
    )
    executor, runtime_args = _cached_launch(
        key,
        device_index,
        positional_args,
        keyword_args,
    )
    with torch.cuda.stream(stream):
        for counter in state.counters:
            counter.zero_()
        executor(*runtime_args)
    _accepted._record_public_launch_owners(
        stream,
        workspace,
        schedule,
        weights,
        state,
    )
    return state.abi_outputs


__all__ = (
    "ExperimentalMacrobatchPlan",
    "MACROBATCH_CANDIDATES",
    "clear_experimental_macrobatch_cache",
    "forward_bf16",
)
