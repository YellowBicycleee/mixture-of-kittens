"""Host gates for the CUDA-shaped CuTe DSL BF16 persistent forward.

The full entry point is the explicit ``fwd_backend="cutedsl"`` specialization:
one persistent launch with the same CUDA ABI, role split, task order, and
readiness arrays.  The smaller FC1 entry point remains only a compile/numerical
microscope for the cluster-2 Gate/Up MMA topology.

The fixed slice is one logical ``M256 x N128 x K4096`` Gate/Up tile.  CTA 0
loads its local ``M128`` A rows and the Gate ``N128`` weight tile; CTA 1 loads
its different local ``M128`` A rows and the Up ``N128`` tile.  The cooperative
MMA exposes ``[Gate128 | Up128]`` in each CTA's local TMEM accumulator.  The
current device body writes that raw packed accumulator for a later numerical
probe.  SwiGLU, Down, communication workers, readiness scheduling, and the
148-CTA persistent launch are N/A in this slice.

Imports of torch/CUTLASS remain lazy so CPU-only source tests can validate the
contract.  There is no fallback to the host-wavefront forward implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import os
import sys
import threading
from typing import Final


TARGET_ARCH: Final = "sm_103a"
FC1_TILE_M: Final = 256
FC1_LOGICAL_N: Final = 128
FC1_PACKED_N: Final = 2 * FC1_LOGICAL_N
FC1_TILE_K: Final = 4096
CTA_M: Final = FC1_TILE_M // 2
THREADS_PER_CTA: Final = 256
CLUSTER_SHAPE: Final = (2, 1, 1)

# The slice and full-forward flags describe the source entry points.  Runtime
# support remains guarded independently by the fixed-shape and dependency
# checks below.
FC1_SLICE_SOURCE_BODY_PRESENT: Final = True
FULL_PERSISTENT_FORWARD_COMPLETE: Final = True

# The production migration target is deliberately fixed.  Supporting another
# shape is a separate specialization, not a reason to weaken this gate.
EP_SIZE: Final = 8
NUM_LOCAL_EXPERTS: Final = 64
HIDDEN_SIZE: Final = 4096
INTERMEDIATE_SIZE: Final = 1024
TOPK: Final = 10
MACROBATCH_ROWS: Final = 32768
MINIBATCH_ROWS: Final = 4096
NUM_COMM_SMS: Final = 40
CLC_DEPTH: Final = 1
GATE_TASK_GROUP_SIZE: Final = 1
DOWN_TASK_GROUP_SIZE: Final = 1
ROW_ALIGNMENT: Final = 256
HIDDEN_ROW_BLOCK: Final = 256
Y_DONE_ROW_BLOCK: Final = 128

# Executor cache identity.  The cache contains compiled code only: runtime
# arguments, tensor owners, state buffers, and streams are rebuilt per call.
_CODEGEN_EPOCH: Final = "r54-ab6-acc2-prefixdecode-fixed6-dsl462-quack064"
_ABI_EPOCH: Final = "persistent-bf16-nine-output-five-counter-v1"
_REQUIRED_CUTLASS_DSL: Final = "4.6.2"
_REQUIRED_QUACK: Final = "0.6.4"
_TVM_FFI_ENV: Final = "CUTE_DSL_ENABLE_TVM_FFI"
_ARCH_ENV: Final = "CUTE_DSL_ARCH"
_COMPILE_OPTIONS: Final = ("cute.compile.default-options",)

_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PersistentBf16Plan:
    """Exact public configuration of the first single-launch migration."""

    arch: str = TARGET_ARCH
    ep_size: int = EP_SIZE
    num_local_experts: int = NUM_LOCAL_EXPERTS
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    topk: int = TOPK
    macrobatch_size: int = MACROBATCH_ROWS
    minibatch_size: int = MINIBATCH_ROWS
    num_comm_sms: int = NUM_COMM_SMS
    clc_depth: int = CLC_DEPTH
    gate_task_group_size: int = GATE_TASK_GROUP_SIZE
    down_task_group_size: int = DOWN_TASK_GROUP_SIZE

    def validate(self) -> None:
        wanted = (
            TARGET_ARCH,
            EP_SIZE,
            NUM_LOCAL_EXPERTS,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            TOPK,
            MACROBATCH_ROWS,
            MINIBATCH_ROWS,
            NUM_COMM_SMS,
            CLC_DEPTH,
            GATE_TASK_GROUP_SIZE,
            DOWN_TASK_GROUP_SIZE,
        )
        actual = (
            self.arch,
            self.ep_size,
            self.num_local_experts,
            self.hidden_size,
            self.intermediate_size,
            self.topk,
            self.macrobatch_size,
            self.minibatch_size,
            self.num_comm_sms,
            self.clc_depth,
            self.gate_task_group_size,
            self.down_task_group_size,
        )
        if actual != wanted:
            raise NotImplementedError(
                "the first persistent BF16 migration is fixed to "
                "SM103 EP8 E64 H4096 I1024 topk10 macro32768 mini4096 "
                "comm40 CLC1 G1D1"
            )


@dataclass(frozen=True)
class PersistentBf16CounterLengths:
    gate_up_tile_ready: int
    hidden_row_block_ready: int
    x_routed_ready: int
    y_routed_ready: int
    y_routed_done: int


@dataclass(frozen=True)
class PersistentBf16State:
    """The unchanged CUDA BF16 nine-tensor ABI and five counter arrays."""

    x_routed: object
    gate_shared: object
    gate_routed: object
    up_shared: object
    up_routed: object
    hidden_shared: object
    hidden_routed: object
    y_shared: object
    y_routed: object
    gate_up_tile_ready: object
    hidden_row_block_ready: object
    x_routed_ready: object
    y_routed_ready: object
    y_routed_done: object

    @property
    def abi_outputs(self) -> tuple[object, ...]:
        return (
            self.x_routed,
            self.gate_shared,
            self.gate_routed,
            self.up_shared,
            self.up_routed,
            self.hidden_shared,
            self.hidden_routed,
            self.y_shared,
            self.y_routed,
        )

    @property
    def counters(self) -> tuple[object, ...]:
        return (
            self.gate_up_tile_ready,
            self.hidden_row_block_ready,
            self.x_routed_ready,
            self.y_routed_ready,
            self.y_routed_done,
        )


@dataclass(frozen=True)
class CompiledPersistentBf16Forward:
    """One compiled executor bound to a stable workspace/state address set."""

    executor: object
    runtime_args: tuple[object, ...]
    state: PersistentBf16State
    stream: object

    def __call__(self) -> None:
        """Reset CUDA-equivalent counters and launch without recompiling."""

        import torch

        with torch.cuda.stream(self.stream):
            for counter in self.state.counters:
                counter.zero_()
            self.executor(*self.runtime_args)


def _normalize_swiglu_limit(swiglu_limit: float | None) -> tuple[float, bool]:
    """Mirror the CUDA BF16 wrapper's optional clamp contract."""

    if swiglu_limit is None:
        return 0.0, False
    if type(swiglu_limit) not in (int, float) or swiglu_limit < 0:
        raise ValueError("swiglu_limit must be None or a non-negative number")
    return float(swiglu_limit), True


def _prepare_public_runtime_environment() -> tuple[object, ...]:
    """Freeze the optional CuTe runtime before importing CUTLASS.

    CUDA-only users never call this function.  If CUTLASS was imported before
    TVM FFI was selected, fail closed instead of running under an ambiguous
    executor ABI.
    """

    ffi_value = os.environ.get(_TVM_FFI_ENV)
    cutlass_is_loaded = any(
        name == "cutlass" or name.startswith("cutlass.") for name in sys.modules
    )
    if ffi_value is None:
        if cutlass_is_loaded:
            raise RuntimeError(
                f"{_TVM_FFI_ENV}=1 must be selected before importing CUTLASS"
            )
        os.environ[_TVM_FFI_ENV] = "1"
    elif ffi_value != "1":
        raise RuntimeError(f"public CuTe DSL forward requires {_TVM_FFI_ENV}=1")

    arch_value = os.environ.get(_ARCH_ENV)
    if arch_value is None:
        if cutlass_is_loaded:
            raise RuntimeError(
                f"{_ARCH_ENV}={TARGET_ARCH} must be selected before importing CUTLASS"
            )
        os.environ[_ARCH_ENV] = TARGET_ARCH
    elif arch_value != TARGET_ARCH:
        raise RuntimeError(
            f"public CuTe DSL forward requires {_ARCH_ENV}={TARGET_ARCH}"
        )

    try:
        cutlass_version = metadata.version("nvidia-cutlass-dsl")
        quack_version = metadata.version("quack-kernels")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "public CuTe DSL forward requires the 'cutedsl' optional dependencies"
        ) from error
    if cutlass_version != _REQUIRED_CUTLASS_DSL:
        raise RuntimeError(
            "public CuTe DSL forward requires "
            f"nvidia-cutlass-dsl=={_REQUIRED_CUTLASS_DSL}; got {cutlass_version}"
        )
    if quack_version != _REQUIRED_QUACK:
        raise RuntimeError(
            "public CuTe DSL forward requires "
            f"quack-kernels=={_REQUIRED_QUACK}; got {quack_version}"
        )
    return (
        ("nvidia-cutlass-dsl", cutlass_version),
        ("quack-kernels", quack_version),
        (_TVM_FFI_ENV, "1"),
        (_ARCH_ENV, TARGET_ARCH),
        ("compile_options", _COMPILE_OPTIONS),
    )


def _current_cuda_context_key() -> int:
    """Return the current CUDA context handle for executor ownership."""

    from cuda.bindings import driver as cuda

    result, context = cuda.cuCtxGetCurrent()
    if result != cuda.CUresult.CUDA_SUCCESS or context is None:
        raise RuntimeError(f"cuCtxGetCurrent failed: {result}")
    try:
        return int(context)
    except TypeError:
        value = getattr(context, "value", None)
        if value is None:
            raise RuntimeError("CUDA context handle is not representable as an integer")
        return int(value)


def _tensor_abi_descriptor(name: str, tensor) -> tuple[object, ...]:
    """Describe compile-relevant tensor ABI without capturing an address."""

    return (
        name,
        tuple(int(extent) for extent in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        str(tensor.dtype),
    )


def _persistent_executor_key(
    workspace,
    schedule,
    weights: tuple[object, ...],
    state: PersistentBf16State,
    *,
    plan: PersistentBf16Plan,
    swiglu_limit_value: float,
    is_clamped: bool,
    runtime_environment: tuple[object, ...],
    device_index: int,
    context_key: int,
) -> tuple[object, ...]:
    """Build the complete static specialization key, never an address key."""

    named_tensors = (
        ("workspace.x_buffer", workspace.x_buffer),
        ("workspace.combine_buffer", workspace.combine_buffer),
        ("schedule.peer_rank", schedule.peer_rank),
        ("schedule.peer_token_idx", schedule.peer_token_idx),
        ("schedule.num_tokens", schedule.num_tokens),
        ("schedule.tokens_per_expert", schedule.tokens_per_expert),
        ("shared_gate_weights", weights[0]),
        ("routed_gate_weights", weights[1]),
        ("shared_up_weights", weights[2]),
        ("routed_up_weights", weights[3]),
        ("shared_down_weights", weights[4]),
        ("routed_down_weights", weights[5]),
        *tuple(
            (f"state.output.{index}", tensor)
            for index, tensor in enumerate(state.abi_outputs)
        ),
        *tuple(
            (f"state.counter.{index}", tensor)
            for index, tensor in enumerate(state.counters)
        ),
    )
    tensor_abi = tuple(
        _tensor_abi_descriptor(name, tensor) for name, tensor in named_tensors
    )
    return (
        "mok-persistent-bf16-executor",
        _CODEGEN_EPOCH,
        ("device", device_index),
        ("context", context_key),
        ("arch", TARGET_ARCH, (10, 3)),
        ("runtime", runtime_environment),
        ("num_local_tokens", workspace.num_local_tokens),
        ("schedule_capacity", workspace.schedule_capacity),
        ("macrobatch_size", plan.macrobatch_size),
        ("minibatch_size", plan.minibatch_size),
        ("num_comm_sms", plan.num_comm_sms),
        ("swiglu", float(swiglu_limit_value), bool(is_clamped)),
        ("peer_pointer_abi", EP_SIZE, "bf16-gmem-align16"),
        ("abi", _ABI_EPOCH, tensor_abi),
    )


def clear_persistent_bf16_executor_cache(*, synchronize: bool = True) -> None:
    """Release executor-only cache entries after public calls are quiescent."""

    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()


def persistent_bf16_output_shapes(num_local_tokens: int) -> tuple[tuple[int, ...], ...]:
    """Return the exact shapes allocated by CUDA's BF16 forward wrapper."""

    if (
        type(num_local_tokens) is not int
        or num_local_tokens < 2 * ROW_ALIGNMENT
        or num_local_tokens % ROW_ALIGNMENT
    ):
        raise ValueError("num_local_tokens must be an integer multiple of 256 at least 512")
    return (
        (MACROBATCH_ROWS, HIDDEN_SIZE),
        (num_local_tokens, INTERMEDIATE_SIZE),
        (MACROBATCH_ROWS, INTERMEDIATE_SIZE),
        (num_local_tokens, INTERMEDIATE_SIZE),
        (MACROBATCH_ROWS, INTERMEDIATE_SIZE),
        (num_local_tokens, INTERMEDIATE_SIZE),
        (MACROBATCH_ROWS, INTERMEDIATE_SIZE),
        (num_local_tokens, HIDDEN_SIZE),
        (MACROBATCH_ROWS, HIDDEN_SIZE),
    )


def persistent_bf16_counter_lengths(
    num_local_tokens: int,
    schedule_capacity: int,
) -> PersistentBf16CounterLengths:
    """Mirror the five CUDA allocations in ``forward.cuh`` exactly."""

    persistent_bf16_output_shapes(num_local_tokens)
    if (
        type(schedule_capacity) is not int
        or schedule_capacity < ROW_ALIGNMENT
        or schedule_capacity % ROW_ALIGNMENT
    ):
        raise ValueError("schedule_capacity must be a positive multiple of 256")
    shared_row_blocks = num_local_tokens // HIDDEN_ROW_BLOCK
    routed_row_blocks = schedule_capacity // HIDDEN_ROW_BLOCK
    gate_up_column_blocks = INTERMEDIATE_SIZE // FC1_LOGICAL_N
    return PersistentBf16CounterLengths(
        gate_up_tile_ready=(
            shared_row_blocks + routed_row_blocks
        ) * gate_up_column_blocks,
        hidden_row_block_ready=shared_row_blocks + routed_row_blocks,
        x_routed_ready=(schedule_capacity + MINIBATCH_ROWS - 1) // MINIBATCH_ROWS,
        y_routed_ready=(schedule_capacity + MINIBATCH_ROWS - 1) // MINIBATCH_ROWS,
        y_routed_done=schedule_capacity // Y_DONE_ROW_BLOCK,
    )


def prepare_persistent_bf16_state(
    *,
    num_local_tokens: int,
    schedule_capacity: int,
    device,
) -> PersistentBf16State:
    """Allocate the unchanged CUDA ABI without launching any backend."""

    import torch

    shapes = persistent_bf16_output_shapes(num_local_tokens)
    outputs = tuple(
        torch.empty(shape, dtype=torch.bfloat16, device=device)
        for shape in shapes
    )
    lengths = persistent_bf16_counter_lengths(
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
    return PersistentBf16State(*outputs, *counters)


def _validate_tensor(name, tensor, *, shape, dtype, device) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        not tensor.is_cuda
        or tensor.device != device
        or tensor.dtype != dtype
        or not tensor.is_contiguous()
        or tuple(tensor.shape) != shape
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} {shape} on {device}"
        )
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")


def validate_persistent_bf16_call(
    workspace,
    schedule,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    state: PersistentBf16State,
    *,
    plan: PersistentBf16Plan = PersistentBf16Plan(),
) -> None:
    """Fail closed before importing CUTLASS or compiling the mega-kernel."""

    import torch

    plan.validate()
    required_workspace = (
        "device",
        "ep_size",
        "num_local_tokens",
        "hidden_size",
        "topk",
        "schedule_capacity",
        "x_buffer",
        "x_buffer_ptrs",
        "combine_buffer_ptrs",
        "combine_buffer",
        "ep_rank",
    )
    missing = tuple(name for name in required_workspace if not hasattr(workspace, name))
    if missing:
        raise TypeError(f"workspace is missing fields: {missing}")
    device = workspace.device
    if (
        workspace.ep_size != EP_SIZE
        or workspace.hidden_size != HIDDEN_SIZE
        or workspace.topk != TOPK
    ):
        raise NotImplementedError("persistent BF16 mega V0 requires EP8 H4096 topk10")
    persistent_bf16_output_shapes(workspace.num_local_tokens)
    lengths = persistent_bf16_counter_lengths(
        workspace.num_local_tokens,
        workspace.schedule_capacity,
    )
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise NotImplementedError("persistent BF16 mega V0 requires B300/SM103")
    for name in ("x_buffer_ptrs", "combine_buffer_ptrs"):
        pointers = getattr(workspace, name)
        if (
            not isinstance(pointers, list)
            or len(pointers) != EP_SIZE
            or any(type(pointer) is not int or pointer <= 0 or pointer % 16 for pointer in pointers)
        ):
            raise ValueError(f"workspace.{name} must contain eight aligned addresses")

    _validate_tensor(
        "workspace.x_buffer",
        workspace.x_buffer,
        shape=(workspace.num_local_tokens, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        "workspace.combine_buffer",
        workspace.combine_buffer,
        shape=(workspace.num_local_tokens * TOPK, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    if not 0 <= workspace.ep_rank < EP_SIZE:
        raise ValueError("workspace.ep_rank is outside EP8")
    if workspace.x_buffer_ptrs[workspace.ep_rank] != workspace.x_buffer.data_ptr():
        raise ValueError("workspace.x_buffer_ptrs does not own the local x buffer")
    if (
        workspace.combine_buffer_ptrs[workspace.ep_rank]
        != workspace.combine_buffer.data_ptr()
    ):
        raise ValueError(
            "workspace.combine_buffer_ptrs does not own the local combine buffer"
        )
    weight_specs = (
        ("shared_gate_weights", shared_gate_weights, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        (
            "routed_gate_weights",
            routed_gate_weights,
            (NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ),
        ("shared_up_weights", shared_up_weights, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        (
            "routed_up_weights",
            routed_up_weights,
            (NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ),
        ("shared_down_weights", shared_down_weights, (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
        (
            "routed_down_weights",
            routed_down_weights,
            (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        ),
    )
    for name, tensor, shape in weight_specs:
        _validate_tensor(
            name,
            tensor,
            shape=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    schedule_specs = (
        ("schedule.peer_rank", schedule.peer_rank, (workspace.schedule_capacity,)),
        (
            "schedule.peer_token_idx",
            schedule.peer_token_idx,
            (workspace.schedule_capacity,),
        ),
        ("schedule.num_tokens", schedule.num_tokens, (1,)),
        (
            "schedule.tokens_per_expert",
            schedule.tokens_per_expert,
            (NUM_LOCAL_EXPERTS,),
        ),
    )
    for name, tensor, shape in schedule_specs:
        _validate_tensor(
            name,
            tensor,
            shape=shape,
            dtype=torch.int32,
            device=device,
        )

    if not isinstance(state, PersistentBf16State):
        raise TypeError("state must be a PersistentBf16State")
    for name, tensor, shape in zip(
        (
            "x_routed",
            "gate_shared",
            "gate_routed",
            "up_shared",
            "up_routed",
            "hidden_shared",
            "hidden_routed",
            "y_shared",
            "y_routed",
        ),
        state.abi_outputs,
        persistent_bf16_output_shapes(workspace.num_local_tokens),
    ):
        _validate_tensor(
            f"state.{name}",
            tensor,
            shape=shape,
            dtype=torch.bfloat16,
            device=device,
        )
    for name, tensor, length in zip(
        (
            "gate_up_tile_ready",
            "hidden_row_block_ready",
            "x_routed_ready",
            "y_routed_ready",
            "y_routed_done",
        ),
        state.counters,
        (
            lengths.gate_up_tile_ready,
            lengths.hidden_row_block_ready,
            lengths.x_routed_ready,
            lengths.y_routed_ready,
            lengths.y_routed_done,
        ),
    ):
        _validate_tensor(
            f"state.{name}",
            tensor,
            shape=(length,),
            dtype=torch.int32,
            device=device,
        )


def _mega_call_arguments(
    workspace,
    schedule,
    weights: tuple[object, ...],
    state: PersistentBf16State,
    *,
    plan: PersistentBf16Plan,
    swiglu_limit_value: float,
    is_clamped: bool,
    stream,
) -> tuple[tuple[object, ...], dict[str, object]]:
    return (
        (
            workspace.x_buffer_ptrs,
            workspace.combine_buffer_ptrs,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.x_buffer,
            *weights,
            state,
        ),
        {
            "num_local_tokens": workspace.num_local_tokens,
            "schedule_capacity": workspace.schedule_capacity,
            "macrobatch_size": plan.macrobatch_size,
            "minibatch_size": plan.minibatch_size,
            "num_comm_sms": plan.num_comm_sms,
            "swiglu_limit": swiglu_limit_value,
            "is_clamped": is_clamped,
            "stream": stream,
        },
    )


def _prepare_cached_mega_launch(
    key: tuple[object, ...],
    device_index: int,
    positional_args: tuple[object, ...],
    keyword_args: dict[str, object],
) -> tuple[object, tuple[object, ...]]:
    """Get code by static signature and bind only this call's addresses."""


    make_mega_args, prepare_mega_bf16, runtime_prefix_args = _load_mega_runtime()

    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            executor, runtime_args = prepare_mega_bf16(
                *positional_args,
                **keyword_args,
            )
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device_index)
            return executor, runtime_args

    cute_args = make_mega_args(*positional_args, **keyword_args)
    runtime_args = cute_args[:runtime_prefix_args] + cute_args[-1:]
    return executor, runtime_args


def _load_mega_runtime():
    """Import CUTLASS only after the public environment gate succeeds."""

    from ._persistent_bf16_mega import (
        _MEGA_RUNTIME_PREFIX_ARGS,
        make_mega_args,
        prepare_mega_bf16,
    )
    return make_mega_args, prepare_mega_bf16, _MEGA_RUNTIME_PREFIX_ARGS


def _record_public_launch_owners(
    stream,
    workspace,
    schedule,
    weights: tuple[object, ...],
    state: PersistentBf16State,
) -> None:
    """Keep every caching-allocator storage alive through this launch."""

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
    seen: set[int] = set()
    for tensor in tensors:
        identity = id(tensor)
        if identity not in seen:
            tensor.record_stream(stream)
            seen.add(identity)


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
    """Run the public fixed-shape persistent CuTe DSL BF16 forward.

    Only the executor is cached.  State, current stream, pointers, and all
    runtime argument wrappers are fresh for every invocation.
    """

    plan = PersistentBf16Plan(
        ep_size=getattr(workspace, "ep_size", -1),
        num_local_experts=getattr(routed_gate_weights, "shape", (-1,))[0],
        hidden_size=getattr(workspace, "hidden_size", -1),
        intermediate_size=getattr(shared_gate_weights, "shape", (-1,))[0],
        topk=getattr(workspace, "topk", -1),
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
    )
    plan.validate()
    if swiglu_limit is not None:
        raise NotImplementedError(
            "public CuTe DSL forward currently supports unclamped SwiGLU only"
        )

    import torch

    device_index = (
        workspace.device.index
        if workspace.device.index is not None
        else torch.cuda.current_device()
    )
    if torch.cuda.current_device() != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    runtime_environment = _prepare_public_runtime_environment()
    stream = torch.cuda.current_stream(workspace.device)
    state = prepare_persistent_bf16_state(
        num_local_tokens=workspace.num_local_tokens,
        schedule_capacity=workspace.schedule_capacity,
        device=workspace.device,
    )
    weights = (
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
    )
    validate_persistent_bf16_call(
        workspace,
        schedule,
        *weights,
        state,
        plan=plan,
    )
    swiglu_limit_value, is_clamped = _normalize_swiglu_limit(swiglu_limit)
    key = _persistent_executor_key(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
    )
    positional_args, keyword_args = _mega_call_arguments(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        stream=stream,
    )
    executor, runtime_args = _prepare_cached_mega_launch(
        key,
        device_index,
        positional_args,
        keyword_args,
    )
    with torch.cuda.stream(stream):
        for counter in state.counters:
            counter.zero_()
        executor(*runtime_args)
    _record_public_launch_owners(stream, workspace, schedule, weights, state)
    return state.abi_outputs


def compile_persistent_forward_bf16(
    workspace,
    schedule,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    state: PersistentBf16State,
    *,
    swiglu_limit: float | None = None,
    stream=None,
    plan: PersistentBf16Plan = PersistentBf16Plan(),
):
    """Compile once and bind one private persistent mega launch."""

    validate_persistent_bf16_call(
        workspace,
        schedule,
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
        state,
        plan=plan,
    )
    swiglu_limit_value, is_clamped = _normalize_swiglu_limit(swiglu_limit)
    import torch

    if stream is None:
        stream = torch.cuda.current_stream(workspace.device)
    from ._persistent_bf16_mega import prepare_mega_bf16

    executor, runtime_args = prepare_mega_bf16(
        workspace.x_buffer_ptrs,
        workspace.combine_buffer_ptrs,
        schedule.peer_rank,
        schedule.peer_token_idx,
        schedule.num_tokens,
        schedule.tokens_per_expert,
        workspace.x_buffer,
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
        state,
        num_local_tokens=workspace.num_local_tokens,
        schedule_capacity=workspace.schedule_capacity,
        macrobatch_size=plan.macrobatch_size,
        minibatch_size=plan.minibatch_size,
        num_comm_sms=plan.num_comm_sms,
        swiglu_limit=swiglu_limit_value,
        is_clamped=is_clamped,
        stream=stream,
    )
    return CompiledPersistentBf16Forward(
        executor=executor,
        runtime_args=runtime_args,
        state=state,
        stream=stream,
    )


def run_persistent_forward_bf16(
    workspace,
    schedule,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    state: PersistentBf16State,
    *,
    swiglu_limit: float | None = None,
    stream=None,
    plan: PersistentBf16Plan = PersistentBf16Plan(),
) -> None:
    """Compile and execute exactly one private persistent mega launch."""

    compiled = compile_persistent_forward_bf16(
        workspace,
        schedule,
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
        state,
        swiglu_limit=swiglu_limit,
        stream=stream,
        plan=plan,
    )
    compiled()


@dataclass(frozen=True)
class Fc1SlicePlan:
    """The only specialization accepted by the private compile probe."""

    arch: str = TARGET_ARCH
    m: int = FC1_TILE_M
    n: int = FC1_LOGICAL_N
    k: int = FC1_TILE_K
    cluster_shape: tuple[int, int, int] = CLUSTER_SHAPE
    threads_per_cta: int = THREADS_PER_CTA

    def validate(self) -> None:
        wanted = (
            TARGET_ARCH,
            FC1_TILE_M,
            FC1_LOGICAL_N,
            FC1_TILE_K,
            CLUSTER_SHAPE,
            THREADS_PER_CTA,
        )
        actual = (
            self.arch,
            self.m,
            self.n,
            self.k,
            self.cluster_shape,
            self.threads_per_cta,
        )
        if actual != wanted:
            raise NotImplementedError(
                "the BF16 FC1 slice is fixed to "
                "SM103 M256xN128xK4096, cluster=(2,1,1), block=256"
            )


def validate_fc1_slice_tensors(x, gate, up, packed_output) -> None:
    """Validate four torch tensors without importing torch at module import."""

    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    device = x.device
    tensors = {
        "x": (x, (FC1_TILE_M, FC1_TILE_K)),
        "gate": (gate, (FC1_LOGICAL_N, FC1_TILE_K)),
        "up": (up, (FC1_LOGICAL_N, FC1_TILE_K)),
        "packed_output": (packed_output, (FC1_TILE_M, FC1_PACKED_N)),
    }
    for name, (tensor, shape) in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} data pointer must be 16-byte aligned")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}")


def _validate_fc1_slice_call(
    x,
    gate,
    up,
    packed_output,
    plan: Fc1SlicePlan,
) -> None:
    """Apply the shared fail-closed host gate before importing CuTe DSL."""

    import torch

    plan.validate()
    validate_fc1_slice_tensors(x, gate, up, packed_output)
    if torch.cuda.get_device_capability(x.device) != (10, 3):
        raise NotImplementedError("the BF16 FC1 slice requires B300/SM103")


def compile_fc1_slice(
    x,
    gate,
    up,
    packed_output,
    *,
    stream=None,
    plan: Fc1SlicePlan = Fc1SlicePlan(),
):
    """Compile the private raw Gate/Up slice; do not run a full Forward.

    The returned executor is intentionally separate from the public backend.
    Calling this function on a non-SM103 device or without exact CuTe DSL
    dependencies is an explicit error, never a fallback.
    """

    _validate_fc1_slice_call(x, gate, up, packed_output, plan)

    from ._persistent_bf16_gemm import compile_fc1_slice as _compile

    return _compile(x, gate, up, packed_output, stream=stream)


def run_fc1_slice(
    x,
    gate,
    up,
    packed_output,
    *,
    stream=None,
    plan: Fc1SlicePlan = Fc1SlicePlan(),
) -> None:
    """Compile and launch the private raw Gate/Up slice exactly once.

    The caller owns ``packed_output`` and stream synchronization.  This probe
    is not connected to the public Forward backend and has no fallback.
    """

    _validate_fc1_slice_call(x, gate, up, packed_output, plan)

    from ._persistent_bf16_gemm import run_fc1_slice as _run

    _run(x, gate, up, packed_output, stream=stream)


__all__ = [
    "CLC_DEPTH",
    "CLUSTER_SHAPE",
    "DOWN_TASK_GROUP_SIZE",
    "EP_SIZE",
    "FC1_LOGICAL_N",
    "FC1_PACKED_N",
    "FC1_SLICE_SOURCE_BODY_PRESENT",
    "FC1_TILE_K",
    "FC1_TILE_M",
    "FULL_PERSISTENT_FORWARD_COMPLETE",
    "GATE_TASK_GROUP_SIZE",
    "Fc1SlicePlan",
    "CompiledPersistentBf16Forward",
    "PersistentBf16CounterLengths",
    "PersistentBf16Plan",
    "PersistentBf16State",
    "clear_persistent_bf16_executor_cache",
    "compile_fc1_slice",
    "compile_persistent_forward_bf16",
    "forward_bf16",
    "persistent_bf16_counter_lengths",
    "persistent_bf16_output_shapes",
    "prepare_persistent_bf16_state",
    "run_fc1_slice",
    "run_persistent_forward_bf16",
    "validate_persistent_bf16_call",
    "validate_fc1_slice_tensors",
]
