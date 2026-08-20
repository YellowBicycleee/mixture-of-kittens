"""Correctness-first CuTe DSL forward for the fixed Qwen BF16/EP8 target.

This is a real sister backend: dispatch, both expert MLPs, SwiGLU, and combine
are CuTe DSL device kernels.  The GEMMs intentionally use a thin scalar
implementation for the first end-to-end correctness gate.  This version has
no valid performance claim (``N/A``): it synchronously reads ``num_tokens``
once and does not yet use Blackwell grouped Tensor Core GEMMs.  The existing
CUDA C++ megakernel remains MoK's default backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import from_dlpack, make_ptr

from .forward_contract import (
    EP_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_LOCAL_EXPERTS,
    TOPK,
    macro_offsets,
    validate_fixed_forward_contract,
)


THREADS = 256


class _DispatchKernel:
    """Pull routed BF16 rows from the eight symmetric x buffers."""

    @cute.jit
    def __call__(
        self,
        peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        x_routed: cute.Tensor,
        num_peer_tokens: cutlass.Constexpr,
        macro_offset: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        peer_layout = cute.make_layout((num_peer_tokens * HIDDEN_SIZE,))
        peers = []
        for peer in cutlass.range_constexpr(EP_SIZE):
            peers.append(cute.make_tensor(peer_ptrs[peer], peer_layout))
        total_elements = cute.size(x_routed)
        self.kernel(
            peers,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            x_routed,
            macro_offset,
            total_elements,
        ).launch(
            grid=((total_elements + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _load_peer(
        self,
        peers: list[cute.Tensor],
        peer_rank: Int32,
        flat_index: Int32,
    ):
        value = BFloat16(0.0)
        if peer_rank == Int32(0):
            value = peers[0][flat_index]
        elif peer_rank == Int32(1):
            value = peers[1][flat_index]
        elif peer_rank == Int32(2):
            value = peers[2][flat_index]
        elif peer_rank == Int32(3):
            value = peers[3][flat_index]
        elif peer_rank == Int32(4):
            value = peers[4][flat_index]
        elif peer_rank == Int32(5):
            value = peers[5][flat_index]
        elif peer_rank == Int32(6):
            value = peers[6][flat_index]
        else:
            value = peers[7][flat_index]
        return value

    @cute.kernel
    def kernel(
        self,
        peers: list[cute.Tensor],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        x_routed: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        total_elements: cutlass.Constexpr,
    ):
        linear = cute.arch.block_idx()[0] * Int32(THREADS) + cute.arch.thread_idx()[0]
        if linear < Int32(total_elements):
            local_row = linear // Int32(HIDDEN_SIZE)
            column = linear - local_row * Int32(HIDDEN_SIZE)
            global_row = local_row + Int32(macro_offset)
            if global_row < num_tokens[0]:
                peer_rank = schedule_peer_rank[global_row]
                if peer_rank >= Int32(0):
                    # scheduler.cuh stores source_token * topk + k.  Dispatch
                    # reads the source token row; combine below retains k.
                    route_idx = schedule_peer_token_idx[global_row]
                    source_token = route_idx // Int32(TOPK)
                    flat_index = source_token * Int32(HIDDEN_SIZE) + column
                    x_routed[local_row, column] = self._load_peer(
                        peers, peer_rank, flat_index
                    )
                else:
                    # The 256-aligned expert padding has peer_rank == -1 and
                    # an undefined token index.  Never read the latter.
                    x_routed[local_row, column] = BFloat16(0.0)


class _SharedGemmKernel:
    """Straightforward BF16 x BF16 -> BF16 CuTe DSL GEMM."""

    def __init__(self, reduction_size: int):
        self.reduction_size = reduction_size

    @cute.jit
    def __call__(
        self,
        activations: cute.Tensor,
        weights: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        total_elements = cute.size(output)
        self.kernel(activations, weights, output, total_elements).launch(
            grid=((total_elements + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        activations: cute.Tensor,
        weights: cute.Tensor,
        output: cute.Tensor,
        total_elements: cutlass.Constexpr,
    ):
        linear = cute.arch.block_idx()[0] * Int32(THREADS) + cute.arch.thread_idx()[0]
        if linear < Int32(total_elements):
            output_columns = output.shape[1]
            row = linear // output_columns
            column = linear - row * output_columns
            accumulator = Float32(0.0)
            for reduction_idx in cutlass.range(
                0, self.reduction_size, 1, unroll=1
            ):
                lhs = Float32(activations[row, reduction_idx])
                rhs = Float32(weights[column, reduction_idx])
                accumulator = accumulator + lhs * rhs
            output[row, column] = accumulator.to(BFloat16)


class _RoutedGemmKernel:
    """Grouped GEMM over the scheduler's contiguous expert segments."""

    def __init__(self, reduction_size: int):
        self.reduction_size = reduction_size

    @cute.jit
    def __call__(
        self,
        activations: cute.Tensor,
        weights: cute.Tensor,
        output: cute.Tensor,
        num_tokens: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        total_elements = cute.size(output)
        self.kernel(
            activations,
            weights,
            output,
            num_tokens,
            tokens_per_expert,
            macro_offset,
            total_elements,
        ).launch(
            grid=((total_elements + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _find_expert(self, global_row: Int32, tokens_per_expert: cute.Tensor):
        expert = Int32(0)
        expert_end = tokens_per_expert[0]
        while global_row >= expert_end and expert < Int32(NUM_LOCAL_EXPERTS - 1):
            expert = expert + Int32(1)
            expert_end = expert_end + tokens_per_expert[expert]
        return expert

    @cute.kernel
    def kernel(
        self,
        activations: cute.Tensor,
        weights: cute.Tensor,
        output: cute.Tensor,
        num_tokens: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        total_elements: cutlass.Constexpr,
    ):
        linear = cute.arch.block_idx()[0] * Int32(THREADS) + cute.arch.thread_idx()[0]
        if linear < Int32(total_elements):
            output_columns = output.shape[1]
            local_row = linear // output_columns
            column = linear - local_row * output_columns
            global_row = local_row + Int32(macro_offset)
            if global_row < num_tokens[0]:
                expert = self._find_expert(global_row, tokens_per_expert)
                accumulator = Float32(0.0)
                for reduction_idx in cutlass.range(
                    0, self.reduction_size, 1, unroll=1
                ):
                    lhs = Float32(activations[local_row, reduction_idx])
                    rhs = Float32(weights[expert, column, reduction_idx])
                    accumulator = accumulator + lhs * rhs
                output[local_row, column] = accumulator.to(BFloat16)


class _SwiGLUKernel:
    @cute.jit
    def __call__(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        hidden: cute.Tensor,
        num_tokens: Optional[cute.Tensor],
        macro_offset: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        total_elements = cute.size(hidden)
        self.kernel(
            gate, up, hidden, num_tokens, macro_offset, total_elements
        ).launch(
            grid=((total_elements + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        hidden: cute.Tensor,
        num_tokens: Optional[cute.Tensor],
        macro_offset: cutlass.Constexpr,
        total_elements: cutlass.Constexpr,
    ):
        linear = cute.arch.block_idx()[0] * Int32(THREADS) + cute.arch.thread_idx()[0]
        if linear < Int32(total_elements):
            row = linear // Int32(INTERMEDIATE_SIZE)
            column = linear - row * Int32(INTERMEDIATE_SIZE)
            if cutlass.const_expr(num_tokens is None):
                gate_value = Float32(gate[row, column])
                up_value = Float32(up[row, column])
                sigmoid = cute.arch.rcp_approx(
                    Float32(1.0) + cute.math.exp2(
                        -gate_value * Float32(1.4426950408889634),
                        fastmath=True,
                    )
                )
                hidden[row, column] = (
                    gate_value * sigmoid * up_value
                ).to(BFloat16)
            else:
                if row + Int32(macro_offset) < num_tokens[0]:
                    gate_value = Float32(gate[row, column])
                    up_value = Float32(up[row, column])
                    sigmoid = cute.arch.rcp_approx(
                        Float32(1.0) + cute.math.exp2(
                            -gate_value * Float32(1.4426950408889634),
                            fastmath=True,
                        )
                    )
                    hidden[row, column] = (
                        gate_value * sigmoid * up_value
                    ).to(BFloat16)


class _CombineKernel:
    """Push routed outputs to the original rank's full route-index row."""

    @cute.jit
    def __call__(
        self,
        combine_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        y_routed: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        total_elements = cute.size(y_routed)
        self.kernel(
            combine_ptrs,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            y_routed,
            macro_offset,
            total_elements,
        ).launch(
            grid=((total_elements + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _store_peer(
        self,
        peers: list[cute.Pointer],
        peer_rank: Int32,
        flat_index: Int64,
        value: BFloat16,
    ):
        if peer_rank == Int32(0):
            cute.arch.store(peers[0] + flat_index, value, scope="sys")
        elif peer_rank == Int32(1):
            cute.arch.store(peers[1] + flat_index, value, scope="sys")
        elif peer_rank == Int32(2):
            cute.arch.store(peers[2] + flat_index, value, scope="sys")
        elif peer_rank == Int32(3):
            cute.arch.store(peers[3] + flat_index, value, scope="sys")
        elif peer_rank == Int32(4):
            cute.arch.store(peers[4] + flat_index, value, scope="sys")
        elif peer_rank == Int32(5):
            cute.arch.store(peers[5] + flat_index, value, scope="sys")
        elif peer_rank == Int32(6):
            cute.arch.store(peers[6] + flat_index, value, scope="sys")
        else:
            cute.arch.store(peers[7] + flat_index, value, scope="sys")

    @cute.kernel
    def kernel(
        self,
        peers: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        y_routed: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        total_elements: cutlass.Constexpr,
    ):
        linear = cute.arch.block_idx()[0] * Int32(THREADS) + cute.arch.thread_idx()[0]
        if linear < Int32(total_elements):
            local_row = linear // Int32(HIDDEN_SIZE)
            column = linear - local_row * Int32(HIDDEN_SIZE)
            global_row = local_row + Int32(macro_offset)
            if global_row < num_tokens[0]:
                peer_rank = schedule_peer_rank[global_row]
                if peer_rank >= Int32(0):
                    # Unlike dispatch, combine addresses the complete route row
                    # (source_token * topk + k), preserving every k slot.
                    route_idx = schedule_peer_token_idx[global_row]
                    flat_index = (
                        Int64(route_idx) * Int64(HIDDEN_SIZE) + Int64(column)
                    )
                    self._store_peer(
                        peers,
                        peer_rank,
                        flat_index,
                        y_routed[local_row, column],
                    )


_DISPATCH = _DispatchKernel()
_SHARED_GATE_UP = _SharedGemmKernel(HIDDEN_SIZE)
_SHARED_DOWN = _SharedGemmKernel(INTERMEDIATE_SIZE)
_ROUTED_GATE_UP = _RoutedGemmKernel(HIDDEN_SIZE)
_ROUTED_DOWN = _RoutedGemmKernel(INTERMEDIATE_SIZE)
_SWIGLU = _SwiGLUKernel()
_COMBINE = _CombineKernel()


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor.device != device or not tensor.is_cuda:
        raise ValueError(f"{name} must be on {device}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}")


def _bf16_ptrs(pointers: Sequence[int]) -> list[cute.Pointer]:
    return [
        make_ptr(BFloat16, int(pointer), cute.AddressSpace.gmem, assumed_align=16)
        for pointer in pointers
    ]


def forward_bf16(
    workspace,
    schedule,
    shared_gate_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    routed_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_down_weights: torch.Tensor,
    *,
    macrobatch_size: int,
    minibatch_size: int,
    swiglu_limit: float | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the fixed-shape CuTe DSL backend and return MoK's nine-tensor ABI."""

    if swiglu_limit is not None:
        raise NotImplementedError("CuTe DSL forward currently supports unclamped SwiGLU only")
    if torch.cuda.get_device_capability(workspace.device) != (10, 3):
        raise NotImplementedError("CuTe DSL forward currently requires B300/SM103")

    validate_fixed_forward_contract(
        ep_size=workspace.ep_size,
        hidden_size=workspace.hidden_size,
        intermediate_size=shared_gate_weights.shape[0],
        num_local_experts=routed_gate_weights.shape[0],
        topk=workspace.topk,
        num_local_tokens=workspace.num_local_tokens,
        schedule_capacity=workspace.schedule_capacity,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        x_ptrs=workspace.x_buffer_ptrs,
        combine_ptrs=workspace.combine_buffer_ptrs,
    )

    device = workspace.device
    t = workspace.num_local_tokens
    m = macrobatch_size
    _check_tensor("workspace.x_buffer", workspace.x_buffer, (t, HIDDEN_SIZE), device)
    weight_shapes = (
        ("shared_gate_weights", shared_gate_weights, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        ("routed_gate_weights", routed_gate_weights, (NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        ("shared_up_weights", shared_up_weights, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        ("routed_up_weights", routed_up_weights, (NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        ("shared_down_weights", shared_down_weights, (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
        ("routed_down_weights", routed_down_weights, (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)),
    )
    for name, tensor, shape in weight_shapes:
        _check_tensor(name, tensor, shape, device)

    schedule_tensors = (
        ("schedule.peer_rank", schedule.peer_rank, (workspace.schedule_capacity,)),
        ("schedule.peer_token_idx", schedule.peer_token_idx, (workspace.schedule_capacity,)),
        ("schedule.num_tokens", schedule.num_tokens, (1,)),
        ("schedule.tokens_per_expert", schedule.tokens_per_expert, (NUM_LOCAL_EXPERTS,)),
    )
    for name, tensor, shape in schedule_tensors:
        if tensor.device != device or tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32 on {device}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")

    # These are the exact shapes returned by the CUDA BF16 custom op.  Routed
    # buffers are ring storage; reverse macro execution intentionally leaves
    # macro 0 resident for the existing CUDA backward path.
    x_routed = torch.empty((m, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    gate_shared = torch.empty((t, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    gate_routed = torch.empty((m, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    up_shared = torch.empty((t, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    up_routed = torch.empty((m, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    hidden_shared = torch.empty((t, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    hidden_routed = torch.empty((m, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    y_shared = torch.empty((t, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    y_routed = torch.empty((m, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    cute_x = from_dlpack(workspace.x_buffer, assumed_align=16)
    cute_schedule_rank = from_dlpack(schedule.peer_rank, assumed_align=16)
    cute_schedule_route = from_dlpack(schedule.peer_token_idx, assumed_align=16)
    cute_num_tokens = from_dlpack(schedule.num_tokens, assumed_align=16)
    cute_tokens_per_expert = from_dlpack(schedule.tokens_per_expert, assumed_align=16)
    cute_x_routed = from_dlpack(x_routed, assumed_align=16)
    cute_gate_shared = from_dlpack(gate_shared, assumed_align=16)
    cute_gate_routed = from_dlpack(gate_routed, assumed_align=16)
    cute_up_shared = from_dlpack(up_shared, assumed_align=16)
    cute_up_routed = from_dlpack(up_routed, assumed_align=16)
    cute_hidden_shared = from_dlpack(hidden_shared, assumed_align=16)
    cute_hidden_routed = from_dlpack(hidden_routed, assumed_align=16)
    cute_y_shared = from_dlpack(y_shared, assumed_align=16)
    cute_y_routed = from_dlpack(y_routed, assumed_align=16)

    cute_shared_gate_weights = from_dlpack(shared_gate_weights, assumed_align=16)
    cute_routed_gate_weights = from_dlpack(routed_gate_weights, assumed_align=16)
    cute_shared_up_weights = from_dlpack(shared_up_weights, assumed_align=16)
    cute_routed_up_weights = from_dlpack(routed_up_weights, assumed_align=16)
    cute_shared_down_weights = from_dlpack(shared_down_weights, assumed_align=16)
    cute_routed_down_weights = from_dlpack(routed_down_weights, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    # Shared expert is independent of the routed schedule.
    _SHARED_GATE_UP(cute_x, cute_shared_gate_weights, cute_gate_shared, stream)
    _SHARED_GATE_UP(cute_x, cute_shared_up_weights, cute_up_shared, stream)
    _SWIGLU(cute_gate_shared, cute_up_shared, cute_hidden_shared, None, 0, stream)
    _SHARED_DOWN(cute_hidden_shared, cute_shared_down_weights, cute_y_shared, stream)

    x_ptrs = _bf16_ptrs(workspace.x_buffer_ptrs)
    combine_ptrs = _bf16_ptrs(workspace.combine_buffer_ptrs)
    # Correctness-first bring-up reads the device scalar once so we launch only
    # real macros instead of the larger capacity envelope.  This D2H sync is
    # why timing for this revision is explicitly N/A.
    routed_num_tokens = int(schedule.num_tokens.item())
    if not 0 <= routed_num_tokens <= workspace.schedule_capacity:
        raise RuntimeError("schedule.num_tokens exceeds the schedule capacity")
    for macro_offset in macro_offsets(routed_num_tokens, m):
        _DISPATCH(
            x_ptrs,
            cute_schedule_rank,
            cute_schedule_route,
            cute_num_tokens,
            cute_x_routed,
            t,
            macro_offset,
            stream,
        )
        _ROUTED_GATE_UP(
            cute_x_routed,
            cute_routed_gate_weights,
            cute_gate_routed,
            cute_num_tokens,
            cute_tokens_per_expert,
            macro_offset,
            stream,
        )
        _ROUTED_GATE_UP(
            cute_x_routed,
            cute_routed_up_weights,
            cute_up_routed,
            cute_num_tokens,
            cute_tokens_per_expert,
            macro_offset,
            stream,
        )
        _SWIGLU(
            cute_gate_routed,
            cute_up_routed,
            cute_hidden_routed,
            cute_num_tokens,
            macro_offset,
            stream,
        )
        _ROUTED_DOWN(
            cute_hidden_routed,
            cute_routed_down_weights,
            cute_y_routed,
            cute_num_tokens,
            cute_tokens_per_expert,
            macro_offset,
            stream,
        )
        _COMBINE(
            combine_ptrs,
            cute_schedule_rank,
            cute_schedule_route,
            cute_num_tokens,
            cute_y_routed,
            macro_offset,
            stream,
        )

    return (
        x_routed,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        y_shared,
        y_routed,
    )
