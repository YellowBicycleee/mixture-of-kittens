"""CuTe DSL forward with QuACK Tensor Core GEMMs for Qwen BF16/EP8.

MoK owns the CuTe DSL dispatch, SwiGLU, and combine kernels and keeps its
reverse macro order and nine-tensor ABI.  QuACK 0.6.4 supplies the dense and
variable-M ``tcgen05`` GEMMs.  The existing CUDA C++ megakernel remains MoK's
default backend.  This revision still reads ``num_tokens`` to the host once;
the CUDA-Event benchmark intentionally includes that synchronization and the
Python launch gaps in its ``functional.forward`` boundary.
"""

from collections.abc import Sequence
from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import from_dlpack, make_ptr

from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw
from .forward_contract import (
    COMBINE_ROW_CHUNK_BYTES,
    COMBINE_TILE_COLUMNS,
    COMBINE_TILE_ROWS,
    DEFAULT_NUM_COMM_SMS,
    DISPATCH_ROW_CHUNK_BYTES,
    DISPATCH_TILE_COLUMNS,
    DISPATCH_TILE_ROWS,
    EP_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_LOCAL_EXPERTS,
    TOPK,
    macro_offsets,
    validate_fixed_forward_contract,
    validate_num_comm_sms,
)
from .quack_gemm import routed_gemm, shared_gemm


THREADS = 256
COMM_THREADS = 256
DISPATCH_WORKERS = DISPATCH_TILE_ROWS
DISPATCH_COLUMN_TILES = HIDDEN_SIZE // DISPATCH_TILE_COLUMNS
DISPATCH_CHUNK_BYTES = DISPATCH_ROW_CHUNK_BYTES
COMBINE_WORKERS = COMBINE_TILE_ROWS
COMBINE_COLUMN_TILES = HIDDEN_SIZE // COMBINE_TILE_COLUMNS
COMBINE_CHUNK_BYTES = COMBINE_ROW_CHUNK_BYTES


@cute.struct
class _DispatchSharedStorage:
    # CuTe DSL 4.6.2 resolves these fields at import time.  Keep the fixed-Qwen
    # extents literal: 128 mbarriers and 128 x 512 x sizeof(BF16) bytes.
    mbarriers: cute.struct.MemRange[cutlass.Int64, 128]
    tile: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, 131072],
        128,
    ]


@cute.struct
class _CombineSharedStorage:
    # Single-stage form of CUDA's 16 x 1024 combine tile.  The CUDA path keeps
    # seven such stages; this thin port establishes the exact task grain first.
    mbarriers: cute.struct.MemRange[cutlass.Int64, 16]
    tile: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, 32768],
        128,
    ]


@cute.jit
def _select_peer_address(
    peer_ptrs: list[cute.Pointer],
    peer_rank: Int32,
) -> Int64:
    """Select one of the eight symmetric pointers once per row chunk."""

    address = Int64(0)
    if peer_rank == Int32(0):
        address = peer_ptrs[0].toint()
    elif peer_rank == Int32(1):
        address = peer_ptrs[1].toint()
    elif peer_rank == Int32(2):
        address = peer_ptrs[2].toint()
    elif peer_rank == Int32(3):
        address = peer_ptrs[3].toint()
    elif peer_rank == Int32(4):
        address = peer_ptrs[4].toint()
    elif peer_rank == Int32(5):
        address = peer_ptrs[5].toint()
    elif peer_rank == Int32(6):
        address = peer_ptrs[6].toint()
    else:
        address = peer_ptrs[7].toint()
    return address


class _DispatchKernel:
    """Stage CUDA-shaped 128x512 peer tiles into the local routed ring."""

    @cute.jit
    def __call__(
        self,
        peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        macro_cu_seqlens: cute.Tensor,
        x_routed: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        macro_rows: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        assert len(peer_ptrs) == EP_SIZE
        self.kernel(
            peer_ptrs,
            schedule_peer_rank,
            schedule_peer_token_idx,
            tokens_per_expert,
            macro_cu_seqlens,
            x_routed,
            macro_offset,
            macro_rows,
        ).launch(
            # Use the CUDA config value as the size of this persistent CTA
            # worker pool.  CUDA does not guarantee one resident CTA per SM,
            # so the value controls worker count rather than physical SM
            # placement.
            grid=(num_comm_sms, 1, 1),
            block=(COMM_THREADS, 1, 1),
            smem=_DispatchSharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        macro_cu_seqlens: cute.Tensor,
        x_routed: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        macro_rows: cutlass.Constexpr,
    ):
        block = cute.arch.block_idx()[0]
        thread = cute.arch.thread_idx()[0]

        smem = utils.SmemAllocator()
        storage = smem.allocate(_DispatchSharedStorage)
        is_worker = thread < Int32(DISPATCH_WORKERS)
        mbarrier = storage.mbarriers.data_ptr() + thread
        row_chunk = (
            storage.tile.data_ptr() + thread * Int32(DISPATCH_CHUNK_BYTES)
        )

        if is_worker:
            cute.arch.mbarrier_init(mbarrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if block == Int32(0) and thread == Int32(0):
            # If S[e] is the global exclusive expert prefix, QuACK needs
            # clamp(S[e] - macro_offset, 0, macro_rows) for this macro.
            prefix = Int32(0)
            macro_cu_seqlens[0] = Int32(0)
            for expert in cutlass.range_constexpr(NUM_LOCAL_EXPERTS):
                prefix = prefix + tokens_per_expert[expert]
                local_end = prefix - Int32(macro_offset)
                local_end = cutlass.max(Int32(0), local_end)
                local_end = cutlass.min(Int32(macro_rows), local_end)
                macro_cu_seqlens[expert + 1] = local_end

        # Match CUDA dispatch_kernel: one task is 128 routed rows x 512 BF16
        # columns.  The 128 workers each move one contiguous 1024-byte row
        # chunk.  Padding never reads its undefined route index.
        task = block
        total_tasks = Int32(
            (macro_rows // DISPATCH_TILE_ROWS) * DISPATCH_COLUMN_TILES
        )
        task_stride = cute.arch.grid_dim()[0]
        phase = Int32(0)
        x_routed_base = x_routed.iterator.toint()
        while task < total_tasks:
            if is_worker:
                row_tile = task // Int32(DISPATCH_COLUMN_TILES)
                column_tile = task - row_tile * Int32(DISPATCH_COLUMN_TILES)
                local_row = row_tile * Int32(DISPATCH_TILE_ROWS) + thread
                global_row = local_row + Int32(macro_offset)
                peer_rank = schedule_peer_rank[global_row]
                if peer_rank >= Int32(0):
                    route_idx = schedule_peer_token_idx[global_row]
                    source_token = route_idx // Int32(TOPK)
                    src_address = (
                        _select_peer_address(peer_ptrs, peer_rank)
                        + Int64(source_token) * Int64(HIDDEN_SIZE)
                        * Int64(BFloat16.width // 8)
                        + Int64(column_tile * Int32(DISPATCH_CHUNK_BYTES))
                    )
                    # This worker is the sole producer and consumer of its
                    # row mbarrier.  Wait for peer GMEM -> SMEM to complete
                    # before the same bytes are submitted as SMEM -> ring.
                    tma_load_1d_raw(
                        row_chunk,
                        src_address,
                        mbarrier,
                        Int32(DISPATCH_CHUNK_BYTES),
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbarrier, Int32(DISPATCH_CHUNK_BYTES)
                    )
                    cute.arch.mbarrier_wait(mbarrier, phase)
                    phase = phase ^ Int32(1)
                else:
                    zero_words = cute.make_tensor(
                        cute.recast_ptr(row_chunk, dtype=Uint32),
                        cute.make_layout((DISPATCH_CHUNK_BYTES // 4,)),
                    )
                    for word in cutlass.range_constexpr(
                        DISPATCH_CHUNK_BYTES // 4
                    ):
                        zero_words[word] = Uint32(0)
                    cute.arch.fence_proxy("async.shared", space="cta")

                dst_address = (
                    x_routed_base
                    + Int64(local_row) * Int64(HIDDEN_SIZE)
                    * Int64(BFloat16.width // 8)
                    + Int64(column_tile * Int32(DISPATCH_CHUNK_BYTES))
                )
                tma_store_1d_raw(
                    dst_address,
                    row_chunk,
                    Int32(DISPATCH_CHUNK_BYTES),
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
            task = task + task_stride


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
    """Stage CUDA-shaped 16x1024 local tiles, then push them to peers."""

    @cute.jit
    def __call__(
        self,
        combine_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        y_routed: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        macro_rows: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        assert len(combine_ptrs) == EP_SIZE
        self.kernel(
            combine_ptrs,
            schedule_peer_rank,
            schedule_peer_token_idx,
            y_routed,
            macro_offset,
            macro_rows,
        ).launch(
            grid=(num_comm_sms, 1, 1),
            block=(COMM_THREADS, 1, 1),
            smem=_CombineSharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        peers: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        y_routed: cute.Tensor,
        macro_offset: cutlass.Constexpr,
        macro_rows: cutlass.Constexpr,
    ):
        block = cute.arch.block_idx()[0]
        thread = cute.arch.thread_idx()[0]

        smem = utils.SmemAllocator()
        storage = smem.allocate(_CombineSharedStorage)
        is_worker = thread < Int32(COMBINE_WORKERS)
        mbarrier = storage.mbarriers.data_ptr() + thread
        row_chunk = (
            storage.tile.data_ptr() + thread * Int32(COMBINE_CHUNK_BYTES)
        )

        if is_worker:
            cute.arch.mbarrier_init(mbarrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        task = block
        total_tasks = Int32(
            (macro_rows // COMBINE_TILE_ROWS) * COMBINE_COLUMN_TILES
        )
        task_stride = cute.arch.grid_dim()[0]
        phase = Int32(0)
        y_routed_base = y_routed.iterator.toint()
        while task < total_tasks:
            if is_worker:
                row_tile = task // Int32(COMBINE_COLUMN_TILES)
                column_tile = task - row_tile * Int32(COMBINE_COLUMN_TILES)
                local_row = row_tile * Int32(COMBINE_TILE_ROWS) + thread
                global_row = local_row + Int32(macro_offset)
                peer_rank = schedule_peer_rank[global_row]
                if peer_rank >= Int32(0):
                    route_idx = schedule_peer_token_idx[global_row]
                    src_address = (
                        y_routed_base
                        + Int64(local_row) * Int64(HIDDEN_SIZE)
                        * Int64(BFloat16.width // 8)
                        + Int64(column_tile * Int32(COMBINE_CHUNK_BYTES))
                    )
                    tma_load_1d_raw(
                        row_chunk,
                        src_address,
                        mbarrier,
                        Int32(COMBINE_CHUNK_BYTES),
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbarrier, Int32(COMBINE_CHUNK_BYTES)
                    )
                    cute.arch.mbarrier_wait(mbarrier, phase)
                    phase = phase ^ Int32(1)

                    # Combine preserves k in route_idx and must promote the
                    # byte offset to Int64 before the 100K/rank-sized product.
                    dst_address = (
                        _select_peer_address(peers, peer_rank)
                        + Int64(route_idx) * Int64(HIDDEN_SIZE)
                        * Int64(BFloat16.width // 8)
                        + Int64(column_tile * Int32(COMBINE_CHUNK_BYTES))
                    )
                    tma_store_1d_raw(
                        dst_address,
                        row_chunk,
                        Int32(COMBINE_CHUNK_BYTES),
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)

                # Padding neither reads its undefined route index nor writes a
                # peer combine buffer.
            task = task + task_stride


_DISPATCH = _DispatchKernel()
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
    num_comm_sms: int = DEFAULT_NUM_COMM_SMS,
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
    validate_num_comm_sms(num_comm_sms)

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
    macro_cu_seqlens = torch.empty(
        (NUM_LOCAL_EXPERTS + 1,), dtype=torch.int32, device=device
    )

    cute_schedule_rank = from_dlpack(schedule.peer_rank, assumed_align=16)
    cute_schedule_route = from_dlpack(schedule.peer_token_idx, assumed_align=16)
    cute_num_tokens = from_dlpack(schedule.num_tokens, assumed_align=16)
    cute_tokens_per_expert = from_dlpack(schedule.tokens_per_expert, assumed_align=16)
    cute_macro_cu_seqlens = from_dlpack(macro_cu_seqlens, assumed_align=16)
    cute_x_routed = from_dlpack(x_routed, assumed_align=16)
    cute_gate_shared = from_dlpack(gate_shared, assumed_align=16)
    cute_gate_routed = from_dlpack(gate_routed, assumed_align=16)
    cute_up_shared = from_dlpack(up_shared, assumed_align=16)
    cute_up_routed = from_dlpack(up_routed, assumed_align=16)
    cute_hidden_shared = from_dlpack(hidden_shared, assumed_align=16)
    cute_hidden_routed = from_dlpack(hidden_routed, assumed_align=16)
    cute_y_routed = from_dlpack(y_routed, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    # Shared expert is independent of the routed schedule.
    shared_gemm(workspace.x_buffer, shared_gate_weights, gate_shared)
    shared_gemm(workspace.x_buffer, shared_up_weights, up_shared)
    _SWIGLU(cute_gate_shared, cute_up_shared, cute_hidden_shared, None, 0, stream)
    shared_gemm(hidden_shared, shared_down_weights, y_shared)

    x_ptrs = _bf16_ptrs(workspace.x_buffer_ptrs)
    combine_ptrs = _bf16_ptrs(workspace.combine_buffer_ptrs)
    # Read the device scalar once so we launch only real macros instead of the
    # larger capacity envelope.  Removing this D2H sync is separate follow-up
    # work; the GEMMs here are already Blackwell Tensor Core kernels.
    routed_num_tokens = int(schedule.num_tokens.item())
    if not 0 <= routed_num_tokens <= workspace.schedule_capacity:
        raise RuntimeError("schedule.num_tokens exceeds the schedule capacity")
    for macro_offset in macro_offsets(routed_num_tokens, m):
        macro_rows = min(m, routed_num_tokens - macro_offset)
        _DISPATCH(
            x_ptrs,
            cute_schedule_rank,
            cute_schedule_route,
            cute_tokens_per_expert,
            cute_macro_cu_seqlens,
            cute_x_routed,
            macro_offset,
            macro_rows,
            num_comm_sms,
            stream,
        )
        routed_gemm(
            x_routed[:macro_rows],
            routed_gate_weights,
            gate_routed[:macro_rows],
            macro_cu_seqlens,
        )
        routed_gemm(
            x_routed[:macro_rows],
            routed_up_weights,
            up_routed[:macro_rows],
            macro_cu_seqlens,
        )
        _SWIGLU(
            cute_gate_routed,
            cute_up_routed,
            cute_hidden_routed,
            cute_num_tokens,
            macro_offset,
            stream,
        )
        routed_gemm(
            hidden_routed[:macro_rows],
            routed_down_weights,
            y_routed[:macro_rows],
            macro_cu_seqlens,
        )
        _COMBINE(
            combine_ptrs,
            cute_schedule_rank,
            cute_schedule_route,
            cute_y_routed,
            macro_offset,
            macro_rows,
            num_comm_sms,
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
