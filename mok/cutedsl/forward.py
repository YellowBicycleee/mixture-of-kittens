"""CuTe DSL forward with QuACK Tensor Core GEMMs for Qwen BF16/EP8.

MoK owns the CuTe DSL dispatch, SwiGLU, and combine kernels and keeps its
reverse macro order and nine-tensor ABI.  QuACK 0.6.4 supplies the dense and
variable-M ``tcgen05`` GEMMs.  The existing CUDA C++ megakernel remains MoK's
default backend.  CuTe schedules carry a host mirror of ``num_tokens`` from
schedule construction; legacy schedules retain the old synchronous fallback.
Routed work advances through fixed 64K windows on dispatch, caller-compute, and
combine streams while the shared expert runs on its cached auxiliary stream.
Pipeline-v2 uses QuACK's public custom epilogues for shared and routed
Gate+Up+SwiGLU. Routed Gate/Up are stored only for macro zero. Down, dispatch,
and combine remain deliberately unchanged in this first pipeline stage.
"""

from collections.abc import Sequence
import threading

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Int32, Int64, Uint32
from cutlass.cute.runtime import from_dlpack, make_ptr

from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw
from .forward_contract import (
    COMBINE_ARENA_BYTES,
    COMBINE_PIPE_DEPTH,
    COMBINE_ROW_CHUNK_BYTES,
    COMBINE_STORAGE_BYTES,
    COMBINE_TILE_COLUMNS,
    COMBINE_TILE_ROWS,
    DEFAULT_NUM_COMM_SMS,
    DISPATCH_ROW_CHUNK_BYTES,
    DISPATCH_STORAGE_BYTES,
    DISPATCH_TILE_COLUMNS,
    DISPATCH_TILE_ROWS,
    EP_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_LOCAL_EXPERTS,
    TOPK,
    WAVEFRONT_WINDOW_ROWS,
    resolve_routed_num_tokens,
    standalone_comm_worker_grids,
    validate_combine_smem_capacity,
    validate_fixed_forward_contract,
    validate_num_comm_sms,
    wavefront_windows,
    should_store_routed_preact,
)
from .quack_gemm import (
    packed_routed_gate_up_weights,
    packed_shared_gate_up_weights,
    routed_gated_swiglu,
    routed_gemm,
    shared_gated_swiglu,
    shared_gemm,
)


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
    # extents literal: 128 mbarriers and 128 x 128 x sizeof(BF16) bytes.
    mbarriers: cute.struct.MemRange[cutlass.Int64, 128]
    tile: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, 32768],
        128,
    ]


@cute.struct
class _CombineSharedStorage:
    # Match CUDA's seven 16 x 1024 BF16 stages.  The barrier prefix pads to the
    # arena's 128-byte alignment, for 229504 bytes of dynamic shared memory.
    mbarriers: cute.struct.MemRange[cutlass.Int64, 7]
    arena: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, 229376],
        128,
    ]


assert _DispatchSharedStorage.size_in_bytes() == DISPATCH_STORAGE_BYTES
assert COMBINE_ARENA_BYTES == 229376
assert _CombineSharedStorage.size_in_bytes() == COMBINE_STORAGE_BYTES


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
    """Stage 128x128 peer tiles into the local routed ring."""

    @cute.jit
    def __call__(
        self,
        peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        macro_cu_seqlens: cute.Tensor,
        x_routed: cute.Tensor,
        global_offset: Int32,
        macro_rows: cutlass.Constexpr,
        worker_ctas: cutlass.Constexpr,
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
            global_offset,
            macro_rows,
        ).launch(
            grid=(worker_ctas, 1, 1),
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
        global_offset: Int32,
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
            # clamp(S[e] - global_offset, 0, macro_rows) for this window.
            prefix = Int32(0)
            macro_cu_seqlens[0] = Int32(0)
            for expert in cutlass.range_constexpr(NUM_LOCAL_EXPERTS):
                prefix = prefix + tokens_per_expert[expert]
                local_end = prefix - global_offset
                local_end = cutlass.max(Int32(0), local_end)
                local_end = cutlass.min(Int32(macro_rows), local_end)
                macro_cu_seqlens[expert + 1] = local_end

        # One task is 128 routed rows x 128 BF16 columns.  The 128 workers each
        # move one contiguous 256-byte row chunk.  Halving the tile storage
        # permits a four-CTA-per-SM worker grid while preserving row ownership.
        # Padding never reads its undefined route index.
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
                global_row = local_row + global_offset
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


class _CombineKernel:
    """Pipeline seven CUDA-shaped 16x1024 local tiles to peers."""

    @cute.jit
    def __call__(
        self,
        combine_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        y_routed: cute.Tensor,
        global_offset: Int32,
        macro_rows: cutlass.Constexpr,
        worker_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        assert len(combine_ptrs) == EP_SIZE
        self.kernel(
            combine_ptrs,
            schedule_peer_rank,
            schedule_peer_token_idx,
            y_routed,
            global_offset,
            macro_rows,
        ).launch(
            grid=(worker_ctas, 1, 1),
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
        global_offset: Int32,
        macro_rows: cutlass.Constexpr,
    ):
        block = cute.arch.block_idx()[0]
        thread = cute.arch.thread_idx()[0]

        smem = utils.SmemAllocator()
        storage = smem.allocate(_CombineSharedStorage)
        is_worker = thread < Int32(COMBINE_WORKERS)

        if thread == Int32(0):
            for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                cute.arch.mbarrier_init(
                    storage.mbarriers.data_ptr() + Int32(stage), 1
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        task = block
        total_tiles = Int32(
            (macro_rows // COMBINE_TILE_ROWS) * COMBINE_COLUMN_TILES
        )
        total_tasks = (
            total_tiles + Int32(COMBINE_PIPE_DEPTH - 1)
        ) // Int32(COMBINE_PIPE_DEPTH)
        task_stride = cute.arch.grid_dim()[0]
        phase_bits = Int32(0)
        y_routed_base = y_routed.iterator.toint()
        while task < total_tasks:
            first_tile = task * Int32(COMBINE_PIPE_DEPTH)
            valid_stages = cutlass.min(
                Int32(COMBINE_PIPE_DEPTH), total_tiles - first_tile
            )
            valid_flags = cute.make_tensor(
                cute.recast_ptr(storage.arena.data_ptr(), dtype=Int32),
                cute.make_layout((COMBINE_PIPE_DEPTH, COMBINE_WORKERS)),
            )
            peer_ranks = [Int32(-1)] * COMBINE_PIPE_DEPTH
            route_indices = [Int32(-1)] * COMBINE_PIPE_DEPTH
            local_rows = [Int32(0)] * COMBINE_PIPE_DEPTH
            column_tiles = [Int32(0)] * COMBINE_PIPE_DEPTH

            for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                if Int32(stage) < valid_stages:
                    tile = first_tile + Int32(stage)
                    row_tile = tile // Int32(COMBINE_COLUMN_TILES)
                    column_tile = tile - row_tile * Int32(COMBINE_COLUMN_TILES)
                    local_row = row_tile * Int32(COMBINE_TILE_ROWS) + thread
                    local_rows[stage] = local_row
                    column_tiles[stage] = column_tile
                    if is_worker:
                        global_row = local_row + global_offset
                        peer_ranks[stage] = schedule_peer_rank[global_row]
                        if peer_ranks[stage] >= Int32(0):
                            route_indices[stage] = schedule_peer_token_idx[
                                global_row
                            ]
                        valid_flags[stage, thread] = Int32(
                            peer_ranks[stage] >= Int32(0)
                        )
            cute.arch.sync_threads()

            # Retire every generic-proxy valid flag before any async-proxy G2S
            # load can overwrite the shared arena.
            valid_counts = [Int32(0)] * COMBINE_PIPE_DEPTH
            if thread == Int32(0):
                for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                    if Int32(stage) < valid_stages:
                        for worker in cutlass.range_constexpr(COMBINE_WORKERS):
                            valid_counts[stage] = (
                                valid_counts[stage] + valid_flags[stage, worker]
                            )
            cute.arch.sync_threads()
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            # Existing stages advance their barrier phase even when every row
            # is padding.  Nonexistent tail stages are never armed or waited
            # and do not touch an arena slot.
            for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                if Int32(stage) < valid_stages:
                    mbarrier = storage.mbarriers.data_ptr() + Int32(stage)
                    if thread == Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbarrier,
                            valid_counts[stage] * Int32(COMBINE_CHUNK_BYTES),
                        )
                    cute.arch.sync_threads()
                    if is_worker:
                        if peer_ranks[stage] >= Int32(0):
                            row_chunk = (
                                storage.arena.data_ptr()
                                + Int32(
                                    stage
                                    * COMBINE_WORKERS
                                    * COMBINE_CHUNK_BYTES
                                )
                                + thread * Int32(COMBINE_CHUNK_BYTES)
                            )
                            source = (
                                y_routed_base
                                + Int64(local_rows[stage]) * Int64(HIDDEN_SIZE)
                                * Int64(BFloat16.width // 8)
                                + Int64(
                                    column_tiles[stage]
                                    * Int32(COMBINE_CHUNK_BYTES)
                                )
                            )
                            tma_load_1d_raw(
                                row_chunk,
                                source,
                                mbarrier,
                                Int32(COMBINE_CHUNK_BYTES),
                            )

            for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                if Int32(stage) < valid_stages:
                    phase = (phase_bits >> Int32(stage)) & Int32(1)
                    cute.arch.mbarrier_wait(
                        storage.mbarriers.data_ptr() + Int32(stage), phase
                    )
                    phase_bits = phase_bits ^ Int32(1 << stage)
                    if is_worker:
                        if peer_ranks[stage] >= Int32(0):
                            row_chunk = (
                                storage.arena.data_ptr()
                                + Int32(
                                    stage
                                    * COMBINE_WORKERS
                                    * COMBINE_CHUNK_BYTES
                                )
                                + thread * Int32(COMBINE_CHUNK_BYTES)
                            )
                            destination = (
                                _select_peer_address(peers, peer_ranks[stage])
                                + Int64(route_indices[stage])
                                * Int64(HIDDEN_SIZE)
                                * Int64(BFloat16.width // 8)
                                + Int64(
                                    column_tiles[stage]
                                    * Int32(COMBINE_CHUNK_BYTES)
                                )
                            )
                            tma_store_1d_raw(
                                destination,
                                row_chunk,
                                Int32(COMBINE_CHUNK_BYTES),
                            )
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_threads()
            task = task + task_stride


_DISPATCH = _DispatchKernel()
_COMBINE = _CombineKernel()

# ``@cute.jit`` still regenerates and validates MLIR on every Python call, even
# when its disk cache hits.  Keep the small fixed-Qwen specialization set as
# compiled executors so the hot path only adapts runtime pointers and launches.
# Keys contain only static specialization data; tensor addresses and streams
# remain runtime arguments.
_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_LOCK = threading.Lock()
_AUX_STREAMS: dict[tuple[str, int], torch.cuda.Stream] = {}
_AUX_STREAMS_LOCK = threading.Lock()


def _compiled_executor(key: tuple[object, ...], op, *compile_args):
    executor = _EXECUTOR_CACHE.get(key)
    if executor is not None:
        return executor
    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            executor = cute.compile(op, *compile_args)
            _EXECUTOR_CACHE[key] = executor
    return executor


def _aux_stream_for_device(role: str, device_index: int) -> torch.cuda.Stream:
    key = (role, device_index)
    stream = _AUX_STREAMS.get(key)
    if stream is not None:
        return stream
    with _AUX_STREAMS_LOCK:
        stream = _AUX_STREAMS.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device_index)
            _AUX_STREAMS[key] = stream
    return stream


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

    # These are the exact shapes returned by the CUDA BF16 custom op. Routed
    # buffers are ring storage; reverse macro execution intentionally leaves
    # macro 0 resident for the existing CUDA backward path. Pipeline-v2's save
    # epilogue writes separate contiguous Gate/Up only for macro zero.
    x_routed = torch.empty((m, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    gate_shared = torch.empty((t, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    gate_routed = torch.empty((m, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    up_shared = torch.empty((t, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    up_routed = torch.empty((m, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    hidden_shared = torch.empty((t, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    hidden_routed = torch.empty((m, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    y_shared = torch.empty((t, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    y_routed = torch.empty((m, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    packed_shared_weights = packed_shared_gate_up_weights(
        shared_gate_weights,
        shared_up_weights,
    )
    packed_routed_weights = packed_routed_gate_up_weights(
        routed_gate_weights,
        routed_up_weights,
    )
    num_ring_slots = (m + WAVEFRONT_WINDOW_ROWS - 1) // WAVEFRONT_WINDOW_ROWS
    cu_seqlens_width = NUM_LOCAL_EXPERTS + 1
    # A 68-int stride keeps every 65-int QuACK view 16-byte aligned.
    cu_seqlens_stride = (cu_seqlens_width + 3) // 4 * 4
    cu_seqlens_storage = torch.empty(
        (num_ring_slots, cu_seqlens_stride), dtype=torch.int32, device=device
    )
    window_cu_seqlens = tuple(
        cu_seqlens_storage[slot, :cu_seqlens_width]
        for slot in range(num_ring_slots)
    )

    cute_schedule_rank = from_dlpack(schedule.peer_rank, assumed_align=16)
    cute_schedule_route = from_dlpack(schedule.peer_token_idx, assumed_align=16)
    cute_tokens_per_expert = from_dlpack(schedule.tokens_per_expert, assumed_align=16)
    cute_window_cu_seqlens = tuple(
        from_dlpack(cu_seqlens, assumed_align=16)
        for cu_seqlens in window_cu_seqlens
    )

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    if torch.cuda.current_device() != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    caller_torch_stream = torch.cuda.current_stream(device)
    caller_cu_stream = cuda.CUstream(caller_torch_stream.cuda_stream)
    shared_torch_stream = _aux_stream_for_device("shared", device_index)
    dispatch_torch_stream = _aux_stream_for_device("dispatch", device_index)
    combine_torch_stream = _aux_stream_for_device("combine", device_index)
    dispatch_cu_stream = cuda.CUstream(dispatch_torch_stream.cuda_stream)
    combine_cu_stream = cuda.CUstream(combine_torch_stream.cuda_stream)
    device_properties = torch.cuda.get_device_properties(device)
    combine_smem_bytes = _CombineSharedStorage.size_in_bytes()
    assert combine_smem_bytes == COMBINE_STORAGE_BYTES
    validate_combine_smem_capacity(
        device_properties.shared_memory_per_block_optin
    )
    sm_count = device_properties.multi_processor_count
    dispatch_worker_ctas, combine_worker_ctas = standalone_comm_worker_grids(
        sm_count
    )

    inputs_ready = torch.cuda.Event()
    inputs_ready.record(caller_torch_stream)
    shared_torch_stream.wait_event(inputs_ready)
    dispatch_torch_stream.wait_event(inputs_ready)
    try:
        with torch.cuda.stream(shared_torch_stream):
            shared_gated_swiglu(
                workspace.x_buffer,
                packed_shared_weights,
                gate_shared,
                up_shared,
                hidden_shared,
            )
            shared_gemm(hidden_shared, shared_down_weights, y_shared)

        x_ptrs = _bf16_ptrs(workspace.x_buffer_ptrs)
        combine_ptrs = _bf16_ptrs(workspace.combine_buffer_ptrs)
        routed_num_tokens = resolve_routed_num_tokens(
            schedule.num_tokens,
            schedule.num_tokens_host,
            workspace.schedule_capacity,
        )
        last_down_done: dict[int, torch.cuda.Event] = {}
        last_combine_done: dict[int, torch.cuda.Event] = {}
        for global_offset, ring_offset, window_rows in wavefront_windows(
            routed_num_tokens, m
        ):
            slot = ring_offset // WAVEFRONT_WINDOW_ROWS
            ring_slice = slice(ring_offset, ring_offset + window_rows)
            x_window = x_routed[ring_slice]
            gate_window = gate_routed[ring_slice]
            up_window = up_routed[ring_slice]
            hidden_window = hidden_routed[ring_slice]
            y_window = y_routed[ring_slice]
            cu_seqlens = window_cu_seqlens[slot]

            cute_x_window = from_dlpack(x_window, assumed_align=16)
            cute_y_window = from_dlpack(y_window, assumed_align=16)

            dispatch_done = torch.cuda.Event()
            down_done = torch.cuda.Event()
            combine_done = torch.cuda.Event()
            previous_down_done = last_down_done.get(slot)
            previous_combine_done = last_combine_done.get(slot)
            # Dispatch rewrites x and cu_seqlens; Down is their final reader.
            if previous_down_done is not None:
                dispatch_torch_stream.wait_event(previous_down_done)
            dispatch = _compiled_executor(
                (
                    "dispatch_window",
                    device_index,
                    workspace.schedule_capacity,
                    window_rows,
                    dispatch_worker_ctas,
                ),
                _DISPATCH,
                x_ptrs,
                cute_schedule_rank,
                cute_schedule_route,
                cute_tokens_per_expert,
                cute_window_cu_seqlens[slot],
                cute_x_window,
                global_offset,
                window_rows,
                dispatch_worker_ctas,
                dispatch_cu_stream,
            )
            dispatch(
                x_ptrs,
                cute_schedule_rank,
                cute_schedule_route,
                cute_tokens_per_expert,
                cute_window_cu_seqlens[slot],
                cute_x_window,
                global_offset,
                dispatch_cu_stream,
            )
            dispatch_done.record(dispatch_torch_stream)

            caller_torch_stream.wait_event(dispatch_done)
            store_preact = should_store_routed_preact(global_offset, m)
            routed_gated_swiglu(
                x_window,
                packed_routed_weights,
                hidden_window,
                cu_seqlens,
                store_preact=store_preact,
                gate_output=gate_window,
                up_output=up_window,
            )
            # Gate/Up/SwiGLU may overlap old Combine; Down rewrites its y slot.
            if previous_combine_done is not None:
                caller_torch_stream.wait_event(previous_combine_done)
            routed_gemm(
                hidden_window,
                routed_down_weights,
                y_window,
                cu_seqlens,
            )
            down_done.record(caller_torch_stream)

            combine_torch_stream.wait_event(down_done)
            combine = _compiled_executor(
                (
                    "combine_window",
                    device_index,
                    workspace.schedule_capacity,
                    window_rows,
                    combine_worker_ctas,
                ),
                _COMBINE,
                combine_ptrs,
                cute_schedule_rank,
                cute_schedule_route,
                cute_y_window,
                global_offset,
                window_rows,
                combine_worker_ctas,
                combine_cu_stream,
            )
            combine(
                combine_ptrs,
                cute_schedule_rank,
                cute_schedule_route,
                cute_y_window,
                global_offset,
                combine_cu_stream,
            )
            combine_done.record(combine_torch_stream)
            last_down_done[slot] = down_done
            last_combine_done[slot] = combine_done
    finally:
        caller_torch_stream.wait_stream(dispatch_torch_stream)
        caller_torch_stream.wait_stream(combine_torch_stream)
        caller_torch_stream.wait_stream(shared_torch_stream)

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
