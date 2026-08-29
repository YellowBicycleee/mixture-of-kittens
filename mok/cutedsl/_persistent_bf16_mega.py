"""BF16 CuTe DSL port of MoK's single persistent Forward mega-kernel.

This module preserves the CUDA execution boundary in ``forward.cuh``: one
cluster-2 launch, a fixed communication-cluster prefix, and a CLC-scheduled
compute suffix.  Communication advances macrobatches in reverse order while
the compute decoder advances 4K minibatches forward inside each macro.  FC1
uses the CUDA V1-C2 *collective* operand mapping (CTA0 Gate, CTA1 Up), performs
the BF16 rounding seam and SwiGLU in its epilogue, and never materializes a
packed Gate/Up weight mirror or a host-wavefront intermediate launch.  The
accepted forward selector is independently CLC depth 1 with G1D1 task grouping.

The device module remains private behind the explicit public CuTe backend.  It
has no fallback and does not alter the default CUDA backend or backward
implementation.
"""

import importlib.metadata as metadata

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

import quack
from quack.activation import swiglu
from quack.gemm_base import NamedBarrierGemm
from quack.pipeline import PipelineTmaUmma, PipelineUmmaAsync

from ._persistent_bf16_gemm import (
    AB_STAGES as _GATE_AB_STAGES,
    ACC_STAGES as _GATE_ACC_STAGES,
    A_BYTES as GEMM_A_BYTES,
    B_BYTES as GEMM_B_BYTES,
    D_STAGES as _GATE_D_STAGES,
    D_BYTES as GEMM_D_BYTES,
    EPILOGUE_N as _GATE_EPILOGUE_N,
    K_TILE as _GATE_TILE_K,
    LOGICAL_N as _GATE_LOGICAL_N,
    MMA_WARP as _GATE_MMA_WARP,
    PACKED_N as _GATE_TILE_N,
    TILE_M as _GATE_TILE_M,
    TMA_WARP as _GATE_AB_LOAD_WARP,
    _Fc1Collective,
    _make_tma_block_load_fn,
)
from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw
from .forward_contract import (
    COMBINE_ROW_CHUNK_BYTES,
    COMBINE_TILE_COLUMNS,
    COMBINE_TILE_ROWS,
    DISPATCH_TILE_ROWS,
    EP_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    TOPK,
)
from .persistent_bf16_contract import (
    CLUSTER_SIZE,
    MLP_SUPERGROUP_SIZE,
    NUM_LOCAL_EXPERTS,
)


CLUSTER_SHAPE = (CLUSTER_SIZE, 1, 1)
THREADS_PER_CTA = 256
WARPS_PER_CTA = THREADS_PER_CTA // 32
CLC_RESPONSE_BYTES = 16
CLC_PIPE_DEPTH = 1
FUSED_GATE_UP_TASK_GROUP_SIZE = 1
FUSED_DOWN_TASK_GROUP_SIZE = 1
CLC_DRAIN_WARPS = WARPS_PER_CTA
CLC_COMPLETION_WARPS = CLUSTER_SIZE * WARPS_PER_CTA
COMBINE_PIPE_DEPTH = 7
MLP_TILE_ROWS = 256
MLP_TILE_COLUMNS = 256
SWIGLU_TILE_ROWS = 128
SWIGLU_TILE_COLUMNS = 128
SWIGLU_PIPE_DEPTH = 3
DISPATCH_TILE_COLUMNS = 512
DISPATCH_ROW_CHUNK_BYTES = DISPATCH_TILE_COLUMNS * 2
COMM_ARENA_BYTES = 7 * 16 * 1024 * 2  # CUDA combine: 7 x 16 x 1024 BF16
GEMM_ARENA_BYTES = GEMM_A_BYTES + GEMM_B_BYTES + GEMM_D_BYTES
REUSABLE_ARENA_BYTES = max(COMM_ARENA_BYTES, GEMM_ARENA_BYTES)
assert REUSABLE_ARENA_BYTES == 229376

# Keep this private device path fail-closed on the selector frozen by the
# accepted harness.  A future CLC2/grouped experiment needs a separate audited
# response ring and raw-task loop instead of silently changing this schedule.
assert CLC_PIPE_DEPTH == 1
assert FUSED_GATE_UP_TASK_GROUP_SIZE == 1
assert FUSED_DOWN_TASK_GROUP_SIZE == 1

_REQUIRED_CUTLASS_DSL = "4.6.2"
_REQUIRED_QUACK_VERSION = "0.6.4"
if metadata.version("nvidia-cutlass-dsl") != _REQUIRED_CUTLASS_DSL:
    raise RuntimeError(
        "MoK's persistent CuTe DSL forward requires "
        f"nvidia-cutlass-dsl=={_REQUIRED_CUTLASS_DSL}"
    )
if quack.__version__ != _REQUIRED_QUACK_VERSION:
    raise RuntimeError(
        "MoK's persistent CuTe DSL forward requires "
        f"quack-kernels=={_REQUIRED_QUACK_VERSION}; got {quack.__version__}"
    )

_MLP_LOGICAL_K_VALUES = (INTERMEDIATE_SIZE, HIDDEN_SIZE)
assert all(k % _GATE_TILE_K == 0 for k in _MLP_LOGICAL_K_VALUES)
_GATE_CTA_M = 128
_GATE_EPILOGUE_SUBTILES = _GATE_TILE_N // _GATE_EPILOGUE_N
_GATE_CLC_WARP = 5
_GATE_BF16_IDLE_WARP = 6
@cute.struct
class _PersistentSharedStorage:
    # One depth-1 CLC full/empty pair plus one 16-byte response.
    clc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2]
    clc_response: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, 4],
        16,
    ]
    # Once a live compute cluster receives a successful capacity-only CLC
    # response, one independent drain slot per warp consumes the remaining
    # no-op responses until hardware reports failure.  Each slot has a full
    # and empty mbarrier plus one 16-byte response.
    clc_drain_mbarriers: cute.struct.MemRange[
        cutlass.Int64,
        16,
    ]
    clc_drain_responses: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, 32],
        16,
    ]

    # Kept explicit for the QuACK collective seam.  Six is the A/B
    # load pipeline, while four is the two-stage accumulator full/empty pair.
    # The eight N=32 epilogue subtiles are not an eight-stage output ring;
    # CUDA reuses three 128x32 BF16 D tiles.
    gemm_ab_mbarriers: cute.struct.MemRange[cutlass.Int64, 12]
    gemm_acc_mbarriers: cute.struct.MemRange[cutlass.Int64, 4]
    tmem_dealloc_mbarrier: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32

    # Dispatch has one CTA-wide transaction barrier; combine has one barrier
    # per seven-stage tile.  All roles alias the same arena rather than adding
    # their per-role storage requirements.
    dispatch_mbarrier: cutlass.Int64
    combine_mbarriers: cute.struct.MemRange[cutlass.Int64, 7]
    # Build the E=64 routed row-block prefix once per CTA.  The 260-byte table
    # fits in the existing padding before the 1024-byte-aligned arena, so this
    # does not increase the 230400-byte shared-storage footprint.
    expert_row_block_offsets: cute.struct.MemRange[
        cutlass.Int32,
        NUM_LOCAL_EXPERTS + 1,
    ]
    arena: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, 229376],
        1024,
    ]
@cute.jit
def _select_peer_address(
    peer_ptrs: list[cute.Pointer],
    peer_rank: Int32,
) -> Int64:
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


@cute.jit
def _counter_wait_gpu(
    counter: cute.Tensor,
    index: Int32,
    required: Int32,
) -> None:
    """Acquire a monotonically increasing counter at CUDA GPU scope."""

    pointer = counter.iterator + index
    observed = Int32(0)
    while observed < required:
        observed = cute.arch.load(
            pointer,
            Int32,
            sem="acquire",
            scope="gpu",
        )


@cute.jit
def _counter_arrive_gpu(
    counter: cute.Tensor,
    index: Int32,
    increment: Int32 = Int32(1),
) -> None:
    """Release one monotonic arrival at CUDA GPU scope."""

    cute.arch.atomic_add(
        counter.iterator + index,
        increment,
        sem="release",
        scope="gpu",
    )


@cute.jit
def _ceil_div_i32(value: Int32, divisor: Int32) -> Int32:
    return (value + divisor - Int32(1)) // divisor


@cute.jit
def _swizzled_mlp_tile_coord_i32(
    row_blocks: Int32,
    col_blocks: Int32,
    task_idx: Int32,
) -> tuple[Int32, Int32]:
    """Device form of ThunderKittens' row-major swizzle with group size 8."""

    supergroup_size = Int32(MLP_SUPERGROUP_SIZE)
    supergroup_numel = row_blocks * supergroup_size
    supergroup_idx = task_idx // supergroup_numel
    supersection_cols = (col_blocks // supergroup_size) * supergroup_size
    supersection_numel = row_blocks * supersection_cols
    finalsection_cols = col_blocks - supersection_cols
    # Down has 16 columns, exactly two full supergroups.  Its remainder
    # branch is unreachable for valid tasks, but CuTe still lowers both
    # dynamic regions; keep the dead-region divisor nonzero.
    safe_finalsection_cols = cutlass.max(Int32(1), finalsection_cols)
    row_idx = Int32(0)
    col_idx = Int32(0)
    if task_idx < supersection_numel:
        row_idx = (task_idx % supergroup_numel) // supergroup_size
        col_idx = supergroup_idx * supergroup_size + task_idx % supergroup_size
    else:
        remainder_task_id = task_idx - supersection_numel
        row_idx = remainder_task_id // safe_finalsection_cols
        col_idx = (
            supersection_cols
            + remainder_task_id % safe_finalsection_cols
        )
    if supergroup_idx % Int32(2) != Int32(0):
        row_idx = row_blocks - row_idx - Int32(1)
    return row_idx, col_idx


@cute.jit
def _expert_for_routed_row_block_i32(
    expert_row_block_offsets: cute.Pointer,
    global_row_block: Int32,
) -> Int32:
    """Return the E=64 upper-bound owner with six fixed comparisons."""

    first = Int32(0)
    last = Int32(NUM_LOCAL_EXPERTS - 1)
    for _ in cutlass.range_constexpr(6):
        middle = (first + last) // Int32(2)
        if expert_row_block_offsets[middle + Int32(1)] <= global_row_block:
            first = middle + Int32(1)
        else:
            last = middle
    return first


@cute.jit
def _routed_gate_up_tile_i32(
    expert_row_block_offsets: cute.Pointer,
    num_tokens: Int32,
    macrobatch_index: Int32,
    minibatch_index: Int32,
    task_index: Int32,
    shared_gate_up_tasks: cutlass.Constexpr,
    macrobatch_size: cutlass.Constexpr,
    minibatch_size: cutlass.Constexpr,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Decode CUDA's routed Gate/Up expert tile and readiness edge.

    The returned row coordinate is relative to the macrobatch ring buffer.
    ``valid`` is zero when this fixed-capacity task has no expert rows in the
    runtime minibatch.
    """

    global_minibatch = (
        macrobatch_index * Int32(macrobatch_size // minibatch_size)
        + minibatch_index
    )
    minibatch_row_blocks = Int32(minibatch_size // MLP_TILE_ROWS)
    first_row_block = global_minibatch * minibatch_row_blocks
    end_row_block = cutlass.min(
        first_row_block + minibatch_row_blocks,
        _ceil_div_i32(num_tokens, Int32(MLP_TILE_ROWS)),
    )
    column_blocks = Int32(INTERMEDIATE_SIZE // _GATE_LOGICAL_N)
    valid = Int32(0)
    expert_index = Int32(-1)
    global_row_block = Int32(0)
    column_block = Int32(0)
    covered_end_row_block = cutlass.min(
        end_row_block,
        expert_row_block_offsets[Int32(NUM_LOCAL_EXPERTS)],
    )
    candidate_row_block = first_row_block + task_index // column_blocks
    if candidate_row_block < covered_end_row_block:
        expert_index = _expert_for_routed_row_block_i32(
            expert_row_block_offsets,
            candidate_row_block,
        )
        expert_first_row_block = expert_row_block_offsets[expert_index]
        expert_end_row_block = expert_row_block_offsets[
            expert_index + Int32(1)
        ]
        first = cutlass.max(first_row_block, expert_first_row_block)
        end = cutlass.min(covered_end_row_block, expert_end_row_block)
        row_blocks = end - first
        remaining_task = task_index - (
            first - first_row_block
        ) * column_blocks
        local_row_block, column_block = _swizzled_mlp_tile_coord_i32(
            row_blocks,
            column_blocks,
            remaining_task,
        )
        valid = Int32(1)
        global_row_block = first + local_row_block

    macrobatch_row_block = global_row_block - (
        macrobatch_index * Int32(macrobatch_size // MLP_TILE_ROWS)
    )
    # The fused epilogue publishes Hidden directly.  The legacy Gate/Up-tile
    # counter remains in the ABI but is intentionally not part of this edge.
    shared_row_blocks = Int32(shared_gate_up_tasks) // column_blocks
    ready_index = shared_row_blocks + global_row_block
    minibatch_first_row = global_minibatch * Int32(minibatch_size)
    minibatch_rows = cutlass.min(
        Int32(minibatch_size),
        num_tokens - minibatch_first_row,
    )
    dispatch_required = (
        _ceil_div_i32(minibatch_rows, Int32(DISPATCH_TILE_ROWS))
        * Int32(HIDDEN_SIZE // DISPATCH_TILE_COLUMNS)
    )
    return (
        valid,
        expert_index,
        macrobatch_row_block,
        column_block,
        ready_index,
        global_minibatch,
        dispatch_required,
    )


@cute.jit
def _routed_down_tile_i32(
    expert_row_block_offsets: cute.Pointer,
    num_tokens: Int32,
    macrobatch_index: Int32,
    minibatch_index: Int32,
    task_index: Int32,
    shared_row_blocks: cutlass.Constexpr,
    macrobatch_size: cutlass.Constexpr,
    minibatch_size: cutlass.Constexpr,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Decode CUDA's routed Down tile and its two readiness edges."""

    global_minibatch = (
        macrobatch_index * Int32(macrobatch_size // minibatch_size)
        + minibatch_index
    )
    minibatch_row_blocks = Int32(minibatch_size // MLP_TILE_ROWS)
    first_row_block = global_minibatch * minibatch_row_blocks
    end_row_block = cutlass.min(
        first_row_block + minibatch_row_blocks,
        _ceil_div_i32(num_tokens, Int32(MLP_TILE_ROWS)),
    )
    column_blocks = Int32(HIDDEN_SIZE // MLP_TILE_COLUMNS)
    valid = Int32(0)
    expert_index = Int32(-1)
    global_row_block = Int32(0)
    column_block = Int32(0)
    covered_end_row_block = cutlass.min(
        end_row_block,
        expert_row_block_offsets[Int32(NUM_LOCAL_EXPERTS)],
    )
    candidate_row_block = first_row_block + task_index // column_blocks
    if candidate_row_block < covered_end_row_block:
        expert_index = _expert_for_routed_row_block_i32(
            expert_row_block_offsets,
            candidate_row_block,
        )
        expert_first_row_block = expert_row_block_offsets[expert_index]
        expert_end_row_block = expert_row_block_offsets[
            expert_index + Int32(1)
        ]
        first = cutlass.max(first_row_block, expert_first_row_block)
        end = cutlass.min(covered_end_row_block, expert_end_row_block)
        row_blocks = end - first
        remaining_task = task_index - (
            first - first_row_block
        ) * column_blocks
        local_row_block, column_block = _swizzled_mlp_tile_coord_i32(
            row_blocks,
            column_blocks,
            remaining_task,
        )
        valid = Int32(1)
        global_row_block = first + local_row_block

    macrobatch_row_block = global_row_block - (
        macrobatch_index * Int32(macrobatch_size // MLP_TILE_ROWS)
    )
    hidden_ready_index = Int32(shared_row_blocks) + global_row_block
    hidden_ready_required = Int32(
        (MLP_TILE_ROWS // SWIGLU_TILE_ROWS)
        * (INTERMEDIATE_SIZE // SWIGLU_TILE_COLUMNS)
    )
    return (
        valid,
        expert_index,
        macrobatch_row_block,
        column_block,
        hidden_ready_index,
        hidden_ready_required,
        global_minibatch,
    )


class _PersistentBf16Mega:
    """One-launch BF16 communication + fused-FC1 + Down mega-kernel."""

    def __init__(self) -> None:
        # V1-C2 names the cluster-2 GEMM collective, not CLC depth.  Its
        # authoritative operand mapping gives every CTA different M128 rows:
        # every CTA owns different M128 activation rows; rank 0 supplies Gate
        # N128 and rank 1 supplies Up N128 to one 2CTA N256 instruction.
        self.collective = _Fc1Collective()

    @cute.jit
    def __call__(
        self,
        x_peer_ptrs: list[cute.Pointer],
        combine_peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        x_shared: cute.Tensor,
        x_routed: cute.Tensor,
        shared_gate_weights: cute.Tensor,
        routed_gate_weights: cute.Tensor,
        shared_up_weights: cute.Tensor,
        routed_up_weights: cute.Tensor,
        shared_down_weights: cute.Tensor,
        routed_down_weights: cute.Tensor,
        gate_shared: cute.Tensor,
        gate_routed: cute.Tensor,
        up_shared: cute.Tensor,
        up_routed: cute.Tensor,
        hidden_shared: cute.Tensor,
        hidden_routed: cute.Tensor,
        y_shared: cute.Tensor,
        y_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        num_local_tokens: cutlass.Constexpr,
        schedule_capacity: cutlass.Constexpr,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
        swiglu_limit: cutlass.Constexpr,
        is_clamped: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        # These are the exact CLC1/G1D1 task counts in forward.cuh.  A CLC
        # item is one fused M256xN128 FC1 tile or one M256xN256 Down tile.
        shared_row_blocks = num_local_tokens // MLP_TILE_ROWS
        minibatch_row_blocks = minibatch_size // MLP_TILE_ROWS
        shared_gate_up_raw_tasks = (
            shared_row_blocks * (INTERMEDIATE_SIZE // _GATE_LOGICAL_N)
        )
        shared_down_raw_tasks = (
            shared_row_blocks * (HIDDEN_SIZE // MLP_TILE_COLUMNS)
        )
        minibatch_gate_up_raw_tasks = (
            minibatch_row_blocks * (INTERMEDIATE_SIZE // _GATE_LOGICAL_N)
        )
        minibatch_down_raw_tasks = (
            minibatch_row_blocks * (HIDDEN_SIZE // MLP_TILE_COLUMNS)
        )
        shared_gate_up_tasks = (
            shared_gate_up_raw_tasks + FUSED_GATE_UP_TASK_GROUP_SIZE - 1
        ) // FUSED_GATE_UP_TASK_GROUP_SIZE
        shared_down_tasks = (
            shared_down_raw_tasks + FUSED_DOWN_TASK_GROUP_SIZE - 1
        ) // FUSED_DOWN_TASK_GROUP_SIZE
        minibatch_gate_up_tasks = (
            minibatch_gate_up_raw_tasks + FUSED_GATE_UP_TASK_GROUP_SIZE - 1
        ) // FUSED_GATE_UP_TASK_GROUP_SIZE
        minibatch_down_tasks = (
            minibatch_down_raw_tasks + FUSED_DOWN_TASK_GROUP_SIZE - 1
        ) // FUSED_DOWN_TASK_GROUP_SIZE
        shared_tasks = shared_gate_up_tasks + shared_down_tasks
        minibatch_tasks = minibatch_gate_up_tasks + minibatch_down_tasks
        comm_clusters = num_comm_sms // CLUSTER_SIZE
        capacity_minibatches = (
            schedule_capacity + minibatch_size - 1
        ) // minibatch_size
        capacity_clusters = (
            comm_clusters
            + shared_tasks
            + capacity_minibatches * minibatch_tasks
        )
        collective = self.collective
        routed_gate_weights_0 = routed_gate_weights[None, None, 0]
        collective.configure(x_routed, routed_gate_weights_0, gate_routed)
        a_smem_layout = cute.slice_(
            collective.a_smem_layout_staged,
            (None, None, None, 0),
        )
        b_smem_layout = cute.slice_(
            collective.b_smem_layout_staged,
            (None, None, None, 0),
        )
        a_op = cpasync.CopyBulkTensorTileG2SOp(collective.cta_group)
        b_op = cpasync.CopyBulkTensorTileG2SOp(collective.cta_group)
        (
            tma_atom_a_shared_gate_up,
            tma_tensor_a_shared_gate_up,
        ) = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            x_shared,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_shared_gate,
            tma_tensor_b_shared_gate,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            shared_gate_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_shared_up,
            tma_tensor_b_shared_up,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            shared_up_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_a_shared_down,
            tma_tensor_a_shared_down,
        ) = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            hidden_shared,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_shared_down,
            tma_tensor_b_shared_down,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            shared_down_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_a_routed_gate_up,
            tma_tensor_a_routed_gate_up,
        ) = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            x_routed,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_routed_gate,
            tma_tensor_b_routed_gate,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            routed_gate_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_routed_up,
            tma_tensor_b_routed_up,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            routed_up_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_a_routed_down,
            tma_tensor_a_routed_down,
        ) = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            hidden_routed,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_routed_down,
            tma_tensor_b_routed_down,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            routed_down_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        tma_atom_d_shared_gate, tma_tensor_d_shared_gate = (
            collective._make_tma_epi_atoms_and_tensors(
                gate_shared,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        tma_atom_d_shared_up, tma_tensor_d_shared_up = (
            collective._make_tma_epi_atoms_and_tensors(
                up_shared,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        tma_atom_d_shared_hidden, tma_tensor_d_shared_hidden = (
            collective._make_tma_epi_atoms_and_tensors(
                hidden_shared,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        tma_atom_d_shared_down, tma_tensor_d_shared_down = (
            collective._make_tma_epi_atoms_and_tensors(
                y_shared,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        tma_atom_d_routed_gate, tma_tensor_d_routed_gate = (
            collective._make_tma_epi_atoms_and_tensors(
                gate_routed,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        tma_atom_d_routed_up, tma_tensor_d_routed_up = (
            collective._make_tma_epi_atoms_and_tensors(
                up_routed,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        tma_atom_d_routed_hidden, tma_tensor_d_routed_hidden = (
            collective._make_tma_epi_atoms_and_tensors(
                hidden_routed,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        tma_atom_d_routed_down, tma_tensor_d_routed_down = (
            collective._make_tma_epi_atoms_and_tensors(
                y_routed,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        collective.num_tma_load_bytes = (
            cute.size_in_bytes(BFloat16, a_smem_layout)
            + cute.size_in_bytes(BFloat16, b_smem_layout)
        ) * cute.size(collective.tiled_mma.thr_id.shape)
        scheduler_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (capacity_clusters * CLUSTER_SIZE, 1, 1),
            CLUSTER_SHAPE,
        )
        self.kernel(
            collective.tiled_mma,
            tma_atom_a_shared_gate_up,
            tma_tensor_a_shared_gate_up,
            tma_atom_b_shared_gate,
            tma_tensor_b_shared_gate,
            tma_atom_b_shared_up,
            tma_tensor_b_shared_up,
            tma_atom_a_shared_down,
            tma_tensor_a_shared_down,
            tma_atom_b_shared_down,
            tma_tensor_b_shared_down,
            tma_atom_d_shared_gate,
            tma_tensor_d_shared_gate,
            tma_atom_d_shared_up,
            tma_tensor_d_shared_up,
            tma_atom_d_shared_hidden,
            tma_tensor_d_shared_hidden,
            tma_atom_d_shared_down,
            tma_tensor_d_shared_down,
            tma_atom_a_routed_gate_up,
            tma_tensor_a_routed_gate_up,
            tma_atom_b_routed_gate,
            tma_tensor_b_routed_gate,
            tma_atom_b_routed_up,
            tma_tensor_b_routed_up,
            tma_atom_a_routed_down,
            tma_tensor_a_routed_down,
            tma_atom_b_routed_down,
            tma_tensor_b_routed_down,
            tma_atom_d_routed_gate,
            tma_tensor_d_routed_gate,
            tma_atom_d_routed_up,
            tma_tensor_d_routed_up,
            tma_atom_d_routed_hidden,
            tma_tensor_d_routed_hidden,
            tma_atom_d_routed_down,
            tma_tensor_d_routed_down,
            collective.cluster_layout_vmnk,
            collective.a_smem_layout_staged,
            collective.b_smem_layout_staged,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            x_peer_ptrs,
            combine_peer_ptrs,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            x_routed,
            gate_up_tile_ready,
            hidden_row_block_ready,
            y_routed,
            x_routed_ready,
            y_routed_ready,
            y_routed_done,
            scheduler_params,
            comm_clusters,
            shared_gate_up_tasks,
            shared_down_tasks,
            shared_tasks,
            shared_row_blocks,
            minibatch_gate_up_tasks,
            minibatch_down_tasks,
            minibatch_tasks,
            num_local_tokens,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            is_clamped,
        ).launch(
            grid=(capacity_clusters * CLUSTER_SIZE, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=_PersistentSharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _dispatch_task(
        self,
        peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        x_routed: cute.Tensor,
        y_routed_ready: cute.Tensor,
        x_routed_ready: cute.Tensor,
        arena: cute.Pointer,
        mbarrier: cute.Pointer,
        phase: Int32,
        num_tokens: Int32,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        macrobatch_index: Int32,
        task_index: Int32,
        previous_macrobatch_index: Int32,
    ) -> Int32:
        thread = cute.arch.thread_idx()[0]
        is_worker = thread < Int32(DISPATCH_TILE_ROWS)
        column_blocks = HIDDEN_SIZE // DISPATCH_TILE_COLUMNS
        row_tile = task_index // Int32(column_blocks)
        column_tile = task_index - row_tile * Int32(column_blocks)
        row_start = row_tile * Int32(DISPATCH_TILE_ROWS)
        local_row = row_start + thread
        macro_offset = macrobatch_index * Int32(macrobatch_size)
        global_row = macro_offset + local_row
        row_chunk = arena + thread * Int32(DISPATCH_ROW_CHUNK_BYTES)

        if thread == Int32(0):
            if previous_macrobatch_index >= Int32(0):
                previous_offset = previous_macrobatch_index * Int32(macrobatch_size)
                previous_rows = cutlass.min(
                    Int32(macrobatch_size),
                    num_tokens - previous_offset,
                )
                if row_start < previous_rows:
                    global_minibatch = (
                        previous_offset + row_start
                    ) // Int32(minibatch_size)
                    minibatch_rows = cutlass.min(
                        Int32(minibatch_size),
                        num_tokens - global_minibatch * Int32(minibatch_size),
                    )
                    required = (
                        _ceil_div_i32(minibatch_rows, Int32(MLP_TILE_ROWS))
                        * Int32(HIDDEN_SIZE // MLP_TILE_COLUMNS)
                        * Int32(CLUSTER_SIZE)
                    )
                    _counter_wait_gpu(y_routed_ready, global_minibatch, required)
        cute.arch.sync_threads()

        peer_rank = Int32(-1)
        route_index = Int32(-1)
        if is_worker:
            peer_rank = schedule_peer_rank[global_row]
            if peer_rank >= Int32(0):
                route_index = schedule_peer_token_idx[global_row]

        # Reuse the first 128 Int32 words of the arena to count valid peer
        # transactions.  TMA overwrites them only after thread 0 has reduced
        # the flags and armed the CTA-wide barrier.
        valid_flags = cute.make_tensor(
            cute.recast_ptr(arena, dtype=Int32),
            cute.make_layout((DISPATCH_TILE_ROWS,)),
        )
        if is_worker:
            valid_flags[thread] = Int32(peer_rank >= Int32(0))
        cute.arch.sync_threads()
        if thread == Int32(0):
            valid = Int32(0)
            for worker in cutlass.range_constexpr(DISPATCH_TILE_ROWS):
                valid = valid + valid_flags[worker]
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbarrier,
                valid * Int32(DISPATCH_ROW_CHUNK_BYTES),
            )
        cute.arch.sync_threads()
        # valid_flags used the generic proxy at the same addresses that the
        # following G2S TMA loads overwrite through the async proxy.  End that
        # scratch lifetime in both proxies before any worker can issue TMA.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        if is_worker:
            if peer_rank >= Int32(0):
                source_token = route_index // Int32(TOPK)
                source = (
                    _select_peer_address(peer_ptrs, peer_rank)
                    + Int64(source_token) * Int64(HIDDEN_SIZE * 2)
                    + Int64(column_tile * Int32(DISPATCH_ROW_CHUNK_BYTES))
                )
                tma_load_1d_raw(
                    row_chunk,
                    source,
                    mbarrier,
                    Int32(DISPATCH_ROW_CHUNK_BYTES),
                )
            else:
                zero_words = cute.make_tensor(
                    cute.recast_ptr(row_chunk, dtype=Uint32),
                    cute.make_layout((DISPATCH_ROW_CHUNK_BYTES // 4,)),
                )
                for word in cutlass.range_constexpr(DISPATCH_ROW_CHUNK_BYTES // 4):
                    zero_words[word] = Uint32(0)
                # Generic-proxy zero fills must be ordered before the async
                # proxy reads the same row for the following TMA store.
                cute.arch.fence_proxy("async.shared", space="cta")

        cute.arch.mbarrier_wait(mbarrier, phase)
        phase = phase ^ Int32(1)
        if is_worker:
            destination = (
                x_routed.iterator.toint()
                + Int64(local_row) * Int64(HIDDEN_SIZE * 2)
                + Int64(column_tile * Int32(DISPATCH_ROW_CHUNK_BYTES))
            )
            tma_store_1d_raw(
                destination,
                row_chunk,
                Int32(DISPATCH_ROW_CHUNK_BYTES),
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.sync_threads()
        if thread == Int32(0):
            global_minibatch = (macro_offset + row_start) // Int32(minibatch_size)
            _counter_arrive_gpu(x_routed_ready, global_minibatch)
        cute.arch.sync_threads()
        return phase

    @cute.jit
    def _combine_task(
        self,
        peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        y_routed: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        arena: cute.Pointer,
        mbarriers: cute.Pointer,
        phase_bits: Int32,
        num_tokens: Int32,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        macrobatch_index: Int32,
        task_index: Int32,
    ) -> Int32:
        thread = cute.arch.thread_idx()[0]
        is_worker = thread < Int32(COMBINE_TILE_ROWS)
        column_blocks = HIDDEN_SIZE // COMBINE_TILE_COLUMNS
        first_tile = task_index * Int32(COMBINE_PIPE_DEPTH)
        macro_offset = macrobatch_index * Int32(macrobatch_size)
        macro_rows = cutlass.min(
            Int32(macrobatch_size),
            num_tokens - macro_offset,
        )
        total_tiles = (
            macro_rows // Int32(COMBINE_TILE_ROWS) * Int32(column_blocks)
        )
        valid_stages = cutlass.min(
            Int32(COMBINE_PIPE_DEPTH),
            total_tiles - first_tile,
        )

        if thread == Int32(0):
            first_row = (
                first_tile // Int32(column_blocks) * Int32(COMBINE_TILE_ROWS)
            )
            last_tile = first_tile + valid_stages - Int32(1)
            last_row = (
                last_tile // Int32(column_blocks) * Int32(COMBINE_TILE_ROWS)
            )
            first_minibatch = (macro_offset + first_row) // Int32(minibatch_size)
            last_minibatch = (macro_offset + last_row) // Int32(minibatch_size)
            minibatch = first_minibatch
            while minibatch <= last_minibatch:
                minibatch_rows = cutlass.min(
                    Int32(minibatch_size),
                    num_tokens - minibatch * Int32(minibatch_size),
                )
                required = (
                    _ceil_div_i32(minibatch_rows, Int32(MLP_TILE_ROWS))
                    * Int32(HIDDEN_SIZE // MLP_TILE_COLUMNS)
                    * Int32(CLUSTER_SIZE)
                )
                _counter_wait_gpu(y_routed_ready, minibatch, required)
                minibatch = minibatch + Int32(1)
        cute.arch.sync_threads()

        valid_flags = cute.make_tensor(
            cute.recast_ptr(arena, dtype=Int32),
            cute.make_layout((COMBINE_PIPE_DEPTH, COMBINE_TILE_ROWS)),
        )
        peer_ranks = [Int32(-1)] * COMBINE_PIPE_DEPTH
        route_indices = [Int32(-1)] * COMBINE_PIPE_DEPTH
        local_rows = [Int32(0)] * COMBINE_PIPE_DEPTH
        column_tiles = [Int32(0)] * COMBINE_PIPE_DEPTH

        for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
            tile = first_tile + Int32(stage)
            row_tile = tile // Int32(column_blocks)
            column_tile = tile - row_tile * Int32(column_blocks)
            local_row = row_tile * Int32(COMBINE_TILE_ROWS) + thread
            local_rows[stage] = local_row
            column_tiles[stage] = column_tile
            if is_worker:
                if Int32(stage) < valid_stages:
                    peer_ranks[stage] = schedule_peer_rank[macro_offset + local_row]
                    if peer_ranks[stage] >= Int32(0):
                        route_indices[stage] = schedule_peer_token_idx[
                            macro_offset + local_row
                        ]
                    valid_flags[stage, thread] = Int32(
                        peer_ranks[stage] >= Int32(0)
                    )
                else:
                    valid_flags[stage, thread] = Int32(0)
        cute.arch.sync_threads()

        # Reduce every stage before any TMA load can overwrite the arena that
        # temporarily holds valid_flags.  Only thread 0 owns these registers.
        valid_counts = [Int32(0)] * COMBINE_PIPE_DEPTH
        if thread == Int32(0):
            for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                if Int32(stage) < valid_stages:
                    for worker in cutlass.range_constexpr(COMBINE_TILE_ROWS):
                        valid_counts[stage] = (
                            valid_counts[stage] + valid_flags[stage, worker]
                        )
        cute.arch.sync_threads()
        # All seven generic-proxy reductions are complete.  Order that retired
        # scratch lifetime against the first async-proxy G2S overwrite, then
        # rendezvous so no worker can issue TMA early.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
            if Int32(stage) < valid_stages:
                mbarrier = mbarriers + Int32(stage)
                if thread == Int32(0):
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbarrier,
                        valid_counts[stage] * Int32(COMBINE_ROW_CHUNK_BYTES),
                    )
                cute.arch.sync_threads()
                if is_worker:
                    if peer_ranks[stage] >= Int32(0):
                        row_chunk = (
                            arena
                            + Int32(
                                stage
                                * COMBINE_TILE_ROWS
                                * COMBINE_ROW_CHUNK_BYTES
                            )
                            + thread * Int32(COMBINE_ROW_CHUNK_BYTES)
                        )
                        source = (
                            y_routed.iterator.toint()
                            + Int64(local_rows[stage]) * Int64(HIDDEN_SIZE * 2)
                            + Int64(
                                column_tiles[stage]
                                * Int32(COMBINE_ROW_CHUNK_BYTES)
                            )
                        )
                        tma_load_1d_raw(
                            row_chunk,
                            source,
                            mbarrier,
                            Int32(COMBINE_ROW_CHUNK_BYTES),
                        )

        for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
            if Int32(stage) < valid_stages:
                phase = (phase_bits >> Int32(stage)) & Int32(1)
                cute.arch.mbarrier_wait(mbarriers + Int32(stage), phase)
                phase_bits = phase_bits ^ Int32(1 << stage)
                if is_worker:
                    if peer_ranks[stage] >= Int32(0):
                        row_chunk = (
                            arena
                            + Int32(
                                stage
                                * COMBINE_TILE_ROWS
                                * COMBINE_ROW_CHUNK_BYTES
                            )
                            + thread * Int32(COMBINE_ROW_CHUNK_BYTES)
                        )
                        destination = (
                            _select_peer_address(peer_ptrs, peer_ranks[stage])
                            + Int64(route_indices[stage]) * Int64(HIDDEN_SIZE * 2)
                            + Int64(
                                column_tiles[stage]
                                * Int32(COMBINE_ROW_CHUNK_BYTES)
                            )
                        )
                        tma_store_1d_raw(
                            destination,
                            row_chunk,
                            Int32(COMBINE_ROW_CHUNK_BYTES),
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                # CUDA maps one pipeline stage to one warp leader for the
                # row-granular y_routed_done producer.
                if cute.arch.warp_idx() == Int32(stage):
                    if cute.arch.lane_idx() == Int32(0):
                        if macrobatch_index > Int32(0):
                            row_start = (
                                (first_tile + Int32(stage)) // Int32(column_blocks)
                                * Int32(COMBINE_TILE_ROWS)
                            )
                            done_index = (
                                macro_offset + row_start
                            ) // Int32(MLP_TILE_ROWS // CLUSTER_SIZE)
                            _counter_arrive_gpu(y_routed_done, done_index)
        cute.arch.sync_threads()
        return phase_bits

    @cute.jit
    def _run_comm_prefix(
        self,
        x_peer_ptrs: list[cute.Pointer],
        combine_peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        x_routed: cute.Tensor,
        y_routed: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        dispatch_mbarrier: cute.Pointer,
        combine_mbarriers: cute.Pointer,
        arena: cute.Pointer,
        cluster_index: Int32,
        cta_rank: Int32,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
    ) -> None:
        thread = cute.arch.thread_idx()[0]
        if thread == Int32(0):
            cute.arch.mbarrier_init(dispatch_mbarrier, 1)
            for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                cute.arch.mbarrier_init(
                    combine_mbarriers + Int32(stage),
                    1,
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        num_tokens = num_tokens_tensor[0]
        num_macrobatches = _ceil_div_i32(
            num_tokens,
            Int32(macrobatch_size),
        )
        comm_cta = cluster_index * Int32(CLUSTER_SIZE) + cta_rank
        last_macro = num_macrobatches - Int32(1)
        dispatch_phase = Int32(0)
        combine_phase_bits = Int32(0)

        macro_rows = Int32(0)
        if num_macrobatches > Int32(0):
            macro_rows = cutlass.min(
                Int32(macrobatch_size),
                num_tokens - last_macro * Int32(macrobatch_size),
            )
        dispatch_tasks = (
            macro_rows // Int32(DISPATCH_TILE_ROWS)
            * Int32(HIDDEN_SIZE // DISPATCH_TILE_COLUMNS)
        )
        task = comm_cta
        while task < dispatch_tasks:
            dispatch_phase = self._dispatch_task(
                x_peer_ptrs,
                schedule_peer_rank,
                schedule_peer_token_idx,
                x_routed,
                y_routed_ready,
                x_routed_ready,
                arena,
                dispatch_mbarrier,
                dispatch_phase,
                num_tokens,
                macrobatch_size,
                minibatch_size,
                last_macro,
                task,
                Int32(-1),
            )
            task = task + Int32(num_comm_sms)

        macro = last_macro
        while macro >= Int32(0):
            macro_rows = cutlass.min(
                Int32(macrobatch_size),
                num_tokens - macro * Int32(macrobatch_size),
            )
            combine_tiles = (
                macro_rows // Int32(COMBINE_TILE_ROWS)
                * Int32(HIDDEN_SIZE // COMBINE_TILE_COLUMNS)
            )
            combine_tasks = _ceil_div_i32(
                combine_tiles,
                Int32(COMBINE_PIPE_DEPTH),
            )
            previous_macro = macro - Int32(1)
            previous_dispatch_tasks = Int32(0)
            if previous_macro >= Int32(0):
                previous_rows = cutlass.min(
                    Int32(macrobatch_size),
                    num_tokens - previous_macro * Int32(macrobatch_size),
                )
                previous_dispatch_tasks = (
                    previous_rows // Int32(DISPATCH_TILE_ROWS)
                    * Int32(HIDDEN_SIZE // DISPATCH_TILE_COLUMNS)
                )

            task = comm_cta
            task_limit = cutlass.max(combine_tasks, previous_dispatch_tasks)
            while task < task_limit:
                if task < combine_tasks:
                    combine_phase_bits = self._combine_task(
                        combine_peer_ptrs,
                        schedule_peer_rank,
                        schedule_peer_token_idx,
                        y_routed,
                        y_routed_ready,
                        y_routed_done,
                        arena,
                        combine_mbarriers,
                        combine_phase_bits,
                        num_tokens,
                        macrobatch_size,
                        minibatch_size,
                        macro,
                        task,
                    )
                if task < previous_dispatch_tasks:
                    dispatch_phase = self._dispatch_task(
                        x_peer_ptrs,
                        schedule_peer_rank,
                        schedule_peer_token_idx,
                        x_routed,
                        y_routed_ready,
                        x_routed_ready,
                        arena,
                        dispatch_mbarrier,
                        dispatch_phase,
                        num_tokens,
                        macrobatch_size,
                        minibatch_size,
                        previous_macro,
                        task,
                        macro,
                    )
                task = task + Int32(num_comm_sms)
            macro = macro - Int32(1)

    @cute.jit
    def _decode_compute_task(
        self,
        compute_cluster: Int32,
        num_tokens: Int32,
        shared_gate_up_tasks: cutlass.Constexpr,
        shared_down_tasks: cutlass.Constexpr,
        shared_tasks: cutlass.Constexpr,
        minibatch_gate_up_tasks: cutlass.Constexpr,
        minibatch_down_tasks: cutlass.Constexpr,
        minibatch_tasks: cutlass.Constexpr,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
    ) -> tuple[Int32, Int32, Int32, Int32]:
        """Decode the fused four-segment CUDA ladder with one staged return.

        Kinds are ``0=shared FC1``, ``1=shared Down``, ``2=routed FC1`` and
        ``3=routed Down``.  The two SwiGLU segments from the historical CuTe
        bring-up no longer exist; SwiGLU is an FC1 epilogue operation.
        """

        kind = Int32(0)
        task = compute_cluster
        macrobatch = Int32(-1)
        minibatch = Int32(-1)
        if compute_cluster >= Int32(shared_gate_up_tasks):
            kind = Int32(1)
            task = compute_cluster - Int32(shared_gate_up_tasks)
            if compute_cluster >= Int32(shared_tasks):
                routed = compute_cluster - Int32(shared_tasks)
                ordered_minibatch = routed // Int32(minibatch_tasks)
                minibatch_task = routed - ordered_minibatch * Int32(
                    minibatch_tasks
                )
                num_macrobatches = _ceil_div_i32(
                    num_tokens, Int32(macrobatch_size)
                )
                true_minibatches = _ceil_div_i32(
                    num_tokens, Int32(minibatch_size)
                )
                minibatches_per_macro = Int32(
                    macrobatch_size // minibatch_size
                )
                last_macro_minibatches = true_minibatches - (
                    num_macrobatches - Int32(1)
                ) * minibatches_per_macro
                if ordered_minibatch < last_macro_minibatches:
                    macrobatch = num_macrobatches - Int32(1)
                    minibatch = ordered_minibatch
                else:
                    index = ordered_minibatch - last_macro_minibatches
                    macrobatch = (
                        num_macrobatches
                        - Int32(2)
                        - index // minibatches_per_macro
                    )
                    minibatch = index % minibatches_per_macro

                kind = Int32(2)
                task = minibatch_task
                if task >= Int32(minibatch_gate_up_tasks):
                    task = task - Int32(minibatch_gate_up_tasks)
                    kind = Int32(3)
        return kind, task, macrobatch, minibatch

    @cute.jit
    def _decode_routed_endpoint_gemm_tile(
        self,
        kind: Int32,
        task: Int32,
        macrobatch: Int32,
        minibatch: Int32,
        expert_row_block_offsets: cute.Pointer,
        num_tokens: Int32,
        shared_gate_up_tasks: cutlass.Constexpr,
        shared_row_blocks: cutlass.Constexpr,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
    ) -> tuple[
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
    ]:
        """Decode a shared/routed fused-FC1 or Down tile with one return."""

        valid_tile = Int32(0)
        expert_index = Int32(-1)
        tile_coord_m = Int32(0)
        tile_coord_n = Int32(0)
        input_ready_index = Int32(0)
        input_ready_required = Int32(0)
        output_ready_index = Int32(0)
        k_tile_count = Int32(0)
        if kind == Int32(0):
            tile_coord_m, tile_coord_n = _swizzled_mlp_tile_coord_i32(
                Int32(shared_row_blocks),
                Int32(INTERMEDIATE_SIZE // _GATE_LOGICAL_N),
                task,
            )
            valid_tile = Int32(1)
            expert_index = Int32(0)
            output_ready_index = tile_coord_m
            k_tile_count = Int32(HIDDEN_SIZE // _GATE_TILE_K)
        elif kind == Int32(1):
            tile_coord_m, tile_coord_n = _swizzled_mlp_tile_coord_i32(
                Int32(shared_row_blocks),
                Int32(HIDDEN_SIZE // MLP_TILE_COLUMNS),
                task,
            )
            valid_tile = Int32(1)
            expert_index = Int32(0)
            input_ready_index = tile_coord_m
            input_ready_required = Int32(
                (MLP_TILE_ROWS // SWIGLU_TILE_ROWS)
                * (INTERMEDIATE_SIZE // SWIGLU_TILE_COLUMNS)
            )
            k_tile_count = Int32(INTERMEDIATE_SIZE // _GATE_TILE_K)
        elif kind == Int32(2):
            (
                valid_tile,
                expert_index,
                tile_coord_m,
                tile_coord_n,
                output_ready_index,
                input_ready_index,
                input_ready_required,
            ) = _routed_gate_up_tile_i32(
                expert_row_block_offsets,
                num_tokens,
                macrobatch,
                minibatch,
                task,
                shared_gate_up_tasks,
                macrobatch_size,
                minibatch_size,
            )
            k_tile_count = Int32(HIDDEN_SIZE // _GATE_TILE_K)
        elif kind == Int32(3):
            (
                valid_tile,
                expert_index,
                tile_coord_m,
                tile_coord_n,
                input_ready_index,
                input_ready_required,
                output_ready_index,
            ) = _routed_down_tile_i32(
                expert_row_block_offsets,
                num_tokens,
                macrobatch,
                minibatch,
                task,
                shared_row_blocks,
                macrobatch_size,
                minibatch_size,
            )
            k_tile_count = Int32(INTERMEDIATE_SIZE // _GATE_TILE_K)
        return (
            valid_tile,
            expert_index,
            tile_coord_m,
            tile_coord_n,
            input_ready_index,
            input_ready_required,
            output_ready_index,
            k_tile_count,
        )

    @cute.jit
    def _sync_after_routed_swiglu_transition(
        self,
        current_kind: Int32,
        next_kind: Int32,
        next_active: Int32,
    ) -> None:
        """Fused FC1 and Down are both cluster-cooperative; no transition."""

        del current_kind, next_kind, next_active

    @cute.jit
    def _drain_clc_capacity_tail(
        self,
        needs_drain: Int32,
        clc_drain_pipeline,
        drain_consumer_state,
        drain_producer_state,
        drain_scheduler,
        cta_rank: Int32,
    ) -> None:
        """Drain capacity-only CLC responses after every role finishes."""

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        drain_active = needs_drain
        while drain_active != Int32(0):
            if cta_rank == Int32(0):
                clc_drain_pipeline.producer_acquire(drain_producer_state)
                drain_scheduler.advance_to_next_work(
                    clc_drain_pipeline.producer_get_barrier(
                        drain_producer_state
                    )
                )
            cute.arch.sync_warp()

            if cute.arch.lane_idx() == Int32(0):
                clc_drain_pipeline.consumer_wait(drain_consumer_state)
            cute.arch.sync_warp()
            drain_work = drain_scheduler.get_current_work()
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == Int32(0):
                clc_drain_pipeline.consumer_release(drain_consumer_state)

            for _ in cutlass.range_constexpr(CLC_DRAIN_WARPS):
                drain_consumer_state.advance()
                drain_producer_state.advance()
            drain_active = Int32(0)
            if drain_work.is_valid_tile:
                drain_active = Int32(1)
            elif cta_rank == Int32(0):
                # Reclaim this warp-owned stage's terminal response before the
                # producer exits.  Each drain warp advances by the full depth,
                # so its phase is independent of the other seven stages and a
                # generic producer_tail() cannot represent this lifecycle.
                clc_drain_pipeline.sync_object_empty.wait(
                    drain_producer_state.index,
                    drain_producer_state.phase,
                )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a_shared_gate_up: cute.CopyAtom,
        x_shared_tma: cute.Tensor,
        tma_atom_b_shared_gate: cute.CopyAtom,
        shared_gate_weights: cute.Tensor,
        tma_atom_b_shared_up: cute.CopyAtom,
        shared_up_weights: cute.Tensor,
        tma_atom_a_shared_down: cute.CopyAtom,
        hidden_shared_tma: cute.Tensor,
        tma_atom_b_shared_down: cute.CopyAtom,
        shared_down_weights: cute.Tensor,
        tma_atom_d_shared_gate: cute.CopyAtom,
        gate_shared: cute.Tensor,
        tma_atom_d_shared_up: cute.CopyAtom,
        up_shared: cute.Tensor,
        tma_atom_d_shared_hidden: cute.CopyAtom,
        hidden_shared: cute.Tensor,
        tma_atom_d_shared_down: cute.CopyAtom,
        y_shared_tma: cute.Tensor,
        tma_atom_a_routed_gate_up: cute.CopyAtom,
        x_routed_tma: cute.Tensor,
        tma_atom_b_routed_gate: cute.CopyAtom,
        routed_gate_weights: cute.Tensor,
        tma_atom_b_routed_up: cute.CopyAtom,
        routed_up_weights: cute.Tensor,
        tma_atom_a_routed_down: cute.CopyAtom,
        hidden_routed_tma: cute.Tensor,
        tma_atom_b_routed_down: cute.CopyAtom,
        routed_down_weights: cute.Tensor,
        tma_atom_d_routed_gate: cute.CopyAtom,
        gate_routed: cute.Tensor,
        tma_atom_d_routed_up: cute.CopyAtom,
        up_routed: cute.Tensor,
        tma_atom_d_routed_hidden: cute.CopyAtom,
        hidden_routed: cute.Tensor,
        tma_atom_d_routed_down: cute.CopyAtom,
        y_routed_tma: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        x_peer_ptrs: list[cute.Pointer],
        combine_peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        x_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        y_routed: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        scheduler_params: utils.ClcDynamicPersistentTileSchedulerParams,
        comm_clusters: cutlass.Constexpr,
        shared_gate_up_tasks: cutlass.Constexpr,
        shared_down_tasks: cutlass.Constexpr,
        shared_tasks: cutlass.Constexpr,
        shared_row_blocks: cutlass.Constexpr,
        minibatch_gate_up_tasks: cutlass.Constexpr,
        minibatch_down_tasks: cutlass.Constexpr,
        minibatch_tasks: cutlass.Constexpr,
        num_local_tokens: cutlass.Constexpr,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
        swiglu_limit: cutlass.Constexpr,
        is_clamped: cutlass.Constexpr,
    ):
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_index = cute.arch.make_warp_uniform(cute.arch.cluster_idx()[0])
        num_tokens = num_tokens_tensor[0]
        # The communication prefix and compute decoder both support multiple
        # macrobatches: communication advances macro indices in reverse while
        # routed compute advances minibatches forward inside that same order.
        # CuTe DSL staged conditionals cannot contain early returns, so capacity
        # clusters are masked below and reclaimed by the explicit CLC drain.
        true_minibatches = _ceil_div_i32(num_tokens, Int32(minibatch_size))
        true_clusters = (
            Int32(comm_clusters)
            + Int32(shared_tasks)
            + true_minibatches * Int32(minibatch_tasks)
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(_PersistentSharedStorage)
        # Keep the user-defined shared-storage object outside the dynamic role
        # region.  Materialize only the leaves used there before entering the
        # role predicate; this also keeps the dynamic lowering structurally
        # simple across the supported toolchain.
        dispatch_mbarrier = storage.dispatch_mbarrier.ptr
        combine_mbarriers = storage.combine_mbarriers.data_ptr()
        comm_arena = storage.arena.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        expert_row_block_offsets = (
            storage.expert_row_block_offsets.data_ptr()
        )
        if thread == Int32(0):
            row_block_offset = Int32(0)
            expert_row_block_offsets[Int32(0)] = row_block_offset
            for expert in cutlass.range_constexpr(NUM_LOCAL_EXPERTS):
                row_block_offset = row_block_offset + (
                    tokens_per_expert[expert] // Int32(MLP_TILE_ROWS)
                )
                expert_row_block_offsets[Int32(expert + 1)] = (
                    row_block_offset
                )
        cute.arch.sync_threads()
        if cluster_index < Int32(comm_clusters):
            self._run_comm_prefix(
                x_peer_ptrs,
                combine_peer_ptrs,
                schedule_peer_rank,
                schedule_peer_token_idx,
                num_tokens_tensor,
                x_routed,
                y_routed,
                x_routed_ready,
                y_routed_ready,
                y_routed_done,
                dispatch_mbarrier,
                combine_mbarriers,
                comm_arena,
                cluster_index,
                cta_rank,
                macrobatch_size,
                minibatch_size,
                num_comm_sms,
            )

        # Preserve the fixed communication prefix and capacity-tail behavior
        # without a staged early return.  Sentinel role 8 matches none of the
        # five compute role predicates below, so only true compute clusters
        # enter their persistent role loops.
        role_warp = Int32(WARPS_PER_CTA)
        if cluster_index >= Int32(comm_clusters):
            if cluster_index < true_clusters:
                role_warp = warp

        is_leader_cta = cta_rank == Int32(0)
        is_two_cta = cute.size(tiled_mma.thr_id.shape) == 2
        arena = storage.arena.data_ptr().align(min_align=1024)
        sA = cute.make_tensor(
            cute.recast_ptr(arena, a_smem_layout.inner, dtype=BFloat16),
            a_smem_layout.outer,
        )
        sB = cute.make_tensor(
            cute.recast_ptr(
                arena + GEMM_A_BYTES,
                b_smem_layout.inner,
                dtype=BFloat16,
            ),
            b_smem_layout.outer,
        )
        sD = cute.make_tensor(
            cute.recast_ptr(
                arena + GEMM_A_BYTES + GEMM_B_BYTES,
                epi_smem_layout.inner,
                dtype=BFloat16,
            ),
            epi_smem_layout.outer,
        )
        ab_pipeline = PipelineTmaUmma.create(
            num_stages=_GATE_AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.collective.num_mcast_ctas_a
                + self.collective.num_mcast_ctas_b
                - 1,
            ),
            tx_count=self.collective.num_tma_load_bytes,
            barrier_storage=storage.gemm_ab_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = PipelineUmmaAsync.create(
            num_stages=_GATE_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.collective.epilog_warp_id) * CLUSTER_SIZE,
            ),
            barrier_storage=storage.gemm_acc_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
        # QuACK 0.6.x delegates the final accumulator-release election to the
        # pipeline; without this flag every epilogue thread would arrive.
        clc_cluster_layout_vmnk = cute.make_layout((1, CLUSTER_SIZE, 1, 1))
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbarriers.data_ptr(),
            num_stages=CLC_PIPE_DEPTH,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                CLC_COMPLETION_WARPS,
            ),
            tx_count=CLC_RESPONSE_BYTES,
            cta_layout_vmnk=clc_cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_drain_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_drain_mbarriers.data_ptr(),
            num_stages=CLC_DRAIN_WARPS,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                CLUSTER_SIZE,
            ),
            tx_count=CLC_RESPONSE_BYTES,
            cta_layout_vmnk=clc_cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=(
                (len(self.collective.epilog_warp_id) + 1)
                * cute.arch.WARP_SIZE
            ),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.collective.epilog_warp_id[0],
            is_two_cta=is_two_cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbarrier.ptr,
        )
        drain_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            CLC_DRAIN_WARPS,
        )
        drain_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.ProducerConsumer,
            CLC_DRAIN_WARPS,
        )
        drain_stage = Int32(0)
        while drain_stage < warp:
            drain_consumer_state.advance()
            drain_producer_state.advance()
            drain_stage = drain_stage + Int32(1)
        # The response array is 16-byte aligned and each warp owns exactly
        # four Int32 values.  Preserve that alignment after the runtime warp
        # offset so the scheduler can load its 16-byte response atomically.
        drain_response_ptr = (
            storage.clc_drain_responses.data_ptr() + warp * Int32(4)
        ).align(min_align=CLC_RESPONSE_BYTES)
        drain_scheduler = utils.ClcDynamicPersistentTileScheduler.create(
            scheduler_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            drain_response_ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE)

        # Each resident warp role owns its pipeline state for the entire CLC
        # lifetime.  Invalid capacity responses advance only CLC in the GEMM
        # roles; valid fused-FC1/Down tasks advance AB, ACC, and store state in
        # the same order in every producer/consumer role.
        if role_warp == Int32(_GATE_AB_LOAD_WARP):
            cpasync.prefetch_descriptor(tma_atom_a_shared_gate_up)
            cpasync.prefetch_descriptor(tma_atom_b_shared_gate)
            cpasync.prefetch_descriptor(tma_atom_b_shared_up)
            cpasync.prefetch_descriptor(tma_atom_a_shared_down)
            cpasync.prefetch_descriptor(tma_atom_b_shared_down)
            cpasync.prefetch_descriptor(tma_atom_d_shared_gate)
            cpasync.prefetch_descriptor(tma_atom_d_shared_up)
            cpasync.prefetch_descriptor(tma_atom_d_shared_hidden)
            cpasync.prefetch_descriptor(tma_atom_d_shared_down)
            cpasync.prefetch_descriptor(tma_atom_a_routed_gate_up)
            cpasync.prefetch_descriptor(tma_atom_b_routed_gate)
            cpasync.prefetch_descriptor(tma_atom_b_routed_up)
            cpasync.prefetch_descriptor(tma_atom_a_routed_down)
            cpasync.prefetch_descriptor(tma_atom_b_routed_down)
            cpasync.prefetch_descriptor(tma_atom_d_routed_gate)
            cpasync.prefetch_descriptor(tma_atom_d_routed_up)
            cpasync.prefetch_descriptor(tma_atom_d_routed_hidden)
            cpasync.prefetch_descriptor(tma_atom_d_routed_down)
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work = scheduler.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                CLC_PIPE_DEPTH,
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _GATE_AB_STAGES,
            )
            thr_mma = tiled_mma.get_slice(cta_rank)
            # Gate and Up are independent N128 tensors.  As in the standalone
            # V1-C2 collective, both use the rank-0 B partition; CTA1 issuing
            # the Up descriptor places that fragment in CTA1-local SMEM.
            b_rank0 = tiled_mma.get_slice(Int32(0))
            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            active_work = Int32(0)
            if work.is_valid_tile:
                if logical_cluster < true_clusters:
                    active_work = Int32(1)

            while active_work != Int32(0):
                (
                    kind,
                    task,
                    macrobatch,
                    minibatch,
                ) = self._decode_compute_task(
                    logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                    minibatch_tasks,
                    macrobatch_size,
                    minibatch_size,
                )
                (
                    valid_tile,
                    expert_index,
                    tile_coord_m,
                    tile_coord_n,
                    input_ready_index,
                    input_ready_required,
                    _,
                    k_tile_count,
                ) = self._decode_routed_endpoint_gemm_tile(
                    kind,
                    task,
                    macrobatch,
                    minibatch,
                    expert_row_block_offsets,
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_row_blocks,
                    macrobatch_size,
                    minibatch_size,
                )
                if valid_tile != Int32(0):
                    with cute.arch.elect_one():
                        if kind == Int32(1):
                            _counter_wait_gpu(
                                hidden_row_block_ready,
                                input_ready_index,
                                input_ready_required,
                            )
                        elif kind == Int32(2):
                            _counter_wait_gpu(
                                x_routed_ready,
                                input_ready_index,
                                input_ready_required,
                            )
                        elif kind == Int32(3):
                            _counter_wait_gpu(
                                hidden_row_block_ready,
                                input_ready_index,
                                input_ready_required,
                            )
                    cute.arch.sync_warp()
                    if kind == Int32(0):
                        gA_gate_up = cute.local_tile(
                            x_shared_tma,
                            cute.select(self.collective.mma_tiler, [0, 2]),
                            (tile_coord_m, None),
                        )
                        gate_tile_base = cute.domain_offset(
                            (
                                tile_coord_n * Int32(_GATE_LOGICAL_N),
                                Int32(0),
                            ),
                            shared_gate_weights,
                        )
                        up_tile_base = cute.domain_offset(
                            (
                                tile_coord_n * Int32(_GATE_LOGICAL_N),
                                Int32(0),
                            ),
                            shared_up_weights,
                        )
                        gB_gate = cute.local_tile(
                            gate_tile_base,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (Int32(0), None),
                        )
                        gB_up = cute.local_tile(
                            up_tile_base,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (Int32(0), None),
                        )
                        tCgA_gate_up = thr_mma.partition_A(gA_gate_up)
                        tCgB_gate = b_rank0.partition_B(gB_gate)
                        tCgB_up = b_rank0.partition_B(gB_up)
                        copy_A_gate_up = _make_tma_block_load_fn(
                            tma_atom_a_shared_gate_up,
                            src_tensor=tCgA_gate_up,
                            dst_tensor=sA,
                        )
                        copy_B_gate = _make_tma_block_load_fn(
                            tma_atom_b_shared_gate,
                            src_tensor=tCgB_gate,
                            dst_tensor=sB,
                        )
                        copy_B_up = _make_tma_block_load_fn(
                            tma_atom_b_shared_up,
                            src_tensor=tCgB_up,
                            dst_tensor=sB,
                        )
                        if cta_rank == Int32(0):
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_gate_up, copy_B_gate],
                                k_tile_count,
                            )
                        else:
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_gate_up, copy_B_up],
                                k_tile_count,
                            )
                    elif kind == Int32(1):
                        gA_down = cute.local_tile(
                            hidden_shared_tma,
                            cute.select(self.collective.mma_tiler, [0, 2]),
                            (tile_coord_m, None),
                        )
                        gB_down = cute.local_tile(
                            shared_down_weights,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (tile_coord_n, None),
                        )
                        tCgA_down = thr_mma.partition_A(gA_down)
                        tCgB_down = thr_mma.partition_B(gB_down)
                        copy_A_down = _make_tma_block_load_fn(
                            tma_atom_a_shared_down,
                            src_tensor=tCgA_down,
                            dst_tensor=sA,
                        )
                        copy_B_down = _make_tma_block_load_fn(
                            tma_atom_b_shared_down,
                            src_tensor=tCgB_down,
                            dst_tensor=sB,
                        )
                        ab_producer_state = self.collective.load_tma(
                            ab_pipeline,
                            ab_producer_state,
                            [copy_A_down, copy_B_down],
                            k_tile_count,
                        )
                    elif kind == Int32(2):
                        gA_gate_up = cute.local_tile(
                            x_routed_tma,
                            cute.select(self.collective.mma_tiler, [0, 2]),
                            (tile_coord_m, None),
                        )
                        routed_gate_expert = routed_gate_weights[
                            None, None, expert_index
                        ]
                        routed_up_expert = routed_up_weights[
                            None, None, expert_index
                        ]
                        gate_tile_base = cute.domain_offset(
                            (
                                tile_coord_n * Int32(_GATE_LOGICAL_N),
                                Int32(0),
                            ),
                            routed_gate_expert,
                        )
                        up_tile_base = cute.domain_offset(
                            (
                                tile_coord_n * Int32(_GATE_LOGICAL_N),
                                Int32(0),
                            ),
                            routed_up_expert,
                        )
                        gB_gate = cute.local_tile(
                            gate_tile_base,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (Int32(0), None),
                        )
                        gB_up = cute.local_tile(
                            up_tile_base,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (Int32(0), None),
                        )
                        tCgA_gate_up = thr_mma.partition_A(gA_gate_up)
                        tCgB_gate = b_rank0.partition_B(gB_gate)
                        tCgB_up = b_rank0.partition_B(gB_up)
                        copy_A_gate_up = _make_tma_block_load_fn(
                            tma_atom_a_routed_gate_up,
                            src_tensor=tCgA_gate_up,
                            dst_tensor=sA,
                        )
                        copy_B_gate = _make_tma_block_load_fn(
                            tma_atom_b_routed_gate,
                            src_tensor=tCgB_gate,
                            dst_tensor=sB,
                        )
                        copy_B_up = _make_tma_block_load_fn(
                            tma_atom_b_routed_up,
                            src_tensor=tCgB_up,
                            dst_tensor=sB,
                        )
                        if cta_rank == Int32(0):
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_gate_up, copy_B_gate],
                                k_tile_count,
                            )
                        else:
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_gate_up, copy_B_up],
                                k_tile_count,
                            )
                    else:
                        gA_down = cute.local_tile(
                            hidden_routed_tma,
                            cute.select(self.collective.mma_tiler, [0, 2]),
                            (tile_coord_m, None),
                        )
                        gB_down = cute.local_tile(
                            routed_down_weights[None, None, expert_index],
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (tile_coord_n, None),
                        )
                        tCgA_down = thr_mma.partition_A(gA_down)
                        tCgB_down = thr_mma.partition_B(gB_down)
                        copy_A_down = _make_tma_block_load_fn(
                            tma_atom_a_routed_down,
                            src_tensor=tCgA_down,
                            dst_tensor=sA,
                        )
                        copy_B_down = _make_tma_block_load_fn(
                            tma_atom_b_routed_down,
                            src_tensor=tCgB_down,
                            dst_tensor=sB,
                        )
                        ab_producer_state = self.collective.load_tma(
                            ab_pipeline,
                            ab_producer_state,
                            [copy_A_down, copy_B_down],
                            k_tile_count,
                        )

                clc_pipeline.consumer_wait(clc_consumer_state)
                work = scheduler.get_current_work()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                next_logical_cluster = (
                    work.tile_idx[0] // Int32(CLUSTER_SIZE)
                )
                next_active_work = Int32(0)
                next_kind = Int32(-1)
                if work.is_valid_tile:
                    if next_logical_cluster < true_clusters:
                        next_active_work = Int32(1)
                        next_kind, _, _, _ = (
                            self._decode_compute_task(
                                next_logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                                minibatch_tasks,
                                macrobatch_size,
                                minibatch_size,
                            )
                        )
                self._sync_after_routed_swiglu_transition(
                    kind,
                    next_kind,
                    next_active_work,
                )
                logical_cluster = next_logical_cluster
                active_work = next_active_work

            ab_pipeline.producer_tail(ab_producer_state)
            needs_drain = Int32(0)
            if work.is_valid_tile:
                if logical_cluster >= true_clusters:
                    needs_drain = Int32(1)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_consumer_state,
                drain_producer_state,
                drain_scheduler,
                cta_rank,
            )

        if role_warp == Int32(_GATE_MMA_WARP):
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work = scheduler.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                CLC_PIPE_DEPTH,
            )
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.collective.acc_dtype)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(
                self.collective.mma_tiler[:2]
            )
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, _GATE_ACC_STAGES)
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _GATE_AB_STAGES,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _GATE_ACC_STAGES,
            )
            mma_op = tiled_mma
            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            active_work = Int32(0)
            if work.is_valid_tile:
                if logical_cluster < true_clusters:
                    active_work = Int32(1)

            while active_work != Int32(0):
                (
                    kind,
                    task,
                    macrobatch,
                    minibatch,
                ) = self._decode_compute_task(
                    logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                    minibatch_tasks,
                    macrobatch_size,
                    minibatch_size,
                )
                (
                    valid_tile,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    k_tile_count,
                ) = self._decode_routed_endpoint_gemm_tile(
                    kind,
                    task,
                    macrobatch,
                    minibatch,
                    expert_row_block_offsets,
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_row_blocks,
                    macrobatch_size,
                    minibatch_size,
                )
                if valid_tile != Int32(0):
                    tCtAcc = tCtAcc_base[
                        None,
                        None,
                        None,
                        acc_producer_state.index,
                    ]
                    (
                        ab_consumer_state,
                        acc_producer_state,
                        mma_op,
                    ) = self.collective.mma(
                        ab_pipeline,
                        acc_pipeline,
                        ab_consumer_state,
                        acc_producer_state,
                        mma_op,
                        tCrA,
                        tCrB,
                        tCtAcc,
                        k_tile_count,
                        is_leader_cta,
                        cta_rank,
                    )

                clc_pipeline.consumer_wait(clc_consumer_state)
                work = scheduler.get_current_work()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                next_logical_cluster = (
                    work.tile_idx[0] // Int32(CLUSTER_SIZE)
                )
                next_active_work = Int32(0)
                next_kind = Int32(-1)
                if work.is_valid_tile:
                    if next_logical_cluster < true_clusters:
                        next_active_work = Int32(1)
                        next_kind, _, _, _ = (
                            self._decode_compute_task(
                                next_logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                                minibatch_tasks,
                                macrobatch_size,
                                minibatch_size,
                            )
                        )
                self._sync_after_routed_swiglu_transition(
                    kind,
                    next_kind,
                    next_active_work,
                )
                logical_cluster = next_logical_cluster
                active_work = next_active_work

            tmem_alloc_barrier.arrive()
            acc_pipeline.producer_tail(acc_producer_state)
            needs_drain = Int32(0)
            if work.is_valid_tile:
                if logical_cluster >= true_clusters:
                    needs_drain = Int32(1)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_consumer_state,
                drain_producer_state,
                drain_scheduler,
                cta_rank,
            )

        if role_warp < Int32(_GATE_MMA_WARP):
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work = scheduler.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                CLC_PIPE_DEPTH,
            )
            tmem.allocate(self.collective.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.collective.acc_dtype)
            acc_shape = tiled_mma.partition_shape_C(
                self.collective.mma_tiler[:2]
            )
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, _GATE_ACC_STAGES)
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.collective.epilog_tmem_copy_and_partition(
                    thread,
                    tCtAcc_base,
                    epi_tile,
                    is_two_cta,
                )
            )
            tTR_rD = cute.make_rmem_tensor(
                tTR_rAcc.shape,
                self.collective.acc_dtype,
            )
            tiled_copy_r2s, tRS_rD, tRS_sD = (
                self.collective.epilog_smem_store_and_partition(
                    tiled_copy_t2r,
                    self.collective.d_layout,
                    self.collective.d_dtype,
                    tTR_rD,
                    sD,
                    thread,
                )
            )
            r_gate_bf16 = cute.make_rmem_tensor_like(tRS_rD, BFloat16)
            r_up_bf16 = cute.make_rmem_tensor_like(tRS_rD, BFloat16)
            rD_bf16 = cute.make_rmem_tensor_like(tRS_rD, BFloat16)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _GATE_ACC_STAGES,
            )
            epi_store_pipeline = self.collective.make_epi_store_pipeline()
            is_tma_warp = Boolean(warp == Int32(0))
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(self.collective.cta_tile_shape_mnk[:2]),
                epi_tile,
            ).shape[1]
            epi_tile_layout = cute.make_ordered_layout(
                epi_tile_shape,
                order=(0, 1),
            )
            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            active_work = Int32(0)
            if work.is_valid_tile:
                if logical_cluster < true_clusters:
                    active_work = Int32(1)

            while active_work != Int32(0):
                (
                    kind,
                    task,
                    macrobatch,
                    minibatch,
                ) = self._decode_compute_task(
                    logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                    minibatch_tasks,
                    macrobatch_size,
                    minibatch_size,
                )
                (
                    valid_tile,
                    _,
                    tile_coord_m,
                    tile_coord_n,
                    _,
                    _,
                    output_ready_index,
                    _,
                ) = self._decode_routed_endpoint_gemm_tile(
                    kind,
                    task,
                    macrobatch,
                    minibatch,
                    expert_row_block_offsets,
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_row_blocks,
                    macrobatch_size,
                    minibatch_size,
                )
                if valid_tile != Int32(0):
                    fc1_tile_coord = (
                        tile_coord_m * Int32(CLUSTER_SIZE) + cta_rank,
                        tile_coord_n,
                        Int32(0),
                        Int32(0),
                    )
                    down_tile_coord = fc1_tile_coord
                    copy_D_shared_gate, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_shared_gate,
                            gate_shared,
                            (_GATE_CTA_M, _GATE_LOGICAL_N),
                            epi_tile,
                            sD,
                            fc1_tile_coord,
                        )
                    )
                    copy_D_shared_up, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_shared_up,
                            up_shared,
                            (_GATE_CTA_M, _GATE_LOGICAL_N),
                            epi_tile,
                            sD,
                            fc1_tile_coord,
                        )
                    )
                    copy_D_shared_hidden, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_shared_hidden,
                            hidden_shared,
                            (_GATE_CTA_M, _GATE_LOGICAL_N),
                            epi_tile,
                            sD,
                            fc1_tile_coord,
                        )
                    )
                    copy_D_shared_down, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_shared_down,
                            y_shared_tma,
                            self.collective.cta_tile_shape_mnk[:2],
                            epi_tile,
                            sD,
                            down_tile_coord,
                        )
                    )
                    copy_D_routed_gate, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_routed_gate,
                            gate_routed,
                            (_GATE_CTA_M, _GATE_LOGICAL_N),
                            epi_tile,
                            sD,
                            fc1_tile_coord,
                        )
                    )
                    copy_D_routed_up, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_routed_up,
                            up_routed,
                            (_GATE_CTA_M, _GATE_LOGICAL_N),
                            epi_tile,
                            sD,
                            fc1_tile_coord,
                        )
                    )
                    copy_D_routed_hidden, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_routed_hidden,
                            hidden_routed,
                            (_GATE_CTA_M, _GATE_LOGICAL_N),
                            epi_tile,
                            sD,
                            fc1_tile_coord,
                        )
                    )
                    copy_D_routed_down, _, _ = (
                        self.collective.epilog_gmem_copy_and_partition(
                            tma_atom_d_routed_down,
                            y_routed_tma,
                            self.collective.cta_tile_shape_mnk[:2],
                            epi_tile,
                            sD,
                            down_tile_coord,
                        )
                    )
                    tTR_tAcc = tTR_tAcc_base[
                        None,
                        None,
                        None,
                        None,
                        None,
                        acc_consumer_state.index,
                    ]
                    tTR_tAcc = cute.group_modes(
                        tTR_tAcc,
                        3,
                        cute.rank(tTR_tAcc),
                    )
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    if kind == Int32(0) or kind == Int32(2):
                        # CUDA tcgen mapping is Gate blocks 0..3 and Up blocks
                        # 4..7.  The paired load order keeps TMEM alive until
                        # Up block 7, where epi_load_acc_subtile releases it.
                        save_context = Int32(kind == Int32(0))
                        if kind == Int32(2):
                            save_context = Int32(macrobatch == Int32(0))
                        for pair in cutlass.range_constexpr(
                            _GATE_LOGICAL_N // _GATE_EPILOGUE_N
                        ):
                            gate_coord = epi_tile_layout.get_hier_coord(pair)
                            up_coord = epi_tile_layout.get_hier_coord(
                                pair + _GATE_LOGICAL_N // _GATE_EPILOGUE_N
                            )
                            self.collective.epi_load_acc_subtile(
                                tiled_copy_t2r,
                                tiled_copy_r2s,
                                tTR_tAcc,
                                tTR_rAcc,
                                tRS_rD,
                                gate_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            r_gate_bf16.store(
                                tRS_rD.load().to(BFloat16)
                            )
                            if save_context != Int32(0):
                                if is_tma_warp:
                                    epi_store_pipeline.producer_acquire()
                                self.collective.epilogue_barrier.arrive_and_wait()
                                cute.copy(
                                    tiled_copy_r2s,
                                    r_gate_bf16,
                                    tRS_sD[None, None, None, 0],
                                )
                                cute.arch.fence_view_async_shared()
                                self.collective.epilogue_barrier.arrive_and_wait()
                                if is_tma_warp:
                                    if kind == Int32(0):
                                        copy_D_shared_gate(
                                            src_idx=0, dst_idx=gate_coord
                                        )
                                    else:
                                        copy_D_routed_gate(
                                            src_idx=0, dst_idx=gate_coord
                                        )
                                    epi_store_pipeline.producer_commit()

                            self.collective.epi_load_acc_subtile(
                                tiled_copy_t2r,
                                tiled_copy_r2s,
                                tTR_tAcc,
                                tTR_rAcc,
                                tRS_rD,
                                up_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            r_up_bf16.store(
                                tRS_rD.load().to(BFloat16)
                            )
                            if save_context != Int32(0):
                                if is_tma_warp:
                                    epi_store_pipeline.producer_acquire()
                                self.collective.epilogue_barrier.arrive_and_wait()
                                cute.copy(
                                    tiled_copy_r2s,
                                    r_up_bf16,
                                    tRS_sD[None, None, None, 1],
                                )
                                cute.arch.fence_view_async_shared()
                                self.collective.epilogue_barrier.arrive_and_wait()
                                if is_tma_warp:
                                    if kind == Int32(0):
                                        copy_D_shared_up(
                                            src_idx=1, dst_idx=gate_coord
                                        )
                                    else:
                                        copy_D_routed_up(
                                            src_idx=1, dst_idx=gate_coord
                                        )
                                    epi_store_pipeline.producer_commit()

                            r_hidden = cute.make_rmem_tensor(
                                r_up_bf16.shape, BFloat16
                            )
                            for value in cutlass.range_constexpr(
                                cute.size(r_hidden)
                            ):
                                gate_value = Float32(r_gate_bf16[value])
                                up_value = Float32(r_up_bf16[value])
                                if is_clamped:
                                    limit = Float32(swiglu_limit)
                                    gate_value = cutlass.min(
                                        gate_value, limit
                                    )
                                    up_value = cutlass.min(
                                        cutlass.max(up_value, -limit),
                                        limit,
                                    )
                                r_hidden[value] = swiglu(
                                    gate_value, up_value
                                ).to(BFloat16)
                            if is_tma_warp:
                                epi_store_pipeline.producer_acquire()
                            self.collective.epilogue_barrier.arrive_and_wait()
                            hidden_buffer = Int32(2)
                            if save_context == Int32(0):
                                hidden_buffer = Int32(pair % _GATE_D_STAGES)
                            cute.copy(
                                tiled_copy_r2s,
                                r_hidden,
                                tRS_sD[None, None, None, hidden_buffer],
                            )
                            cute.arch.fence_view_async_shared()
                            self.collective.epilogue_barrier.arrive_and_wait()
                            if is_tma_warp:
                                if kind == Int32(0):
                                    copy_D_shared_hidden(
                                        src_idx=hidden_buffer,
                                        dst_idx=gate_coord,
                                    )
                                else:
                                    copy_D_routed_hidden(
                                        src_idx=hidden_buffer,
                                        dst_idx=gate_coord,
                                    )
                                epi_store_pipeline.producer_commit()
                        acc_consumer_state.advance()
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                            with cute.arch.elect_one():
                                _counter_arrive_gpu(
                                    hidden_row_block_ready,
                                    output_ready_index,
                                    Int32(1),
                                )
                    else:
                        # Down keeps the original M128xN256 BF16 epilogue.
                        if kind == Int32(3):
                            num_macrobatches = _ceil_div_i32(
                                num_tokens, Int32(macrobatch_size)
                            )
                            if macrobatch + Int32(1) < num_macrobatches:
                                prior_offset = (
                                    (macrobatch + Int32(1))
                                    * Int32(macrobatch_size)
                                )
                                prior_rows = cutlass.min(
                                    Int32(macrobatch_size),
                                    num_tokens - prior_offset,
                                )
                                local_cta_row = (
                                    tile_coord_m * Int32(MLP_TILE_ROWS)
                                    + cta_rank * Int32(_GATE_CTA_M)
                                )
                                if local_cta_row < prior_rows:
                                    if is_tma_warp:
                                        with cute.arch.elect_one():
                                            prior_done_index = (
                                                prior_offset + local_cta_row
                                            ) // Int32(_GATE_CTA_M)
                                            _counter_wait_gpu(
                                                y_routed_done,
                                                prior_done_index,
                                                Int32(32),
                                            )
                                    self.collective.epilogue_barrier.arrive_and_wait()
                        for epi_idx in cutlass.range_constexpr(
                            _GATE_EPILOGUE_SUBTILES
                        ):
                            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                            self.collective.epi_load_acc_subtile(
                                tiled_copy_t2r,
                                tiled_copy_r2s,
                                tTR_tAcc,
                                tTR_rAcc,
                                tRS_rD,
                                epi_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            rD_bf16.store(tRS_rD.load().to(BFloat16))
                            if is_tma_warp:
                                epi_store_pipeline.producer_acquire()
                            self.collective.epilogue_barrier.arrive_and_wait()
                            epi_buffer = epi_idx % _GATE_D_STAGES
                            cute.copy(
                                tiled_copy_r2s,
                                rD_bf16,
                                tRS_sD[None, None, None, epi_buffer],
                            )
                            cute.arch.fence_view_async_shared()
                            self.collective.epilogue_barrier.arrive_and_wait()
                            if is_tma_warp:
                                if kind == Int32(1):
                                    copy_D_shared_down(
                                        src_idx=epi_buffer,
                                        dst_idx=epi_coord,
                                    )
                                else:
                                    copy_D_routed_down(
                                        src_idx=epi_buffer,
                                        dst_idx=epi_coord,
                                    )
                                epi_store_pipeline.producer_commit()
                        acc_consumer_state.advance()
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                            if kind == Int32(3):
                                with cute.arch.elect_one():
                                    _counter_arrive_gpu(
                                        y_routed_ready,
                                        output_ready_index,
                                        Int32(1),
                                    )

                clc_pipeline.consumer_wait(clc_consumer_state)
                work = scheduler.get_current_work()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                next_logical_cluster = (
                    work.tile_idx[0] // Int32(CLUSTER_SIZE)
                )
                next_active_work = Int32(0)
                next_kind = Int32(-1)
                if work.is_valid_tile:
                    if next_logical_cluster < true_clusters:
                        next_active_work = Int32(1)
                        next_kind, _, _, _ = (
                            self._decode_compute_task(
                                next_logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                                minibatch_tasks,
                                macrobatch_size,
                                minibatch_size,
                            )
                        )
                self._sync_after_routed_swiglu_transition(
                    kind,
                    next_kind,
                    next_active_work,
                )
                logical_cluster = next_logical_cluster
                active_work = next_active_work

            if is_tma_warp:
                epi_store_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            needs_drain = Int32(0)
            if work.is_valid_tile:
                if logical_cluster >= true_clusters:
                    needs_drain = Int32(1)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_consumer_state,
                drain_producer_state,
                drain_scheduler,
                cta_rank,
            )

        if role_warp == Int32(_GATE_CLC_WARP):
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work = scheduler.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                CLC_PIPE_DEPTH,
            )
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer,
                CLC_PIPE_DEPTH,
            )
            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            active_work = Int32(0)
            if work.is_valid_tile:
                if logical_cluster < true_clusters:
                    active_work = Int32(1)

            while active_work != Int32(0):
                kind, _, _, _ = self._decode_compute_task(
                    logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                    minibatch_tasks,
                    macrobatch_size,
                    minibatch_size,
                )
                if cta_rank == Int32(0):
                    clc_pipeline.producer_acquire(clc_producer_state)
                    scheduler.advance_to_next_work(
                        clc_pipeline.producer_get_barrier(clc_producer_state)
                    )
                    clc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work = scheduler.get_current_work()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                next_logical_cluster = (
                    work.tile_idx[0] // Int32(CLUSTER_SIZE)
                )
                next_active_work = Int32(0)
                next_kind = Int32(-1)
                if work.is_valid_tile:
                    if next_logical_cluster < true_clusters:
                        next_active_work = Int32(1)
                        next_kind, _, _, _ = (
                            self._decode_compute_task(
                                next_logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                                minibatch_tasks,
                                macrobatch_size,
                                minibatch_size,
                            )
                        )
                self._sync_after_routed_swiglu_transition(
                    kind,
                    next_kind,
                    next_active_work,
                )
                logical_cluster = next_logical_cluster
                active_work = next_active_work

            if cta_rank == Int32(0):
                clc_pipeline.producer_tail(clc_producer_state)
            needs_drain = Int32(0)
            if work.is_valid_tile:
                if logical_cluster >= true_clusters:
                    needs_drain = Int32(1)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_consumer_state,
                drain_producer_state,
                drain_scheduler,
                cta_rank,
            )

        if role_warp == Int32(_GATE_BF16_IDLE_WARP):
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work = scheduler.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                CLC_PIPE_DEPTH,
            )
            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            active_work = Int32(0)
            if work.is_valid_tile:
                if logical_cluster < true_clusters:
                    active_work = Int32(1)

            while active_work != Int32(0):
                kind, _, _, _ = self._decode_compute_task(
                    logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                    minibatch_tasks,
                    macrobatch_size,
                    minibatch_size,
                )
                clc_pipeline.consumer_wait(clc_consumer_state)
                work = scheduler.get_current_work()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                next_logical_cluster = (
                    work.tile_idx[0] // Int32(CLUSTER_SIZE)
                )
                next_active_work = Int32(0)
                next_kind = Int32(-1)
                if work.is_valid_tile:
                    if next_logical_cluster < true_clusters:
                        next_active_work = Int32(1)
                        next_kind, _, _, _ = (
                            self._decode_compute_task(
                                next_logical_cluster - Int32(comm_clusters),
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                                minibatch_tasks,
                                macrobatch_size,
                                minibatch_size,
                            )
                        )
                self._sync_after_routed_swiglu_transition(
                    kind,
                    next_kind,
                    next_active_work,
                )
                logical_cluster = next_logical_cluster
                active_work = next_active_work

            needs_drain = Int32(0)
            if work.is_valid_tile:
                if logical_cluster >= true_clusters:
                    needs_drain = Int32(1)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_consumer_state,
                drain_producer_state,
                drain_scheduler,
                cta_rank,
            )


_MEGA = _PersistentBf16Mega()
_MEGA_RUNTIME_PREFIX_ARGS = 27


def _persistent_bf16_ptrs(pointers) -> list[cute.Pointer]:
    if len(pointers) != EP_SIZE:
        raise ValueError(f"expected {EP_SIZE} peer pointers")
    return [
        make_ptr(
            BFloat16,
            int(pointer),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for pointer in pointers
    ]


def make_mega_args(
    x_peer_addresses,
    combine_peer_addresses,
    schedule_peer_rank,
    schedule_peer_token_idx,
    num_tokens,
    tokens_per_expert,
    x_shared,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    state,
    *,
    num_local_tokens: int,
    schedule_capacity: int,
    macrobatch_size: int = 32768,
    minibatch_size: int = 4096,
    num_comm_sms: int = 40,
    swiglu_limit: float = 0.0,
    is_clamped: bool = False,
    stream=None,
):
    """Build the exact CuTe argument tuple after the host gate succeeds.

    ``state`` is duck-typed to the nine-output/five-counter object owned by
    :mod:`persistent_bf16`; keeping that import outside this module prevents
    the CPU-only host validator from importing CUTLASS.  Routed weights stay
    as strided ``[N,K,E]`` views, never a packed or copied mirror.
    """

    import torch

    if stream is None:
        stream = torch.cuda.current_stream(x_shared.device)
    cuda_stream = cuda.CUstream(stream.cuda_stream)
    routed_gate_nke = routed_gate_weights.permute(1, 2, 0)
    routed_up_nke = routed_up_weights.permute(1, 2, 0)
    routed_down_nke = routed_down_weights.permute(1, 2, 0)
    return (
        _persistent_bf16_ptrs(x_peer_addresses),
        _persistent_bf16_ptrs(combine_peer_addresses),
        from_dlpack(schedule_peer_rank, assumed_align=16),
        from_dlpack(schedule_peer_token_idx, assumed_align=16),
        from_dlpack(num_tokens, assumed_align=16),
        from_dlpack(tokens_per_expert, assumed_align=16),
        from_dlpack(x_shared, assumed_align=16),
        from_dlpack(state.x_routed, assumed_align=16),
        from_dlpack(shared_gate_weights, assumed_align=16),
        from_dlpack(routed_gate_nke, assumed_align=16),
        from_dlpack(shared_up_weights, assumed_align=16),
        from_dlpack(routed_up_nke, assumed_align=16),
        from_dlpack(shared_down_weights, assumed_align=16),
        from_dlpack(routed_down_nke, assumed_align=16),
        from_dlpack(state.gate_shared, assumed_align=16),
        from_dlpack(state.gate_routed, assumed_align=16),
        from_dlpack(state.up_shared, assumed_align=16),
        from_dlpack(state.up_routed, assumed_align=16),
        from_dlpack(state.hidden_shared, assumed_align=16),
        from_dlpack(state.hidden_routed, assumed_align=16),
        from_dlpack(state.y_shared, assumed_align=16),
        from_dlpack(state.y_routed, assumed_align=16),
        from_dlpack(state.gate_up_tile_ready, assumed_align=16),
        from_dlpack(state.hidden_row_block_ready, assumed_align=16),
        from_dlpack(state.x_routed_ready, assumed_align=16),
        from_dlpack(state.y_routed_ready, assumed_align=16),
        from_dlpack(state.y_routed_done, assumed_align=16),
        num_local_tokens,
        schedule_capacity,
        macrobatch_size,
        minibatch_size,
        num_comm_sms,
        swiglu_limit,
        is_clamped,
        cuda_stream,
    )


def compile_mega_bf16(*args, **kwargs):
    """Compile the one-launch BF16 specialization without executing it."""

    executor, _ = prepare_mega_bf16(*args, **kwargs)
    return executor


def prepare_mega_bf16(*args, **kwargs):
    """Compile once and bind the dynamic executor arguments for reuse."""

    cute_args = make_mega_args(*args, **kwargs)
    executor = cute.compile(_MEGA, *cute_args)
    # CUTLASS DSL removes the seven trailing Constexpr parameters from the
    # executor ABI.  Keep them in the compile call, but launch with the first
    # 27 dynamic arguments followed by the CUDA stream (the final argument).
    runtime_args = cute_args[:_MEGA_RUNTIME_PREFIX_ARGS] + cute_args[-1:]
    return executor, runtime_args


def run_mega_bf16(*args, **kwargs) -> None:
    """Compile and execute once with the identical CuTe argument tuple."""

    executor, runtime_args = prepare_mega_bf16(*args, **kwargs)
    executor(*runtime_args)


__all__ = [
    "_MEGA",
    "compile_mega_bf16",
    "make_mega_args",
    "prepare_mega_bf16",
    "run_mega_bf16",
]
