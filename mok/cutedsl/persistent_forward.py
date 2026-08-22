"""CUDA-shaped persistent CuTe DSL forward bring-up core.

This module is deliberately not selected by :mod:`mok.functional` yet.  It
contains the first device-resident topology shared with
``csrc/mok_megakernel.cuh``:

* a capacity-sized cluster-2 launch;
* a fixed communication-cluster prefix that performs real peer dispatch and
  combine TMA traffic in reverse macrobatch order;
* a compute-only depth-1 CLC suffix with the CUDA task decoder; and
* GPU-scope monotonic counter acquire/release operations.

The first compute risk slice below is one private, exact-shape Shared Gate
collective.  Its bring-up wrapper exists only to cold-JIT and validate that
device collective; it is not selected by the public backend.  The remaining
collectives are still disconnected from ``_PersistentForwardBringupKernel``,
and this file must not be described as a functional or performance-comparable
forward backend yet.
"""

from dataclasses import dataclass

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum

import quack
import quack.copy_utils as quack_copy_utils
from quack.gemm_base import NamedBarrierGemm
from quack.gemm_default_epi import GemmDefaultSm100
from quack.pipeline import PipelineTmaUmma, PipelineUmmaAsync
from quack.varlen_utils import VarlenArguments

from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw
from .forward_contract import (
    COMBINE_ROW_CHUNK_BYTES,
    COMBINE_TILE_COLUMNS,
    COMBINE_TILE_ROWS,
    DISPATCH_ROW_CHUNK_BYTES,
    DISPATCH_TILE_COLUMNS,
    DISPATCH_TILE_ROWS,
    EP_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_LOCAL_EXPERTS,
    TOPK,
    validate_fixed_forward_contract,
    validate_num_comm_sms,
)
from .persistent_forward_contract import (
    CLC_COMPLETION_WARPS,
    CLC_DRAIN_WARPS,
    CLC_PIPE_DEPTH,
    CLUSTER_SIZE,
    COMBINE_PIPE_DEPTH,
    MLP_TILE_COLUMNS,
    MLP_TILE_ROWS,
    SWIGLU_PIPE_DEPTH,
    make_forward_geometry,
    nine_output_shapes,
)


CLUSTER_SHAPE = (CLUSTER_SIZE, 1, 1)
THREADS_PER_CTA = 256
WARPS_PER_CTA = THREADS_PER_CTA // 32
SCHEDULER_WARP = 5
CLC_RESPONSE_BYTES = 16
COMM_ARENA_BYTES = 7 * 16 * 1024 * 2  # CUDA combine: 7 x 16 x 1024 BF16
GEMM_A_BYTES = 6 * 128 * 64 * 2
GEMM_B_BYTES = 6 * 128 * 64 * 2
GEMM_D_BYTES = 3 * 128 * 32 * 2
GEMM_ARENA_BYTES = GEMM_A_BYTES + GEMM_B_BYTES + GEMM_D_BYTES
REUSABLE_ARENA_BYTES = max(COMM_ARENA_BYTES, GEMM_ARENA_BYTES)

_REQUIRED_QUACK_VERSION = "0.6.4"
if quack.__version__ != _REQUIRED_QUACK_VERSION:
    raise RuntimeError(
        "MoK's persistent CuTe DSL forward requires "
        f"quack-kernels=={_REQUIRED_QUACK_VERSION}; got {quack.__version__}"
    )

_GATE_TILE_M = 256
_GATE_TILE_N = 256
_GATE_TILE_K = 64
_GATE_CTA_M = 128
_GATE_EPILOGUE_N = 32
_GATE_AB_STAGES = 6
_GATE_ACC_STAGES = 1
_GATE_D_STAGES = 3
_GATE_EPILOGUE_SUBTILES = _GATE_TILE_N // _GATE_EPILOGUE_N
_GATE_MMA_WARP = 4
_GATE_CLC_WARP = 5
_GATE_BF16_IDLE_WARP = 6
_GATE_AB_LOAD_WARP = 7


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
        2 * CLC_DRAIN_WARPS,
    ]
    clc_drain_responses: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, 4 * CLC_DRAIN_WARPS],
        16,
    ]

    # Kept explicit for the QuACK collective seam.  Six is the A/B
    # load pipeline, while two is the accumulator full/empty pair.  The eight
    # N=32 epilogue subtiles are not an eight-stage output ring; CUDA reuses
    # three 128x32 BF16 D tiles.
    gemm_ab_mbarriers: cute.struct.MemRange[cutlass.Int64, 12]
    gemm_acc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2]
    swiglu_mbarriers: cute.struct.MemRange[cutlass.Int64, 3]
    tmem_dealloc_mbarrier: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32

    # Dispatch has one CTA-wide transaction barrier; combine has one barrier
    # per seven-stage tile.  All roles alias the same arena rather than adding
    # their per-role storage requirements.
    dispatch_mbarrier: cutlass.Int64
    combine_mbarriers: cute.struct.MemRange[cutlass.Int64, 7]
    arena: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, REUSABLE_ARENA_BYTES],
        1024,
    ]


@dataclass(frozen=True)
class PersistentForwardState:
    """CUDA BF16 forward ABI tensors plus its five internal counters."""

    x_routed: torch.Tensor
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor
    up_shared: torch.Tensor
    up_routed: torch.Tensor
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor
    y_shared: torch.Tensor
    y_routed: torch.Tensor
    gate_up_tile_ready: torch.Tensor
    hidden_row_block_ready: torch.Tensor
    x_routed_ready: torch.Tensor
    y_routed_ready: torch.Tensor
    y_routed_done: torch.Tensor

    @property
    def abi_outputs(self) -> tuple[torch.Tensor, ...]:
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
    def counters(self) -> tuple[torch.Tensor, ...]:
        return (
            self.gate_up_tile_ready,
            self.hidden_row_block_ready,
            self.x_routed_ready,
            self.y_routed_ready,
            self.y_routed_done,
        )


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
        if observed < required:
            cute.arch.nanosleep(sleep_time=16)


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


class _SharedGateTileCollective(GemmDefaultSm100):
    """One device-resident 256x256x64 Shared Gate BF16 collective.

    The collective owns no launch policy.  Its caller supplies the exact TMA
    descriptors and the persistent kernel's shared storage, so the same device
    body can be called from the future CLC task loop.
    """

    @classmethod
    def _compute_stages(cls, *args, **kwargs):
        del args, kwargs
        return (
            _GATE_ACC_STAGES,
            _GATE_AB_STAGES,
            _GATE_D_STAGES,
            0,
        )

    def __init__(self) -> None:
        super().__init__(
            acc_dtype=Float32,
            a_dtype=BFloat16,
            mma_tiler_mnk=(_GATE_TILE_M, _GATE_TILE_N, _GATE_TILE_K),
            cluster_shape_mnk=CLUSTER_SHAPE,
            use_clc_persistence=False,
            use_pdl=False,
        )
        # Preserve the CUDA persistent-kernel role map.  Warp 5 remains
        # reserved for CLC, and warp 6 has no BF16-path work.
        self.mma_warp_id = _GATE_MMA_WARP
        self.scheduler_warp_id = _GATE_CLC_WARP
        self.epi_load_warp_id = _GATE_BF16_IDLE_WARP
        self.ab_load_warp_id = _GATE_AB_LOAD_WARP
        assert self.epilog_warp_id == (0, 1, 2, 3)
        assert self.threads_per_cta == THREADS_PER_CTA

    def configure(
        self,
        mA_mk: cute.Tensor,
        mB_nk: cute.Tensor,
        mD_mn: cute.Tensor,
    ) -> None:
        """Resolve QuACK's fixed layouts without entering its host GEMM."""

        self.a_dtype = mA_mk.element_type
        self.b_dtype = mB_nk.element_type
        self.a_mma_dtype = self.a_dtype
        self.b_mma_dtype = self.b_dtype
        self.a_unpack = False
        self.b_unpack = False
        self.a_smem_dtype = self.a_dtype
        self.b_smem_dtype = self.b_dtype
        self.d_dtype = mD_mn.element_type
        self.c_dtype = None
        self.sf_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA_mk)
        self.b_layout = LayoutEnum.from_tensor(mB_nk)
        self.d_layout = LayoutEnum.from_tensor(mD_mn)
        self.c_layout = None
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()
        self.varlen_m = False
        self.varlen_k = False
        self._setup_attributes(self.EpilogueArguments(), VarlenArguments())

        assert self.a_dtype is BFloat16
        assert self.b_dtype is BFloat16
        assert self.d_dtype is BFloat16
        assert self.use_2cta_instrs
        assert self.cta_tile_shape_mnk == (
            _GATE_CTA_M,
            _GATE_TILE_N,
            _GATE_TILE_K,
        )
        assert self.num_acc_stage == _GATE_ACC_STAGES
        assert self.ab_stage == _GATE_AB_STAGES
        assert self.epi_stage == _GATE_D_STAGES
        assert cute.size(self.epi_tile[0]) == _GATE_CTA_M
        assert cute.size(self.epi_tile[1]) == _GATE_EPILOGUE_N
        assert cute.cosize(self.a_smem_layout_staged.outer) == (
            _GATE_AB_STAGES * _GATE_CTA_M * _GATE_TILE_K
        )
        assert cute.cosize(self.b_smem_layout_staged.outer) == (
            _GATE_AB_STAGES * _GATE_CTA_M * _GATE_TILE_K
        )
        assert cute.cosize(self.epi_smem_layout_staged.outer) == (
            _GATE_D_STAGES * _GATE_CTA_M * _GATE_EPILOGUE_N
        )

    @cute.jit
    def __call__(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mn: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        storage: _PersistentSharedStorage,
        ready: cute.Tensor,
    ) -> None:
        """Execute the tile; this device collective never launches a kernel."""

        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread = cute.arch.thread_idx()[0]
        cta_rank = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        mma_tile_coord_v = cta_rank
        is_leader_cta = mma_tile_coord_v == Int32(0)
        is_two_cta = cute.size(tiled_mma.thr_id.shape) == 2

        if warp == Int32(_GATE_AB_LOAD_WARP):
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_d)

        arena = storage.arena.data_ptr().align(
            min_align=1024,
        )
        sA = cute.make_tensor(
            cute.recast_ptr(
                arena,
                a_smem_layout.inner,
                dtype=BFloat16,
            ),
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
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                1,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1,
            ),
            tx_count=self.num_tma_load_bytes,
            barrier_storage=storage.gemm_ab_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = PipelineUmmaAsync.create(
            num_stages=_GATE_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                1,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.epilog_warp_id) * CLUSTER_SIZE,
            ),
            barrier_storage=storage.gemm_acc_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=(
                (len(self.epilog_warp_id) + 1) * cute.arch.WARP_SIZE
            ),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=is_two_cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbarrier.ptr,
        )

        pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk,
            is_relaxed=True,
        )
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        gA_mk = cute.local_tile(
            mA_mk,
            cute.select(self.mma_tiler, [0, 2]),
            (Int32(0), None),
        )
        gB_nk = cute.local_tile(
            mB_nk,
            cute.select(self.mma_tiler, [1, 2]),
            (Int32(0), None),
        )
        tCgA = thr_mma.partition_A(gA_mk)
        tCgB = thr_mma.partition_B(gB_nk)
        copy_A = quack_copy_utils.tma_get_block_copy_fn(
            tma_atom_a,
            src_tensor=tCgA,
            dst_tensor=sA,
            tma_multicast={
                "cluster_shape": CLUSTER_SHAPE[:2],
                "multicast_dim": "M",
            },
        )
        copy_B = quack_copy_utils.tma_get_block_copy_fn(
            tma_atom_b,
            src_tensor=tCgB,
            dst_tensor=sB,
            tma_multicast={
                "cluster_shape": CLUSTER_SHAPE[:2],
                "multicast_dim": "N",
            },
        )

        tile_coord_mnkl = (
            cta_rank,
            Int32(0),
            Int32(0),
            Int32(0),
        )
        copy_D, _, _ = self.epilog_gmem_copy_and_partition(
            tma_atom_d,
            mD_mn,
            self.cta_tile_shape_mnk[:2],
            epi_tile,
            sD,
            tile_coord_mnkl,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, _GATE_ACC_STAGES)
        )

        if warp == Int32(_GATE_AB_LOAD_WARP):
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _GATE_AB_STAGES,
            )
            ab_producer_state = self.load_tma(
                ab_pipeline,
                ab_producer_state,
                [copy_A, copy_B],
                Int32(1),
            )
            ab_pipeline.producer_tail(ab_producer_state)

        if warp == Int32(_GATE_MMA_WARP):
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            tCtAcc = cute.make_tensor(
                acc_tmem_ptr,
                tCtAcc_fake.layout,
            )[None, None, None, 0]
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _GATE_AB_STAGES,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _GATE_ACC_STAGES,
            )
            (
                _,
                acc_producer_state,
                tiled_mma,
            ) = self.mma(
                ab_pipeline,
                acc_pipeline,
                ab_consumer_state,
                acc_producer_state,
                tiled_mma,
                tCrA,
                tCrB,
                tCtAcc,
                Int32(1),
                is_leader_cta,
                cta_rank,
            )
            tmem_alloc_barrier.arrive()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp < Int32(_GATE_MMA_WARP):
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(
                acc_tmem_ptr,
                tCtAcc_fake.layout,
            )

            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    thread,
                    tCtAcc_base,
                    epi_tile,
                    is_two_cta,
                )
            )
            tTR_rD = cute.make_rmem_tensor(
                tTR_rAcc.shape,
                self.acc_dtype,
            )
            tiled_copy_r2s, tRS_rD, tRS_sD = (
                self.epilog_smem_store_and_partition(
                    tiled_copy_t2r,
                    self.d_layout,
                    self.d_dtype,
                    tTR_rD,
                    sD,
                    thread,
                )
            )
            tTR_tAcc = tTR_tAcc_base[
                None,
                None,
                None,
                None,
                None,
                0,
            ]
            tTR_tAcc = cute.group_modes(
                tTR_tAcc,
                3,
                cute.rank(tTR_tAcc),
            )
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _GATE_ACC_STAGES,
            )
            acc_pipeline.consumer_wait(acc_consumer_state)
            epi_store_pipeline = self.make_epi_store_pipeline()
            is_tma_warp = Boolean(warp == Int32(0))
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(self.cta_tile_shape_mnk[:2]),
                epi_tile,
            ).shape[1]
            epi_tile_layout = cute.make_ordered_layout(
                epi_tile_shape,
                order=(0, 1),
            )

            for epi_idx in cutlass.range_constexpr(
                _GATE_EPILOGUE_SUBTILES
            ):
                epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                self.epi_load_acc_subtile(
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
                rD_bf16 = tRS_rD.to(BFloat16)
                if is_tma_warp:
                    epi_store_pipeline.producer_acquire()
                self.epilogue_barrier.arrive_and_wait()
                epi_buffer = epi_idx % _GATE_D_STAGES
                cute.copy(
                    tiled_copy_r2s,
                    rD_bf16,
                    tRS_sD[None, None, None, epi_buffer],
                )
                cute.arch.fence_view_async_shared()
                self.epilogue_barrier.arrive_and_wait()
                if is_tma_warp:
                    copy_D(
                        src_idx=epi_buffer,
                        dst_idx=epi_coord,
                    )
                    epi_store_pipeline.producer_commit()

            acc_consumer_state.advance()
            if is_tma_warp:
                epi_store_pipeline.producer_tail()
                with cute.arch.elect_one():
                    _counter_arrive_gpu(
                        ready,
                        Int32(0),
                        Int32(1),
                    )

            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)


class _SharedGateTileBringupKernel:
    """Private exact-shape host launcher for cold JIT and correctness only."""

    def __init__(self) -> None:
        self.collective = _SharedGateTileCollective()

    @cute.jit
    def __call__(
        self,
        mA_mk: cute.Tensor,
        mB_nk: cute.Tensor,
        mD_mn: cute.Tensor,
        ready: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        collective = self.collective
        collective.configure(mA_mk, mB_nk, mD_mn)

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
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA_mk,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB_nk,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        tma_atom_d, tma_tensor_d = (
            collective._make_tma_epi_atoms_and_tensors(
                mD_mn,
                collective.epi_smem_layout_staged,
                collective.epi_tile,
                op_type="store",
            )
        )
        collective.num_tma_load_bytes = (
            cute.size_in_bytes(BFloat16, a_smem_layout)
            + cute.size_in_bytes(BFloat16, b_smem_layout)
        ) * cute.size(collective.tiled_mma.thr_id.shape)

        self.kernel(
            collective.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            collective.cluster_layout_vmnk,
            collective.a_smem_layout_staged,
            collective.b_smem_layout_staged,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            ready,
        ).launch(
            grid=(2, 1, 1),
            block=(256, 1, 1),
            cluster=(2, 1, 1),
            smem=_PersistentSharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mn: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        ready: cute.Tensor,
    ) -> None:
        smem = utils.SmemAllocator()
        storage = smem.allocate(_PersistentSharedStorage)
        self.collective(
            tiled_mma,
            tma_atom_a,
            mA_mk,
            tma_atom_b,
            mB_nk,
            tma_atom_d,
            mD_mn,
            cluster_layout_vmnk,
            a_smem_layout,
            b_smem_layout,
            epi_smem_layout,
            epi_tile,
            storage,
            ready,
        )


_SHARED_GATE_TILE_BRINGUP = _SharedGateTileBringupKernel()

class _PersistentForwardBringupKernel:
    """Private communication + scheduling core awaiting compute collectives."""

    @cute.jit
    def __call__(
        self,
        x_peer_ptrs: list[cute.Pointer],
        combine_peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        x_routed: cute.Tensor,
        y_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        compute_task_log: cute.Tensor,
        num_local_tokens: cutlass.Constexpr,
        schedule_capacity: cutlass.Constexpr,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        del gate_up_tile_ready, hidden_row_block_ready
        geometry = make_forward_geometry(
            num_local_tokens=num_local_tokens,
            schedule_capacity=schedule_capacity,
            macrobatch_size=macrobatch_size,
            minibatch_size=minibatch_size,
            num_comm_sms=num_comm_sms,
        )
        scheduler_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (geometry.capacity_launch_ctas, 1, 1),
            CLUSTER_SHAPE,
        )
        self.kernel(
            x_peer_ptrs,
            combine_peer_ptrs,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            x_routed,
            y_routed,
            x_routed_ready,
            y_routed_ready,
            y_routed_done,
            compute_task_log,
            scheduler_params,
            geometry.comm_clusters,
            geometry.shared_gate_up_tasks,
            geometry.shared_swiglu_tasks,
            geometry.shared_down_tasks,
            geometry.shared_tasks,
            geometry.minibatch_routed_gate_up_tasks,
            geometry.minibatch_routed_swiglu_tasks,
            geometry.minibatch_routed_down_tasks,
            geometry.minibatch_tasks,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
        ).launch(
            grid=(geometry.capacity_launch_ctas, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=_PersistentSharedStorage.size_in_bytes(),
            stream=stream,
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
        storage: _PersistentSharedStorage,
        cluster_index: Int32,
        cta_rank: Int32,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
    ) -> None:
        thread = cute.arch.thread_idx()[0]
        if thread == Int32(0):
            cute.arch.mbarrier_init(storage.dispatch_mbarrier.ptr, 1)
            for stage in cutlass.range_constexpr(COMBINE_PIPE_DEPTH):
                cute.arch.mbarrier_init(
                    storage.combine_mbarriers.data_ptr() + Int32(stage),
                    1,
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        num_tokens = num_tokens_tensor[0]
        num_macrobatches = _ceil_div_i32(
            num_tokens,
            Int32(macrobatch_size),
        )
        if num_macrobatches <= Int32(0):
            return
        comm_cta = cluster_index * Int32(CLUSTER_SIZE) + cta_rank
        last_macro = num_macrobatches - Int32(1)
        dispatch_phase = Int32(0)
        combine_phase_bits = Int32(0)
        arena = storage.arena.data_ptr()

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
                storage.dispatch_mbarrier.ptr,
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
                        storage.combine_mbarriers.data_ptr(),
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
                        storage.dispatch_mbarrier.ptr,
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
        shared_swiglu_tasks: cutlass.Constexpr,
        shared_down_tasks: cutlass.Constexpr,
        shared_tasks: cutlass.Constexpr,
        minibatch_gate_up_tasks: cutlass.Constexpr,
        minibatch_swiglu_tasks: cutlass.Constexpr,
        minibatch_down_tasks: cutlass.Constexpr,
        minibatch_tasks: cutlass.Constexpr,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
    ) -> tuple[Int32, Int32, Int32, Int32]:
        # kind codes: shared Gate/Up/SwiGLU/Down=0..3, routed=4..7.
        if compute_cluster < Int32(shared_gate_up_tasks):
            return Int32(0), compute_cluster, Int32(-1), Int32(-1)
        if compute_cluster < Int32(2 * shared_gate_up_tasks):
            return (
                Int32(1),
                compute_cluster - Int32(shared_gate_up_tasks),
                Int32(-1),
                Int32(-1),
            )
        if compute_cluster < Int32(2 * shared_gate_up_tasks + shared_swiglu_tasks):
            return (
                Int32(2),
                compute_cluster - Int32(2 * shared_gate_up_tasks),
                Int32(-1),
                Int32(-1),
            )
        if compute_cluster < Int32(shared_tasks):
            return (
                Int32(3),
                compute_cluster
                - Int32(2 * shared_gate_up_tasks + shared_swiglu_tasks),
                Int32(-1),
                Int32(-1),
            )

        routed = compute_cluster - Int32(shared_tasks)
        ordered_minibatch = routed // Int32(minibatch_tasks)
        minibatch_task = routed - ordered_minibatch * Int32(minibatch_tasks)
        num_macrobatches = _ceil_div_i32(num_tokens, Int32(macrobatch_size))
        true_minibatches = _ceil_div_i32(num_tokens, Int32(minibatch_size))
        minibatches_per_macro = Int32(macrobatch_size // minibatch_size)
        last_macro_minibatches = true_minibatches - (
            num_macrobatches - Int32(1)
        ) * minibatches_per_macro
        if ordered_minibatch < last_macro_minibatches:
            macrobatch = num_macrobatches - Int32(1)
            minibatch = ordered_minibatch
        else:
            index = ordered_minibatch - last_macro_minibatches
            macrobatch = (
                num_macrobatches - Int32(2) - index // minibatches_per_macro
            )
            minibatch = index % minibatches_per_macro

        if minibatch_task < Int32(minibatch_gate_up_tasks):
            return Int32(4), minibatch_task, macrobatch, minibatch
        if minibatch_task < Int32(2 * minibatch_gate_up_tasks):
            return (
                Int32(5),
                minibatch_task - Int32(minibatch_gate_up_tasks),
                macrobatch,
                minibatch,
            )
        if minibatch_task < Int32(
            2 * minibatch_gate_up_tasks + minibatch_swiglu_tasks
        ):
            return (
                Int32(6),
                minibatch_task - Int32(2 * minibatch_gate_up_tasks),
                macrobatch,
                minibatch,
            )
        return (
            Int32(7),
            minibatch_task
            - Int32(2 * minibatch_gate_up_tasks + minibatch_swiglu_tasks),
            macrobatch,
            minibatch,
        )

    @cute.kernel
    def kernel(
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
        compute_task_log: cute.Tensor,
        scheduler_params: utils.ClcDynamicPersistentTileSchedulerParams,
        comm_clusters: cutlass.Constexpr,
        shared_gate_up_tasks: cutlass.Constexpr,
        shared_swiglu_tasks: cutlass.Constexpr,
        shared_down_tasks: cutlass.Constexpr,
        shared_tasks: cutlass.Constexpr,
        minibatch_gate_up_tasks: cutlass.Constexpr,
        minibatch_swiglu_tasks: cutlass.Constexpr,
        minibatch_down_tasks: cutlass.Constexpr,
        minibatch_tasks: cutlass.Constexpr,
        macrobatch_size: cutlass.Constexpr,
        minibatch_size: cutlass.Constexpr,
        num_comm_sms: cutlass.Constexpr,
    ):
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_index = cute.arch.make_warp_uniform(cute.arch.cluster_idx()[0])
        num_tokens = num_tokens_tensor[0]
        true_minibatches = _ceil_div_i32(num_tokens, Int32(minibatch_size))
        true_clusters = (
            Int32(comm_clusters + shared_tasks)
            + true_minibatches * Int32(minibatch_tasks)
        )
        if cluster_index >= true_clusters:
            return

        smem = utils.SmemAllocator()
        storage = smem.allocate(_PersistentSharedStorage)
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
                storage,
                cluster_index,
                cta_rank,
                macrobatch_size,
                minibatch_size,
                num_comm_sms,
            )
            return

        cluster_layout_vmnk = cute.make_layout((1, CLUSTER_SIZE, 1, 1))
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbarriers.data_ptr(),
            num_stages=CLC_PIPE_DEPTH,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                CLC_COMPLETION_WARPS * 32,
            ),
            tx_count=CLC_RESPONSE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_drain_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_drain_mbarriers.data_ptr(),
            num_stages=CLC_DRAIN_WARPS,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            # One lane from each physical CTA consumes a response in each
            # warp-owned drain slot.
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                CLUSTER_SIZE,
            ),
            tx_count=CLC_RESPONSE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            CLC_PIPE_DEPTH,
        )
        clc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.ProducerConsumer,
            CLC_PIPE_DEPTH,
        )
        drain_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            CLC_DRAIN_WARPS,
        )
        drain_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.ProducerConsumer,
            CLC_DRAIN_WARPS,
        )
        # Give every warp a stable, independent stage.  Advancing by all eight
        # stages after each response toggles only that slot's phase.
        drain_stage = Int32(0)
        while drain_stage < warp:
            drain_consumer_state.advance()
            drain_producer_state.advance()
            drain_stage = drain_stage + Int32(1)
        pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE)

        scheduler = utils.ClcDynamicPersistentTileScheduler.create(
            scheduler_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_response.data_ptr(),
        )
        drain_scheduler = utils.ClcDynamicPersistentTileScheduler.create(
            scheduler_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_drain_responses.data_ptr() + warp * Int32(4),
        )
        work = scheduler.initial_work_tile_info()
        logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
        active_work = Int32(0)
        if work.is_valid_tile:
            if logical_cluster < true_clusters:
                active_work = Int32(1)

        while active_work != Int32(0):
            if warp == Int32(SCHEDULER_WARP):
                if cta_rank == Int32(0):
                    # PipelineClcFetchAsync uses lanes 0..cluster_size-1 to
                    # arm the destination CTA barriers; the scheduler itself
                    # elects one issuing lane.
                    clc_pipeline.producer_acquire(clc_producer_state)
                    scheduler.advance_to_next_work(
                        clc_pipeline.producer_get_barrier(clc_producer_state)
                    )
                    clc_producer_state.advance()

            compute_cluster = logical_cluster - Int32(comm_clusters)
            kind, task, macrobatch, minibatch = self._decode_compute_task(
                compute_cluster,
                num_tokens,
                shared_gate_up_tasks,
                shared_swiglu_tasks,
                shared_down_tasks,
                shared_tasks,
                minibatch_gate_up_tasks,
                minibatch_swiglu_tasks,
                minibatch_down_tasks,
                minibatch_tasks,
                macrobatch_size,
                minibatch_size,
            )
            if thread == Int32(0):
                # The log is the compile/smoke seam for the task decoder.  It
                # is not an implementation of Gate/Up/SwiGLU/Down.
                log_index = logical_cluster * Int32(CLUSTER_SIZE) + cta_rank
                compute_task_log[log_index] = (
                    kind * Int32(100000000)
                    + (macrobatch + Int32(1)) * Int32(1000000)
                    + (minibatch + Int32(1)) * Int32(10000)
                    + task
                )

            current_is_local = Int32(0)
            if kind == Int32(2):
                current_is_local = Int32(1)
            elif kind == Int32(6):
                current_is_local = Int32(1)
            clc_pipeline.consumer_wait(clc_consumer_state)
            work = scheduler.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
            next_logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            next_is_local = Int32(0)
            next_is_active = Int32(0)
            if work.is_valid_tile:
                if next_logical_cluster < true_clusters:
                    next_is_active = Int32(1)
                    next_compute = next_logical_cluster - Int32(comm_clusters)
                    next_kind, _, _, _ = self._decode_compute_task(
                        next_compute,
                        num_tokens,
                        shared_gate_up_tasks,
                        shared_swiglu_tasks,
                        shared_down_tasks,
                        shared_tasks,
                        minibatch_gate_up_tasks,
                        minibatch_swiglu_tasks,
                        minibatch_down_tasks,
                        minibatch_tasks,
                        macrobatch_size,
                        minibatch_size,
                    )
                    if next_kind == Int32(2):
                        next_is_local = Int32(1)
                    elif next_kind == Int32(6):
                        next_is_local = Int32(1)
            logical_cluster = next_logical_cluster

            # Do not insert a per-task CTA/cluster barrier: CUDA permits the
            # previous epilogue store to overlap the next task's A/B load and
            # MMA.  Only a CTA-local SwiGLU -> cooperative GEMM transition
            # needs a cluster-wide rendezvous.
            if current_is_local != Int32(0):
                if next_is_active != Int32(0):
                    if next_is_local == Int32(0):
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()

            active_work = next_is_active

        if warp == Int32(SCHEDULER_WARP):
            if cta_rank == Int32(0):
                clc_pipeline.producer_tail(clc_producer_state)

        # Match CUDA's post-loop cluster rendezvous before no-op CLC drain.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        needs_drain = Int32(0)
        if work.is_valid_tile:
            if logical_cluster >= true_clusters:
                needs_drain = Int32(1)

        if needs_drain != Int32(0):
            drain_work = work
            while drain_work.is_valid_tile:
                # Every CTA-0 warp owns one response slot.  All lanes call
                # producer_acquire because lanes 0 and 1 arm the two CTAs;
                # advance_to_next_work elects the issuing lane internally.
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
                # Keep the response stable until every lane has performed the
                # generic-proxy read used as the warp-uniform loop condition.
                cute.arch.sync_warp()
                if cute.arch.lane_idx() == Int32(0):
                    clc_drain_pipeline.consumer_release(drain_consumer_state)

                for _ in cutlass.range_constexpr(CLC_DRAIN_WARPS):
                    drain_consumer_state.advance()
                    drain_producer_state.advance()


_PERSISTENT_FORWARD_BRINGUP = _PersistentForwardBringupKernel()


def _check_bf16_tensor(
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


def _run_shared_gate_tile_bf16(
    A: torch.Tensor,
    W_nk: torch.Tensor,
    out: torch.Tensor,
    ready: torch.Tensor,
) -> None:
    """Cold-JIT bring-up entry for the private Shared Gate tile collective."""

    device = A.device
    if not A.is_cuda:
        raise RuntimeError("Shared Gate tile bring-up requires CUDA")
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("Shared Gate tile bring-up requires B300/SM103")
    _check_bf16_tensor(
        "A",
        A,
        (_GATE_TILE_M, _GATE_TILE_K),
        device,
    )
    _check_bf16_tensor(
        "W_nk",
        W_nk,
        (_GATE_TILE_N, _GATE_TILE_K),
        device,
    )
    _check_bf16_tensor(
        "out",
        out,
        (_GATE_TILE_M, _GATE_TILE_N),
        device,
    )
    if (
        ready.device != device
        or not ready.is_cuda
        or ready.dtype != torch.int32
        or not ready.is_contiguous()
        or tuple(ready.shape) != (1,)
    ):
        raise ValueError("ready must be contiguous int32 [1] on A.device")

    cute_A = from_dlpack(A, assumed_align=16)
    cute_W = from_dlpack(W_nk, assumed_align=16)
    cute_out = from_dlpack(out, assumed_align=16)
    cute_ready = from_dlpack(ready, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    _SHARED_GATE_TILE_BRINGUP(
        cute_A,
        cute_W,
        cute_out,
        cute_ready,
        stream,
    )


def prepare_persistent_forward_bf16(
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
    num_comm_sms: int,
) -> PersistentForwardState:
    """Validate and allocate the outer CUDA BF16 ABI without launching.

    This seam intentionally keeps ``schedule.num_tokens`` device resident.
    The caller must continue using the existing functional CuTe or CUDA
    backend until the four compute collectives are connected to the private
    persistent core.
    """

    del swiglu_limit
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
    geometry = make_forward_geometry(
        num_local_tokens=workspace.num_local_tokens,
        schedule_capacity=workspace.schedule_capacity,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
    )
    device = workspace.device
    weight_shapes = (
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
    _check_bf16_tensor(
        "workspace.x_buffer",
        workspace.x_buffer,
        (workspace.num_local_tokens, HIDDEN_SIZE),
        device,
    )
    for name, tensor, shape in weight_shapes:
        _check_bf16_tensor(name, tensor, shape, device)

    schedule_shapes = (
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
    for name, tensor, shape in schedule_shapes:
        if (
            tensor.device != device
            or not tensor.is_cuda
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"{name} must be contiguous int32 {shape} on {device}")

    output_shapes = nine_output_shapes(
        workspace.num_local_tokens,
        macrobatch_size,
    )
    outputs = tuple(
        torch.empty(shape, dtype=torch.bfloat16, device=device)
        for shape in output_shapes
    )
    counter_lengths = geometry.counters
    counters = tuple(
        torch.zeros(length, dtype=torch.int32, device=device)
        for length in (
            counter_lengths.gate_up_tile_ready,
            counter_lengths.hidden_row_block_ready,
            counter_lengths.x_routed_ready,
            counter_lengths.y_routed_ready,
            counter_lengths.y_routed_done,
        )
    )
    return PersistentForwardState(*outputs, *counters)
