"""Private SM103 cluster-2 BF16 Gate/Up FC1 compile slice.

This is the smallest device body that exercises the CUDA fused-FC1 topology:

* one logical M256 x N128 x K4096 Gate/Up tile;
* CTA 0 loads A[0:128] plus a Gate N128 tile;
* CTA 1 loads A[128:256] plus a separate Up N128 tile;
* the 2CTA tcgen05 MMA produces local M128 x [Gate128 | Up128] TMEM;
* the epilogue performs the TMEM -> register -> BF16 SMEM handoff and stores
  the raw packed accumulator to a M256 x N256 probe tensor.

There is no packed weight mirror and no cross-CTA A multicast.  SwiGLU, Down,
communication, readiness counters, CLC scheduling, and persistent looping are
N/A here.  The module is private and is never selected by ``mok.functional``.
"""

from __future__ import annotations

import importlib.metadata as metadata

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Boolean, Float32, Int32
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


_REQUIRED_CUTLASS_DSL = "4.6.2"
_REQUIRED_QUACK = "0.6.4"
if metadata.version("nvidia-cutlass-dsl") != _REQUIRED_CUTLASS_DSL:
    raise RuntimeError("the BF16 FC1 slice requires nvidia-cutlass-dsl==4.6.2")
if quack.__version__ != _REQUIRED_QUACK:
    raise RuntimeError("the BF16 FC1 slice requires quack-kernels==0.6.4")

TILE_M = 256
LOGICAL_N = 128
PACKED_N = 256
TILE_K = 4096
CTA_M = 128
K_TILE = 64
K_TILES = TILE_K // K_TILE
CLUSTER_SHAPE = (2, 1, 1)
THREADS = 256

AB_STAGES = 4
ACC_STAGES = 1
D_STAGES = 3
EPILOGUE_N = 32
EPILOGUE_SUBTILES = PACKED_N // EPILOGUE_N
MMA_WARP = 4
IDLE_WARPS = (5, 6)
TMA_WARP = 7

A_BYTES = AB_STAGES * CTA_M * K_TILE * 2
B_BYTES = AB_STAGES * LOGICAL_N * K_TILE * 2
D_BYTES = D_STAGES * CTA_M * EPILOGUE_N * 2
ARENA_BYTES = A_BYTES + B_BYTES + D_BYTES

@cute.struct
class _Storage:
    ab_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * AB_STAGES]
    acc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * ACC_STAGES]
    tmem_dealloc_mbarrier: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32
    arena: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, ARENA_BYTES],
        1024,
    ]


class _Fc1Collective(GemmDefaultSm100):
    """QuACK collective stripped to one raw Gate/Up tile."""

    @classmethod
    def _compute_stages(cls, *args, **kwargs):
        del args, kwargs
        return ACC_STAGES, AB_STAGES, D_STAGES, 0

    def __init__(self) -> None:
        super().__init__(
            acc_dtype=Float32,
            a_dtype=BFloat16,
            # N=256 is the raw Gate+Up accumulator.  Each input weight tensor
            # supplies one N=128 V-half to the 2CTA instruction.
            mma_tiler_mnk=(TILE_M, PACKED_N, K_TILE),
            cluster_shape_mnk=CLUSTER_SHAPE,
            use_clc_persistence=False,
            use_pdl=False,
        )
        self.mma_warp_id = MMA_WARP
        self.scheduler_warp_id = IDLE_WARPS[0]
        self.epi_load_warp_id = IDLE_WARPS[1]
        self.ab_load_warp_id = TMA_WARP
        assert self.epilog_warp_id == (0, 1, 2, 3)
        assert self.threads_per_cta == THREADS

    def configure(
        self,
        mA_mk: cute.Tensor,
        mB_gate_nk: cute.Tensor,
        mD_mn: cute.Tensor,
    ) -> None:
        self.a_dtype = mA_mk.element_type
        self.b_dtype = mB_gate_nk.element_type
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
        self.b_layout = LayoutEnum.from_tensor(mB_gate_nk)
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
        assert self.cta_tile_shape_mnk == (CTA_M, PACKED_N, K_TILE)
        assert cute.cosize(self.a_smem_layout_staged.outer) == (
            AB_STAGES * CTA_M * K_TILE
        )
        assert cute.cosize(self.b_smem_layout_staged.outer) == (
            AB_STAGES * LOGICAL_N * K_TILE
        )


class _Fc1Slice:
    def __init__(self) -> None:
        self.collective = _Fc1Collective()

    @cute.jit
    def __call__(
        self,
        mA_mk: cute.Tensor,
        mB_gate_nk: cute.Tensor,
        mB_up_nk: cute.Tensor,
        mD_mn: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        collective = self.collective
        collective.configure(mA_mk, mB_gate_nk, mD_mn)

        a_smem_layout = cute.slice_(
            collective.a_smem_layout_staged, (None, None, None, 0)
        )
        b_smem_layout = cute.slice_(
            collective.b_smem_layout_staged, (None, None, None, 0)
        )

        # Use the CUTLASS DSL 4.6.2 MMA-aware helpers.  The QuACK staged SMEM
        # slice has three real modes (atom, rest_M/N, rest_K); the generic TMA
        # helper with a rank-2 tiler would mistake rest_K for a stage and drop
        # it.  These helpers construct the complete A/B cta_v_map while the
        # non-multicast operation keeps each descriptor local to one CTA.
        tma_op = cpasync.CopyBulkTensorTileG2SOp(collective.cta_group)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op,
            mA_mk,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        tma_atom_b_gate, tma_tensor_b_gate = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op,
            mB_gate_nk,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        tma_atom_b_up, tma_tensor_b_up = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op,
            mB_up_nk,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        tma_atom_d, tma_tensor_d = collective._make_tma_epi_atoms_and_tensors(
            mD_mn,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            op_type="store",
        )

        # The leader transaction barrier observes A+B from both CTAs.
        collective.num_tma_load_bytes = (
            cute.size_in_bytes(BFloat16, a_smem_layout)
            + cute.size_in_bytes(BFloat16, b_smem_layout)
        ) * 2

        self.kernel(
            collective.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b_gate,
            tma_tensor_b_gate,
            tma_atom_b_up,
            tma_tensor_b_up,
            tma_atom_d,
            tma_tensor_d,
            collective.cluster_layout_vmnk,
            collective.a_smem_layout_staged,
            collective.b_smem_layout_staged,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
        ).launch(
            grid=(2, 1, 1),
            block=(THREADS, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=_Storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b_gate: cute.CopyAtom,
        mB_gate_nk: cute.Tensor,
        tma_atom_b_up: cute.CopyAtom,
        mB_up_nk: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mn: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ) -> None:
        collective = self.collective
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = cta_rank == Int32(0)
        is_two_cta = cute.size(tiled_mma.thr_id.shape) == 2

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Storage)
        arena = storage.arena.data_ptr().align(min_align=1024)
        sA = cute.make_tensor(
            cute.recast_ptr(arena, a_smem_layout.inner, dtype=BFloat16),
            a_smem_layout.outer,
        )
        sB = cute.make_tensor(
            cute.recast_ptr(arena + A_BYTES, b_smem_layout.inner, dtype=BFloat16),
            b_smem_layout.outer,
        )
        sD = cute.make_tensor(
            cute.recast_ptr(
                arena + A_BYTES + B_BYTES,
                epi_smem_layout.inner,
                dtype=BFloat16,
            ),
            epi_smem_layout.outer,
        )

        ab_pipeline = PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                collective.num_mcast_ctas_a + collective.num_mcast_ctas_b - 1,
            ),
            tx_count=collective.num_tma_load_bytes,
            barrier_storage=storage.ab_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = PipelineUmmaAsync.create(
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(collective.epilog_warp_id) * 2,
            ),
            barrier_storage=storage.acc_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=(len(collective.epilog_warp_id) + 1) * cute.arch.WARP_SIZE,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=collective.epilog_warp_id[0],
            is_two_cta=is_two_cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbarrier.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # The MMA-aware TMA tensors are partitioned exactly as their consumers.
        # Current-rank partition_A gives CTA0 A[0:128] and CTA1 A[128:256].
        thr_mma = tiled_mma.get_slice(cta_rank)
        gA_cluster = cute.local_tile(
            mA_mk,
            cute.select(collective.mma_tiler, [0, 2]),
            (Int32(0), None),
        )
        gA = thr_mma.partition_A(gA_cluster)

        # Gate and Up are separate N128 tensors rather than the two halves of
        # one packed N256 tensor.  Both descriptors therefore use the rank-0 B
        # V-fragment (their first N128 rows); issuing the Up descriptor from
        # CTA1 places that fragment in CTA1-local sB, which the 2CTA MMA exposes
        # as the Up half.  There is no B multicast or packed mirror.
        b_rank0 = tiled_mma.get_slice(Int32(0))
        gB_gate_cluster = cute.local_tile(
            mB_gate_nk,
            cute.select(collective.mma_tiler, [1, 2]),
            (Int32(0), None),
        )
        gB_up_cluster = cute.local_tile(
            mB_up_nk,
            cute.select(collective.mma_tiler, [1, 2]),
            (Int32(0), None),
        )
        gB_gate = b_rank0.partition_B(gB_gate_cluster)
        gB_up = b_rank0.partition_B(gB_up_cluster)
        copy_A = quack_copy_utils.tma_get_block_copy_fn(
            tma_atom_a, src_tensor=gA, dst_tensor=sA
        )
        copy_B_gate = quack_copy_utils.tma_get_block_copy_fn(
            tma_atom_b_gate, src_tensor=gB_gate, dst_tensor=sB
        )
        copy_B_up = quack_copy_utils.tma_get_block_copy_fn(
            tma_atom_b_up, src_tensor=gB_up, dst_tensor=sB
        )

        tile_coord_mnkl = (cta_rank, Int32(0), Int32(0), Int32(0))
        copy_D, _, _ = collective.epilog_gmem_copy_and_partition(
            tma_atom_d,
            mD_mn,
            collective.cta_tile_shape_mnk[:2],
            epi_tile,
            sD,
            tile_coord_mnkl,
        )

        acc_shape = tiled_mma.partition_shape_C(collective.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, ACC_STAGES))

        if warp == Int32(TMA_WARP):
            if cta_rank == Int32(0):
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b_gate)
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, AB_STAGES
                )
                producer_state = collective.load_tma(
                    ab_pipeline,
                    producer_state,
                    [copy_A, copy_B_gate],
                    Int32(K_TILES),
                )
                ab_pipeline.producer_tail(producer_state)
            else:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b_up)
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, AB_STAGES
                )
                producer_state = collective.load_tma(
                    ab_pipeline,
                    producer_state,
                    [copy_A, copy_B_up],
                    Int32(K_TILES),
                )
                ab_pipeline.producer_tail(producer_state)

        if warp == Int32(MMA_WARP):
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(collective.acc_dtype)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, AB_STAGES
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, ACC_STAGES
            )
            tCtAcc = tCtAcc_base[None, None, None, acc_producer_state.index]
            ab_consumer_state, acc_producer_state, _ = collective.mma(
                ab_pipeline,
                acc_pipeline,
                ab_consumer_state,
                acc_producer_state,
                tiled_mma,
                tCrA,
                tCrB,
                tCtAcc,
                Int32(K_TILES),
                is_leader_cta,
                cta_rank,
            )
            tmem_alloc_barrier.arrive()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp < Int32(MMA_WARP):
            tmem.allocate(collective.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(collective.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tiled_copy_t2r, tTR_tAcc, tTR_rAcc = (
                collective.epilog_tmem_copy_and_partition(
                    thread, tCtAcc_base, epi_tile, is_two_cta
                )
            )
            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, collective.acc_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = (
                collective.epilog_smem_store_and_partition(
                    tiled_copy_t2r,
                    collective.d_layout,
                    collective.d_dtype,
                    tTR_rD,
                    sD,
                    thread,
                )
            )
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, ACC_STAGES
            )
            epi_store_pipeline = collective.make_epi_store_pipeline()
            is_tma_warp = Boolean(warp == Int32(0))
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(collective.cta_tile_shape_mnk[:2]), epi_tile
            ).shape[1]
            epi_tile_layout = cute.make_ordered_layout(
                epi_tile_shape, order=(0, 1)
            )
            tTR_tAcc = tTR_tAcc[
                None, None, None, None, None, acc_consumer_state.index
            ]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            acc_pipeline.consumer_wait(acc_consumer_state)
            for epi_idx in cutlass.range_constexpr(EPILOGUE_SUBTILES):
                epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                collective.epi_load_acc_subtile(
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tTR_tAcc,
                    tTR_rAcc,
                    tRS_rD,
                    epi_coord,
                    acc_pipeline,
                    acc_consumer_state,
                    EPILOGUE_SUBTILES - 1,
                )
                # This BF16 conversion is the CUDA preactivation precision seam.
                rD_bf16 = tRS_rD.to(BFloat16)
                if is_tma_warp:
                    epi_store_pipeline.producer_acquire()
                collective.epilogue_barrier.arrive_and_wait()
                epi_buffer = epi_idx % D_STAGES
                cute.copy(
                    tiled_copy_r2s,
                    rD_bf16,
                    tRS_sD[None, None, None, epi_buffer],
                )
                cute.arch.fence_view_async_shared()
                collective.epilogue_barrier.arrive_and_wait()
                if is_tma_warp:
                    copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                    epi_store_pipeline.producer_commit()

            acc_consumer_state.advance()
            if is_tma_warp:
                epi_store_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)


_FC1_SLICE = _Fc1Slice()


def _make_fc1_slice_args(x, gate, up, packed_output, stream):
    """Convert one validated host call to its complete CuTe argument tuple."""

    import torch

    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    cuda_stream = cuda.CUstream(stream.cuda_stream)
    return (
        from_dlpack(x, assumed_align=16),
        from_dlpack(gate, assumed_align=16),
        from_dlpack(up, assumed_align=16),
        from_dlpack(packed_output, assumed_align=16),
        cuda_stream,
    )


def compile_fc1_slice(x, gate, up, packed_output, *, stream=None):
    """Compile the exact four-tensor specialization and return its executor."""

    cute_args = _make_fc1_slice_args(x, gate, up, packed_output, stream)
    return cute.compile(_FC1_SLICE, *cute_args)


def run_fc1_slice(x, gate, up, packed_output, *, stream=None) -> None:
    """Compile and execute the exact specialization once with identical args."""

    cute_args = _make_fc1_slice_args(x, gate, up, packed_output, stream)
    executor = cute.compile(_FC1_SLICE, *cute_args)
    executor(*cute_args)


__all__ = ["compile_fc1_slice", "run_fc1_slice"]
