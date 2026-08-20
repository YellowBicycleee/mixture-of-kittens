"""CuTe DSL 4.6.2 Stage-0 risk spike for MoK's persistent megakernel.

This is intentionally *not* a MoE forward implementation.  It exercises the
four primitives that decide whether a faithful sister backend is viable:

* the eight raw symmetric pointers already stored in ``MoKWorkspace``;
* runtime peer selection followed by a TMA load from that peer;
* release/acquire counters at both system and GPU scope; and
* a Blackwell two-CTA CLC queue containing communication and compute roles.

No symbol in this module is re-exported by ``mok.__init__`` and the CUDA C++
backend remains the default path.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .contract import EP_SIZE, QWEN_HIDDEN_SIZE, ROW_TILE_ELEMENTS, validate_stage0_contract


CLUSTER_SHAPE = (2, 1, 1)
THREADS_PER_CTA = 256
SCHEDULER_WARP = 7
WORKER_THREADS = SCHEDULER_WARP * 32
CLC_STAGES = 1
TMA_STAGES = 1
CLC_RESPONSE_BYTES = 16
TMA_TRANSACTION_BYTES = ROW_TILE_ELEMENTS * (BFloat16.width // 8)


@cute.struct
class _SharedStorage:
    clc_mbarriers: cute.struct.MemRange[cutlass.Int64, CLC_STAGES * 2]
    clc_response: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, 4],
        CLC_RESPONSE_BYTES,
    ]
    tma_mbarriers: cute.struct.MemRange[cutlass.Int64, TMA_STAGES * 2]
    row_tile: cute.struct.Align[
        cute.struct.MemRange[BFloat16, ROW_TILE_ELEMENTS],
        128,
    ]


class _Stage0Kernel:
    """One compile-time-specialized Qwen BF16/EP8 Stage-0 kernel."""

    @cute.jit
    def __call__(
        self,
        peer_ptrs: list[cute.Pointer],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        output: cute.Tensor,
        system_counters: cute.Tensor,
        gpu_counters: cute.Tensor,
        role_log: cute.Tensor,
        completion_log: cute.Tensor,
        num_peer_tokens: cutlass.Constexpr,
        num_comm_clusters: cutlass.Constexpr,
        num_compute_clusters: cutlass.Constexpr,
    ):
        assert len(peer_ptrs) == EP_SIZE
        assert schedule_peer_rank.element_type is cutlass.Int32
        assert schedule_peer_token_idx.element_type is cutlass.Int32
        assert output.element_type is BFloat16

        peer_layout = cute.make_layout((num_peer_tokens * QWEN_HIDDEN_SIZE,))
        output_layout = cute.make_layout(
            (num_comm_clusters * CLUSTER_SHAPE[0] * QWEN_HIDDEN_SIZE,)
        )
        output_flat = cute.make_tensor(output.iterator, output_layout)

        smem_layout = cute.make_layout((ROW_TILE_ELEMENTS,))
        tma_load_atoms = []
        tma_load_tensors = []
        for peer_index in cutlass.range_constexpr(EP_SIZE):
            peer_tensor = cute.make_tensor(peer_ptrs[peer_index], peer_layout)
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                peer_tensor,
                smem_layout,
                (ROW_TILE_ELEMENTS,),
            )
            tma_load_atoms.append(tma_atom)
            tma_load_tensors.append(tma_tensor)

        num_logical_clusters = num_comm_clusters + num_compute_clusters
        scheduler_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (num_logical_clusters * CLUSTER_SHAPE[0], 1, 1),
            CLUSTER_SHAPE,
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(scheduler_params)

        self.kernel(
            tma_load_atoms,
            tma_load_tensors,
            schedule_peer_rank,
            schedule_peer_token_idx,
            output_flat,
            system_counters,
            gpu_counters,
            role_log,
            completion_log,
            scheduler_params,
            num_comm_clusters,
        ).launch(
            grid=grid,
            block=(THREADS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=_SharedStorage.size_in_bytes(),
        )

    @cute.jit
    def _issue_one_peer_tma(
        self,
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
        flat_tile_index: Int32,
        row_tile: cute.Tensor,
        tma_barrier: cute.Pointer,
    ):
        tiled_peer = cute.zipped_divide(tma_tensor, (ROW_TILE_ELEMENTS,))
        peer_tile = tiled_peer[(None,), flat_tile_index]
        smem_part, gmem_part = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            row_tile,
            peer_tile,
        )
        cute.copy(
            tma_atom,
            gmem_part,
            smem_part,
            tma_bar_ptr=tma_barrier,
        )

    @cute.jit
    def _issue_dynamic_peer_tma(
        self,
        peer_rank: Int32,
        flat_tile_index: Int32,
        tma_load_atoms: list[cute.CopyAtom],
        tma_load_tensors: list[cute.Tensor],
        row_tile: cute.Tensor,
        tma_barrier: cute.Pointer,
    ):
        # A runtime list index cannot select a tensor-map descriptor.  The
        # explicit ladder emits eight descriptors and selects one at runtime,
        # which is exactly the EP8 behavior this risk spike must prove.
        if peer_rank == Int32(0):
            self._issue_one_peer_tma(
                tma_load_atoms[0], tma_load_tensors[0], flat_tile_index, row_tile, tma_barrier
            )
        elif peer_rank == Int32(1):
            self._issue_one_peer_tma(
                tma_load_atoms[1], tma_load_tensors[1], flat_tile_index, row_tile, tma_barrier
            )
        elif peer_rank == Int32(2):
            self._issue_one_peer_tma(
                tma_load_atoms[2], tma_load_tensors[2], flat_tile_index, row_tile, tma_barrier
            )
        elif peer_rank == Int32(3):
            self._issue_one_peer_tma(
                tma_load_atoms[3], tma_load_tensors[3], flat_tile_index, row_tile, tma_barrier
            )
        elif peer_rank == Int32(4):
            self._issue_one_peer_tma(
                tma_load_atoms[4], tma_load_tensors[4], flat_tile_index, row_tile, tma_barrier
            )
        elif peer_rank == Int32(5):
            self._issue_one_peer_tma(
                tma_load_atoms[5], tma_load_tensors[5], flat_tile_index, row_tile, tma_barrier
            )
        elif peer_rank == Int32(6):
            self._issue_one_peer_tma(
                tma_load_atoms[6], tma_load_tensors[6], flat_tile_index, row_tile, tma_barrier
            )
        else:
            self._issue_one_peer_tma(
                tma_load_atoms[7], tma_load_tensors[7], flat_tile_index, row_tile, tma_barrier
            )

    @cute.jit
    def _pair_arrive_and_wait_system(
        self,
        counters: cute.Tensor,
        task_index: Int32,
        cta_rank: Int32,
    ):
        counter_ptr = counters.iterator + task_index
        cute.arch.atomic_add(
            counter_ptr,
            Int32(1),
            sem="release",
            scope="sys",
        )
        if cta_rank == Int32(0):
            observed = Int32(0)
            while observed < Int32(CLUSTER_SHAPE[0]):
                observed = cute.arch.load(
                    counter_ptr,
                    Int32,
                    sem="acquire",
                    scope="sys",
                )
                if observed < Int32(CLUSTER_SHAPE[0]):
                    cute.arch.nanosleep(sleep_time=16)

    @cute.jit
    def _pair_arrive_and_wait_gpu(
        self,
        counters: cute.Tensor,
        task_index: Int32,
        cta_rank: Int32,
    ):
        counter_ptr = counters.iterator + task_index
        cute.arch.atomic_add(
            counter_ptr,
            Int32(1),
            sem="release",
            scope="gpu",
        )
        if cta_rank == Int32(0):
            observed = Int32(0)
            while observed < Int32(CLUSTER_SHAPE[0]):
                observed = cute.arch.load(
                    counter_ptr,
                    Int32,
                    sem="acquire",
                    scope="gpu",
                )
                if observed < Int32(CLUSTER_SHAPE[0]):
                    cute.arch.nanosleep(sleep_time=16)

    @cute.kernel
    def kernel(
        self,
        tma_load_atoms: list[cute.CopyAtom],
        tma_load_tensors: list[cute.Tensor],
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        output_flat: cute.Tensor,
        system_counters: cute.Tensor,
        gpu_counters: cute.Tensor,
        role_log: cute.Tensor,
        completion_log: cute.Tensor,
        scheduler_params: utils.ClcDynamicPersistentTileSchedulerParams,
        num_comm_clusters: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        smem = utils.SmemAllocator()
        storage = smem.allocate(_SharedStorage)
        row_tile = storage.row_tile.get_tensor(cute.make_layout((ROW_TILE_ELEMENTS,)))

        tma_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.tma_mbarriers.data_ptr(),
            num_stages=TMA_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, WORKER_THREADS
            ),
            tx_count=TMA_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        # PipelineClcFetchAsync names this layout VMNK.  Stage 0 has one MMA
        # value group and two physical CTAs along the scheduler's M mode.
        cta_layout_vmnk = cute.make_layout((1, CLUSTER_SHAPE[0], 1, 1))
        # Two CTAs x seven worker warps, plus the scheduler warp in CTA 0.
        clc_consumer_threads = CLUSTER_SHAPE[0] * WORKER_THREADS + 32
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbarriers.data_ptr(),
            num_stages=CLC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, clc_consumer_threads
            ),
            tx_count=CLC_RESPONSE_BYTES,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, CLC_STAGES
        )

        pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE)

        scheduler = utils.ClcDynamicPersistentTileScheduler.create(
            scheduler_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_response.data_ptr(),
        )
        work_tile = scheduler.initial_work_tile_info()

        # Only CTA 0's last warp issues CLC queries.  It also consumes every
        # response, matching PipelineClcFetchAsync's documented participant
        # count and keeping the producer from reusing a live response slot.
        if warp_idx == SCHEDULER_WARP and cta_rank == Int32(0):
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, CLC_STAGES
            )
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                clc_barrier = clc_pipeline.producer_get_barrier(clc_producer_state)
                scheduler.advance_to_next_work(clc_barrier)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = scheduler.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        # Seven warps in each CTA consume the same cluster task.  Communication
        # tasks give one routed row to each CTA; compute tasks retain the two-CTA
        # pairing and prove the GPU-scope producer/consumer counter skeleton.
        if warp_idx < SCHEDULER_WARP:
            tma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, TMA_STAGES
            )
            tma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, TMA_STAGES
            )
            worker_barrier = pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=WORKER_THREADS,
            )

            while work_tile.is_valid_tile:
                cta_tile_index = work_tile.tile_idx[0]
                logical_task_index = cta_tile_index // Int32(CLUSTER_SHAPE[0])
                logical_cta_rank = cta_tile_index - (
                    logical_task_index * Int32(CLUSTER_SHAPE[0])
                )

                if logical_task_index < Int32(num_comm_clusters):
                    schedule_index = (
                        logical_task_index * Int32(CLUSTER_SHAPE[0])
                        + logical_cta_rank
                    )
                    peer_rank = schedule_peer_rank[schedule_index]
                    peer_token_index = schedule_peer_token_idx[schedule_index]

                    for column_tile in cutlass.range_constexpr(
                        QWEN_HIDDEN_SIZE // ROW_TILE_ELEMENTS
                    ):
                        if warp_idx == 0:
                            tma_pipeline.producer_acquire(tma_producer_state)
                            peer_flat_tile = (
                                peer_token_index
                                * Int32(QWEN_HIDDEN_SIZE // ROW_TILE_ELEMENTS)
                                + Int32(column_tile)
                            )
                            self._issue_dynamic_peer_tma(
                                peer_rank,
                                peer_flat_tile,
                                tma_load_atoms,
                                tma_load_tensors,
                                row_tile,
                                tma_pipeline.producer_get_barrier(tma_producer_state),
                            )
                            tma_pipeline.producer_commit(tma_producer_state)
                            tma_producer_state.advance()

                        tma_pipeline.consumer_wait(tma_consumer_state)
                        for element_round in cutlass.range_constexpr(
                            (ROW_TILE_ELEMENTS + WORKER_THREADS - 1) // WORKER_THREADS
                        ):
                            tile_element = tidx + Int32(element_round * WORKER_THREADS)
                            if tile_element < Int32(ROW_TILE_ELEMENTS):
                                output_offset = (
                                    schedule_index * Int32(QWEN_HIDDEN_SIZE)
                                    + Int32(column_tile * ROW_TILE_ELEMENTS)
                                    + tile_element
                                )
                                output_flat[output_offset] = row_tile[tile_element]

                        # As in the official 4.6.2 distributed TMA example, every
                        # async consumer arrives on the empty barrier.
                        tma_pipeline.sync_object_empty.arrive(
                            tma_consumer_state.index,
                            tma_pipeline.consumer_mask,
                        )
                        tma_consumer_state.advance()

                    worker_barrier.arrive_and_wait()
                    if tidx == Int32(0):
                        role_log[logical_task_index * Int32(2) + logical_cta_rank] = (
                            Int32(100) + logical_cta_rank
                        )
                        self._pair_arrive_and_wait_system(
                            system_counters,
                            logical_task_index,
                            logical_cta_rank,
                        )
                        if logical_cta_rank == Int32(0):
                            completion_log[logical_task_index] = Int32(1)
                    worker_barrier.arrive_and_wait()
                else:
                    if tidx == Int32(0):
                        role_log[logical_task_index * Int32(2) + logical_cta_rank] = (
                            Int32(200) + logical_cta_rank
                        )
                        self._pair_arrive_and_wait_gpu(
                            gpu_counters,
                            logical_task_index,
                            logical_cta_rank,
                        )
                        if logical_cta_rank == Int32(0):
                            completion_log[logical_task_index] = Int32(2)
                    worker_barrier.arrive_and_wait()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = scheduler.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            if warp_idx == 0:
                tma_pipeline.producer_tail(tma_producer_state)


_STAGE0_KERNEL = _Stage0Kernel()


def run_stage0(
    workspace,
    schedule_peer_rank,
    schedule_peer_token_idx,
    output,
    system_counters,
    gpu_counters,
    role_log,
    completion_log,
    *,
    num_compute_clusters: int = 2,
) -> None:
    """Launch Stage 0 using an existing :class:`mok.functional.MoKWorkspace`.

    The runner intentionally accepts the workspace object by duck typing so
    importing this experimental module does not couple it to the CUDA backend.
    Counters must be zero and log tensors must be initialized by the caller.
    """

    ranks = schedule_peer_rank.detach().cpu().tolist()
    token_indices = schedule_peer_token_idx.detach().cpu().tolist()
    validate_stage0_contract(
        workspace.x_buffer_ptrs,
        ranks,
        token_indices,
        num_peer_tokens=workspace.num_local_tokens,
        hidden_size=workspace.hidden_size,
    )
    if workspace.ep_size != EP_SIZE:
        raise ValueError(f"Stage 0 requires EP{EP_SIZE}; got EP{workspace.ep_size}")
    if num_compute_clusters <= 0:
        raise ValueError("num_compute_clusters must be positive")

    schedule_rows = len(ranks)
    num_comm_clusters = schedule_rows // CLUSTER_SHAPE[0]
    num_logical_clusters = num_comm_clusters + num_compute_clusters
    expected_shapes = {
        "schedule_peer_rank": (schedule_rows,),
        "schedule_peer_token_idx": (schedule_rows,),
        "output": (schedule_rows, QWEN_HIDDEN_SIZE),
        "system_counters": (num_logical_clusters,),
        "gpu_counters": (num_logical_clusters,),
        "role_log": (num_logical_clusters * CLUSTER_SHAPE[0],),
        "completion_log": (num_logical_clusters,),
    }
    tensors = {
        "schedule_peer_rank": schedule_peer_rank,
        "schedule_peer_token_idx": schedule_peer_token_idx,
        "output": output,
        "system_counters": system_counters,
        "gpu_counters": gpu_counters,
        "role_log": role_log,
        "completion_log": completion_log,
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"{name} must have shape {expected_shapes[name]}; got {tuple(tensor.shape)}"
            )
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")
        if tensor.device != workspace.device:
            raise ValueError(f"{name} must be on workspace device {workspace.device}")
    if output.dtype != workspace.x_buffer.dtype:
        raise TypeError("output must use the workspace BF16 activation dtype")
    for name in (
        "schedule_peer_rank",
        "schedule_peer_token_idx",
        "system_counters",
        "gpu_counters",
        "role_log",
        "completion_log",
    ):
        if tensors[name].dtype != workspace.barrier_buffer.dtype:
            raise TypeError(f"{name} must use the workspace int32 barrier dtype")

    peer_ptrs = [
        make_ptr(
            BFloat16,
            int(pointer),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for pointer in workspace.x_buffer_ptrs
    ]
    _STAGE0_KERNEL(
        peer_ptrs,
        from_dlpack(schedule_peer_rank, assumed_align=16),
        from_dlpack(schedule_peer_token_idx, assumed_align=16),
        from_dlpack(output, assumed_align=16),
        from_dlpack(system_counters, assumed_align=16),
        from_dlpack(gpu_counters, assumed_align=16),
        from_dlpack(role_log, assumed_align=16),
        from_dlpack(completion_log, assumed_align=16),
        workspace.num_local_tokens,
        num_comm_clusters,
        num_compute_clusters,
    )
