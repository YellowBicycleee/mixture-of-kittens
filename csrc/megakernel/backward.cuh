struct minibatch_expert_offsets_config {
    static constexpr int NUM_THREADS = WARP_THREADS;
};

struct minibatch_expert_offsets_globals {
    index_gl minibatch_expert_offsets;
    index_gl completed_segment_offsets;
    index_gl expert_row_offsets;
    index_gl completed_segment_experts;
    index_gl minibatch_task_owner_buckets;
    index_gl num_tokens;
    index_gl tokens_per_expert;
    int num_local_experts;
    int macrobatch_size;
    int minibatch_size;
    int capacity_minibatches;
    int completed_segment_capacity;
    int minibatch_task_owner_bucket_capacity;
    int minibatch_routed_bwd_tasks;
    int minibatch_task_owner_bucket_width;
    int segment_wgrad_tasks;
};

// Builds compact metadata without changing the public route schedule:
//  * minibatch_expert_offsets counts (global minibatch, active expert) pairs
//    and supplies the exact per-minibatch source-read arrival count.
//  * completed_segment_offsets counts (macrobatch, active expert) segments by
//    the global minibatch containing each segment's final row.  Segment Wgrad
//    is placed immediately after that minibatch's replay/backward work.
//  * The EP8 full-context specialization also records expert row offsets,
//    completed-segment ordinal -> expert, and a coarse routed-task owner table.
//    Its power-of-two bucket width is at most one minibatch's fixed BWD task
//    count.  Every minibatch task block is at least that wide, so a bucket
//    crosses at most one block boundary and the hot decoder needs one shift
//    and one prefix comparison.
// This is a device function passed to kittens::py::global_kernel because
// backward.cuh is included inside the dispatch_mlp_swiglu_combiner class
// template (a raw __global__ member function is not legal CUDA C++).  Capacity
// tail entries are filled with the final counts for diagnostics.
template <bool BUILD_EP8_O1_DECODE = false,
          bool BUILD_EP8_WGRAD_EXPERT_ROW_PREFIX = false>
static __device__ __forceinline__ void build_minibatch_expert_offsets_kernel(
    const minibatch_expert_offsets_globals &g
) {
    static_assert(!BUILD_EP8_WGRAD_EXPERT_ROW_PREFIX || BUILD_EP8_O1_DECODE);
    if (threadIdx.x != 0) return;

    const int routed_rows = g.num_tokens[{0}];
    if (routed_rows < 0 || g.minibatch_size <= 0 ||
        g.macrobatch_size <= 0 || g.macrobatch_size % g.minibatch_size != 0)
        asm volatile("{trap;}");
    const int num_minibatches =
        (routed_rows + g.minibatch_size - 1) / g.minibatch_size;
    if (num_minibatches > g.capacity_minibatches)
        asm volatile("{trap;}");

    int expert_rows_total = 0;
    if constexpr (BUILD_EP8_WGRAD_EXPERT_ROW_PREFIX)
        g.expert_row_offsets[{0}] = 0;
    for (int expert_idx = 0; expert_idx < g.num_local_experts; ++expert_idx) {
        const int expert_rows = g.tokens_per_expert[{expert_idx}];
        if (expert_rows < 0 || expert_rows % config::MLP_Mb != 0)
            asm volatile("{trap;}");
        expert_rows_total += expert_rows;
        if constexpr (BUILD_EP8_WGRAD_EXPERT_ROW_PREFIX)
            g.expert_row_offsets[{expert_idx + 1}] = expert_rows_total;
    }
    if (expert_rows_total != routed_rows)
        asm volatile("{trap;}");

    int pair_count = 0;
    int completed_segment_count = 0;
    g.minibatch_expert_offsets[{0}] = 0;
    g.completed_segment_offsets[{0}] = 0;
    for (int global_minibatch_idx = 0;
         global_minibatch_idx < num_minibatches;
         ++global_minibatch_idx) {
        const int minibatch_begin = global_minibatch_idx * g.minibatch_size;
        const int minibatch_end = min(minibatch_begin + g.minibatch_size, routed_rows);
        const int macrobatch_idx = minibatch_begin / g.macrobatch_size;
        const int macrobatch_begin = macrobatch_idx * g.macrobatch_size;
        const int macrobatch_end = min(macrobatch_begin + g.macrobatch_size, routed_rows);
        int expert_begin = 0;
        for (int expert_idx = 0; expert_idx < g.num_local_experts; ++expert_idx) {
            const int expert_end = expert_begin + g.tokens_per_expert[{expert_idx}];
            pair_count += max(expert_begin, minibatch_begin) <
                          min(expert_end, minibatch_end);
            const int segment_begin = max(expert_begin, macrobatch_begin);
            const int segment_end = min(expert_end, macrobatch_end);
            const bool completes_here =
                segment_begin < segment_end &&
                (segment_end - 1) / g.minibatch_size == global_minibatch_idx;
            if (completes_here) {
                if constexpr (BUILD_EP8_O1_DECODE) {
                    if (completed_segment_count >= g.completed_segment_capacity)
                        asm volatile("{trap;}");
                    g.completed_segment_experts[{completed_segment_count}] = expert_idx;
                }
                ++completed_segment_count;
            }
            expert_begin = expert_end;
        }
        g.minibatch_expert_offsets[{global_minibatch_idx + 1}] = pair_count;
        g.completed_segment_offsets[{global_minibatch_idx + 1}] = completed_segment_count;
    }
    for (int global_minibatch_idx = num_minibatches + 1;
         global_minibatch_idx <= g.capacity_minibatches;
         ++global_minibatch_idx) {
        g.minibatch_expert_offsets[{global_minibatch_idx}] = pair_count;
        g.completed_segment_offsets[{global_minibatch_idx}] = completed_segment_count;
    }

    if constexpr (BUILD_EP8_O1_DECODE) {
        if (g.minibatch_routed_bwd_tasks <= 0 ||
            g.minibatch_task_owner_bucket_width <= 0 ||
            g.segment_wgrad_tasks <= 0 ||
            g.minibatch_task_owner_bucket_capacity <= 0)
            asm volatile("{trap;}");

        const int total_routed_tasks =
            num_minibatches * g.minibatch_routed_bwd_tasks +
            completed_segment_count * g.segment_wgrad_tasks;
        const int num_owner_buckets =
            total_routed_tasks == 0
                ? 0
                : 1 + (total_routed_tasks - 1) /
                          g.minibatch_task_owner_bucket_width;
        if (num_owner_buckets > g.minibatch_task_owner_bucket_capacity)
            asm volatile("{trap;}");

        int owner = 0;
        for (int bucket = 0; bucket < num_owner_buckets; ++bucket) {
            const int bucket_begin =
                bucket * g.minibatch_task_owner_bucket_width;
            while (owner + 1 < num_minibatches) {
                const int next_owner_begin =
                    (owner + 1) * g.minibatch_routed_bwd_tasks +
                    g.completed_segment_offsets[{owner + 1}] *
                        g.segment_wgrad_tasks;
                if (next_owner_begin > bucket_begin)
                    break;
                ++owner;
            }
            g.minibatch_task_owner_buckets[{bucket}] = owner;
        }
        for (int bucket = num_owner_buckets;
             bucket < g.minibatch_task_owner_bucket_capacity;
             ++bucket)
            g.minibatch_task_owner_buckets[{bucket}] = max(0, num_minibatches - 1);
        for (int segment = completed_segment_count;
             segment < g.completed_segment_capacity;
             ++segment)
            g.completed_segment_experts[{segment}] = -1;
    }
}

template <bool IS_CLAMPED, bool MINIBATCH_RELEASE = false,
          bool FULL_CONTEXT = false>
static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_bwd_kernel(const globals_bwd &g) {
    static_assert(!FULL_CONTEXT || (MINIBATCH_RELEASE && USE_MXFP8));
    const int num_local_experts = g.w_routed_gate.depth();
    const int intermediate_dim_col_blocks = g.hidden_shared.cols() / config::MLP_Nb;
    const int hidden_dim_col_blocks = g.d_y_shared.cols() / config::MLP_Nb;

    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();

    const int shared_row_blocks = g.d_y_shared.rows() / config::MLP_Mb;
    const int shared_dgrad_down_tasks = shared_row_blocks * intermediate_dim_col_blocks;
    const int shared_swiglu_bwd_tiles = (g.hidden_shared.rows() / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_bwd_tasks = (shared_swiglu_bwd_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
    const int shared_dgrad_gate_up_tasks = shared_row_blocks * hidden_dim_col_blocks;
    const int shared_wgrad_tasks = intermediate_dim_col_blocks * hidden_dim_col_blocks; // per weight matrix
    const int shared_tasks = shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + 3 * shared_wgrad_tasks;

    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int minibatch_routed_dgrad_down_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_fp8_routed.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_bwd_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
    const int minibatch_routed_dgrad_gate_up_tasks = minibatch_routed_row_blocks * hidden_dim_col_blocks;
    const int minibatch_routed_bwd_tasks = minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks + minibatch_routed_dgrad_gate_up_tasks;

    const int wgrad_matrix_tasks = num_local_experts * intermediate_dim_col_blocks * hidden_dim_col_blocks;
    const int wgrad_tasks = 3 * wgrad_matrix_tasks;
    const int wgrad_tile_tasks = intermediate_dim_col_blocks * hidden_dim_col_blocks;

    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_fwd_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_replay_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_fwd_tasks;

    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int macrobatch_size = g.macrobatch_size;
    const int num_tokens = g.num_tokens[{0}];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
    // FULL_CONTEXT is a distinct EP8 specialization: gradients still use
    // macrobatch_size as their ring capacity, while saved x/gate/up/hidden
    // rows use their global routed-row address.  Keep the mode compile-time so
    // replay and saved-address predicates disappear from the hot kernel.
    if constexpr (FULL_CONTEXT) {
        if (g.context_size <= macrobatch_size ||
            num_tokens < 0 || num_tokens > g.context_size)
            asm volatile("{trap;}");
    }
    // Fine retirement can only expose overlap when a physical ring slot has a
    // later generation.  With one generation, reuse the original macro task
    // order and communication loop; this avoids fine-decoder/interleaving
    // overhead without changing the explicit minibatch API or multi-generation
    // synchronization protocol.
    const bool use_minibatch_pipeline = MINIBATCH_RELEASE && num_macrobatches > 1;
    // In the EP8 MXFP8 pipeline, reverse-dispatch is the transitive sink for
    // every non-Wgrad ring reader: dY -> dHidden -> dGate/dUp -> dX and the
    // router partials are all consumed before each comm CTA returns.  Wgrad
    // source reads remain a separate lifetime because they are not ancestors
    // of reverse-dispatch.
    const bool use_ep8_mxfp8_transitive_retire =
        use_minibatch_pipeline && NUM_DEVICES == 8 && USE_MXFP8;

    auto num_minibatches_of = [&](int macrobatch_idx) { return (min(num_tokens - macrobatch_idx * macrobatch_size, macrobatch_size) + g.minibatch_size - 1) / g.minibatch_size; };
    auto num_dispatch_tasks_of = [&](int macrobatch_idx) {
        const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
        const int col_blocks = (g.d_y_shared.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;
        return (macrobatch_tokens / config::DISPATCH_Mb) * col_blocks;
    };
    auto num_combine_tasks_of = [&](int macrobatch_idx) {
        const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
        const int col_blocks = (g.d_y_shared.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
        const int tiles = (macrobatch_tokens / config::COMBINE_Mb) * col_blocks;
        return (tiles + config::COMBINE_PIPE_DEPTH - 1) / config::COMBINE_PIPE_DEPTH;
    };
    auto routed_buffers_done_required_count_of = [&](int macrobatch_idx) {
        return config::CLUSTER_SIZE * (num_minibatches_of(macrobatch_idx) * minibatch_routed_bwd_tasks + wgrad_tasks) + num_combine_tasks_of(macrobatch_idx);
    };
    auto wgrad_read_required_count_of = [&](int global_minibatch_idx) {
        const int active_experts =
            g.minibatch_expert_offsets[{global_minibatch_idx + 1}] -
            g.minibatch_expert_offsets[{global_minibatch_idx}];
        return 3 * active_experts * wgrad_tile_tasks;
    };
    auto minibatch_non_wgrad_required_count = [&]() {
        if (use_ep8_mxfp8_transitive_retire)
            return g.num_comm_sms;
        // The generic fine path retains the conservative per-compute arrivals.
        return config::CLUSTER_SIZE * minibatch_routed_bwd_tasks +
               g.num_comm_sms;
    };

    const int num_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int saved_macrobatch_num_minibatches = num_minibatches_of(0);
    const int saved_context_num_minibatches = FULL_CONTEXT
        ? num_minibatches
        : saved_macrobatch_num_minibatches;
    const int saved_macrobatch_tasks = saved_macrobatch_num_minibatches * minibatch_routed_bwd_tasks + wgrad_tasks;
    const int replayed_macrobatch_tasks = minibatches_per_macrobatch * (minibatch_routed_replay_tasks + minibatch_routed_bwd_tasks) + wgrad_tasks;
    const int num_replay_minibatches = FULL_CONTEXT
        ? 0
        : num_minibatches - saved_context_num_minibatches;
    const int completed_segments = MINIBATCH_RELEASE
        ? g.completed_segment_offsets[{num_minibatches}]
        : 0;
    const int compact_segment_wgrad_tasks =
        3 * completed_segments * wgrad_tile_tasks;
    const int true_num_clusters = use_minibatch_pipeline
        ? comm_clusters + shared_tasks + num_minibatches * minibatch_routed_bwd_tasks +
          num_replay_minibatches * minibatch_routed_replay_tasks +
          compact_segment_wgrad_tasks
        : comm_clusters + shared_tasks + num_minibatches * minibatch_routed_bwd_tasks +
          num_replay_minibatches * minibatch_routed_replay_tasks + num_macrobatches * wgrad_tasks;
    if (cluster_idx >= true_num_clusters) return;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_fwd_bitfield = 0xFFFF0000;
    uint32_t swiglu_bwd_bitfield = 0xFFFF0000;
    uint32_t dispatch_bitfield = 0xFFFF0000;
    uint32_t combine_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_finished[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore swiglu_fwd_inputs_arrived[config::SWIGLU_FWD_PIPE_DEPTH];
    __shared__ semaphore swiglu_bwd_inputs_arrived[config::SWIGLU_BWD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_inputs_arrived;
    __shared__ semaphore combine_inputs_arrived[config::COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_FWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_fwd_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_BWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_bwd_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
            init_semaphore(gemm_scales_arrived[i], 0, 1);
            init_semaphore(gemm_inputs_finished[i], 0, 1);
            init_semaphore(gemm_scales_finished[i], 0, 1);
        }
        init_semaphore(gemm_outputs_arrived, 0, 1);
        init_semaphore(gemm_outputs_finished, 0, config::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < config::CLC_PIPE_DEPTH; ++i) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, config::CLUSTER_SIZE * config::NUM_WARPS);
        }
        #pragma unroll
        for (int i = 0; i < config::CLC_DRAIN_PIPE_DEPTH; ++i) {
            init_semaphore(drain_schedule_arrived[i], 0, 1);
            init_semaphore(drain_schedule_finished[i], 0, config::CLUSTER_SIZE);
        }
        init_semaphore(dispatch_inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < config::COMBINE_PIPE_DEPTH; ++i) {
            init_semaphore(combine_inputs_arrived[i], 0, 1);
        }
    }

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH>>(256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH>>(384);
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        auto reverse_combine = [&](int macrobatch_idx, int task_idx) {
            dispatch_kernel<true, FULL_CONTEXT>(g.d_y_buffer, g.d_y_fp8_routed, &g.d_y_sc_routed, &g.d_y_fp8_t_routed, &g.d_y_sc_t_routed,
                                     &g.router_weights, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                     nullptr, nullptr, g.d_y_routed_ready,
                                     dispatch_inputs_arrived, dispatch_bitfield,
                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     -1, 0, smem_base_addr);
        };
        auto reverse_dispatch = [&](int macrobatch_idx, int task_idx) {
            combine_kernel(g.d_x_routed_buffer, g.d_x_routed, &g.d_router_weight_buffer, &g.d_router_weight_partials,
                           g.schedule_peer_rank, g.schedule_peer_token_idx,
                           g.d_x_routed_ready, use_minibatch_pipeline ? nullptr : (num_macrobatches > 1 ? &g.routed_buffers_done : nullptr),
                           combine_inputs_arrived, combine_bitfield,
                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, smem_base_addr);
        };
        auto replay_dispatch = [&](int macrobatch_idx, int task_idx) {
            if constexpr (!FULL_CONTEXT) {
                dispatch_kernel<false>(g.x_routed_send_buffer, g.x_fp8_routed, &g.x_sc_routed, &g.x_fp8_t_routed, &g.x_sc_t_routed,
                                         nullptr, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                         nullptr, nullptr, g.replayed_x_routed_ready,
                                         dispatch_inputs_arrived, dispatch_bitfield,
                                         num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                         -1, 0, smem_base_addr);
            }
        };
        if (!use_minibatch_pipeline) {
            preload_router_weights_kernel(g.router_weight_buffer, g.router_weights,
                                          g.schedule_peer_rank, g.schedule_peer_token_idx,
                                          nullptr, g.router_weights_ready,
                                          num_tokens, FULL_CONTEXT ? num_tokens : macrobatch_size,
                                          0, comm_cta_idx, g.num_comm_sms, -1, 0);
            for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks_of(0); task_idx += g.num_comm_sms)
                reverse_combine(0, task_idx);
            for (int macrobatch_idx = 0; macrobatch_idx < num_macrobatches; ++macrobatch_idx) {
                // All reverse-dispatch tasks must complete before this CTA moves on: the next macrobatch's pulls
                // wait on routed_buffers_done, which counts every rank's reverse-dispatch arrivals (including this CTA's)
                for (int task_idx = comm_cta_idx; task_idx < num_combine_tasks_of(macrobatch_idx); task_idx += g.num_comm_sms)
                    reverse_dispatch(macrobatch_idx, task_idx);
                if (macrobatch_idx + 1 < num_macrobatches) {
                    preload_router_weights_kernel(g.router_weight_buffer, g.router_weights,
                                                  g.schedule_peer_rank, g.schedule_peer_token_idx,
                                                  &g.routed_buffers_done, g.router_weights_ready,
                                                  num_tokens, macrobatch_size, macrobatch_idx + 1, comm_cta_idx, g.num_comm_sms,
                                                  macrobatch_idx, routed_buffers_done_required_count_of(macrobatch_idx));
                    for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks_of(macrobatch_idx + 1); task_idx += g.num_comm_sms) {
                        reverse_combine(macrobatch_idx + 1, task_idx);
                        if constexpr (!FULL_CONTEXT)
                            replay_dispatch(macrobatch_idx + 1, task_idx);
                    }
                }
            }
        } else {
            const int dispatch_col_blocks = (g.d_y_shared.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;
            const int combine_col_blocks = (g.d_y_shared.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
            auto minibatch_rows_of = [&](int global_minibatch_idx) {
                return min(g.minibatch_size, num_tokens - global_minibatch_idx * g.minibatch_size);
            };
            auto preload_and_reverse_combine = [&](int global_minibatch_idx,
                                                   int previous_global_minibatch_idx,
                                                   bool router_weights_preloaded = false) {
                const int macrobatch_idx = global_minibatch_idx / minibatches_per_macrobatch;
                const int local_minibatch_idx = global_minibatch_idx % minibatches_per_macrobatch;
                if (!router_weights_preloaded) {
                    if constexpr (FULL_CONTEXT) {
                        // Router scores are immutable and globally addressed in
                        // this specialization.  Each comm CTA still protects the
                        // reused gradient slot, but no router copy or comm-CTA
                        // transfer_done rendezvous is needed after the bulk load.
                        if (threadIdx.x == 0 && previous_global_minibatch_idx >= 0) {
                            barrier_wait(g.routed_buffers_done,
                                         previous_global_minibatch_idx,
                                         minibatch_non_wgrad_required_count());
                            barrier_wait(g.wgrad_read_consumed,
                                         previous_global_minibatch_idx,
                                         wgrad_read_required_count_of(previous_global_minibatch_idx));
                        }
                        __syncthreads();
                    } else {
                        preload_router_weights_minibatch_kernel(
                            g.router_weight_buffer, g.router_weights,
                            g.schedule_peer_rank, g.schedule_peer_token_idx,
                            &g.routed_buffers_done, minibatch_non_wgrad_required_count(),
                            &g.wgrad_read_consumed,
                            previous_global_minibatch_idx >= 0 ? wgrad_read_required_count_of(previous_global_minibatch_idx) : 0,
                            g.router_weights_ready,
                            num_tokens, macrobatch_size, g.minibatch_size,
                            global_minibatch_idx, comm_cta_idx, g.num_comm_sms,
                            previous_global_minibatch_idx);
                    }
                }

                const int rows = minibatch_rows_of(global_minibatch_idx);
                const int tasks = (rows / config::DISPATCH_Mb) * dispatch_col_blocks;
                const int task_base = local_minibatch_idx * (g.minibatch_size / config::DISPATCH_Mb) * dispatch_col_blocks;
                for (int local_task_idx = comm_cta_idx; local_task_idx < tasks; local_task_idx += g.num_comm_sms)
                    reverse_combine(macrobatch_idx, task_base + local_task_idx);
            };
            auto replay_dispatch_minibatch = [&](int global_minibatch_idx) {
                if constexpr (!FULL_CONTEXT) {
                    const int macrobatch_idx = global_minibatch_idx / minibatches_per_macrobatch;
                    const int local_minibatch_idx = global_minibatch_idx % minibatches_per_macrobatch;
                    const int rows = minibatch_rows_of(global_minibatch_idx);
                    const int tasks = (rows / config::DISPATCH_Mb) * dispatch_col_blocks;
                    const int task_base = local_minibatch_idx * (g.minibatch_size / config::DISPATCH_Mb) * dispatch_col_blocks;
                    for (int local_task_idx = comm_cta_idx; local_task_idx < tasks; local_task_idx += g.num_comm_sms)
                        replay_dispatch(macrobatch_idx, task_base + local_task_idx);
                }
            };
            auto reverse_dispatch_minibatch = [&](int global_minibatch_idx) {
                const int macrobatch_idx = global_minibatch_idx / minibatches_per_macrobatch;
                const int local_minibatch_idx = global_minibatch_idx % minibatches_per_macrobatch;
                const int rows = minibatch_rows_of(global_minibatch_idx);
                const int tile_count = (rows / config::COMBINE_Mb) * combine_col_blocks;
                const int tile_base = local_minibatch_idx * (g.minibatch_size / config::COMBINE_Mb) * combine_col_blocks;
                const int tasks = (tile_count + config::COMBINE_PIPE_DEPTH - 1) / config::COMBINE_PIPE_DEPTH;
                for (int local_task_idx = comm_cta_idx; local_task_idx < tasks; local_task_idx += g.num_comm_sms) {
                    combine_kernel(g.d_x_routed_buffer, g.d_x_routed, &g.d_router_weight_buffer, &g.d_router_weight_partials,
                                   g.schedule_peer_rank, g.schedule_peer_token_idx,
                                   g.d_x_routed_ready, nullptr,
                                   combine_inputs_arrived, combine_bitfield,
                                   num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, local_task_idx,
                                   smem_base_addr, tile_base, tile_count);
                }
                if (threadIdx.x == 0 &&
                    global_minibatch_idx + minibatches_per_macrobatch < num_minibatches)
                    barrier_arrive(g.routed_buffers_done, global_minibatch_idx);
                __syncthreads();
            };

            // The initially saved forward context has no prior ring owner.
            // Copy the saved router-score extent with one comm-CTA collective,
            // then expose reverse-combine work at minibatch granularity.  The
            // FULL_CONTEXT specialization gathers every global routed row once;
            // later generations only wait for gradient-ring slot retirement.
            preload_router_weights_kernel(
                g.router_weight_buffer, g.router_weights,
                g.schedule_peer_rank, g.schedule_peer_token_idx,
                nullptr, g.router_weights_ready,
                num_tokens, FULL_CONTEXT ? num_tokens : macrobatch_size,
                0, comm_cta_idx,
                g.num_comm_sms, -1, 0);
            for (int minibatch_idx = 0; minibatch_idx < num_minibatches_of(0); ++minibatch_idx)
                preload_and_reverse_combine(minibatch_idx, -1, true);

            for (int macrobatch_idx = 0; macrobatch_idx < num_macrobatches; ++macrobatch_idx) {
                for (int minibatch_idx = 0; minibatch_idx < num_minibatches_of(macrobatch_idx); ++minibatch_idx) {
                    const int global_minibatch_idx = macrobatch_idx * minibatches_per_macrobatch + minibatch_idx;
                    reverse_dispatch_minibatch(global_minibatch_idx);
                    if (macrobatch_idx + 1 < num_macrobatches &&
                        minibatch_idx < num_minibatches_of(macrobatch_idx + 1)) {
                        const int next_global_minibatch_idx = global_minibatch_idx + minibatches_per_macrobatch;
                        preload_and_reverse_combine(next_global_minibatch_idx, global_minibatch_idx);
                        if constexpr (!FULL_CONTEXT)
                            replay_dispatch_minibatch(next_global_minibatch_idx);
                    }
                }
            }
        }
        return;
    }

    // Prefix of routed work before global minibatch g.  A compact Wgrad task
    // appears after the replay/backward work of the minibatch containing the
    // final row of its expert/macro segment.  The same decoder is used by task
    // execution and CTA-local classification.
    auto minibatch_routed_task_prefix = [&](int global_minibatch_idx) {
        int replay_minibatches = 0;
        if constexpr (!FULL_CONTEXT)
            replay_minibatches =
                max(0, global_minibatch_idx - saved_context_num_minibatches);
        return global_minibatch_idx * minibatch_routed_bwd_tasks +
               replay_minibatches * minibatch_routed_replay_tasks +
               3 * g.completed_segment_offsets[{global_minibatch_idx}] *
                   wgrad_tile_tasks;
    };
    auto decode_minibatch_routed_task = [&](int idx, int &global_minibatch_idx,
                                             int &minibatch_task_idx, bool &replayed) {
        if constexpr (FULL_CONTEXT) {
            const int owner_bucket =
                idx >> g.minibatch_task_owner_bucket_shift;
            int owner = g.minibatch_task_owner_buckets[{owner_bucket}];
            if (idx >= minibatch_routed_task_prefix(owner + 1))
                ++owner;
            global_minibatch_idx = owner;
        } else {
            int begin = 0;
            int end = num_minibatches;
            while (begin < end) {
                const int middle = begin + (end - begin) / 2;
                if (minibatch_routed_task_prefix(middle + 1) <= idx)
                    begin = middle + 1;
                else
                    end = middle;
            }
            global_minibatch_idx = begin;
        }
        minibatch_task_idx =
            idx - minibatch_routed_task_prefix(global_minibatch_idx);
        if constexpr (FULL_CONTEXT)
            replayed = false;
        else
            replayed = global_minibatch_idx >= saved_context_num_minibatches;
    };

    // Maps a segment completing in this global minibatch to the dense
    // (expert, output-tile) task index expected by the grouped GEMM.  A segment
    // is the nonempty intersection of one expert with one macrobatch.
    auto completed_segment_wgrad_dense_task = [&](int global_minibatch_idx, int lane_task_idx) {
        const int lane = lane_task_idx / wgrad_tile_tasks;
        const int tile = lane_task_idx % wgrad_tile_tasks;
        if constexpr (FULL_CONTEXT) {
            const int segment_ordinal =
                g.completed_segment_offsets[{global_minibatch_idx}] + lane;
            const int expert_idx =
                g.completed_segment_experts[{segment_ordinal}];
            if (expert_idx < 0 || expert_idx >= num_local_experts)
                asm volatile("{trap;}");
            return expert_idx * wgrad_tile_tasks + tile;
        } else {
            const int macrobatch_idx = global_minibatch_idx / minibatches_per_macrobatch;
            const int macrobatch_begin = macrobatch_idx * macrobatch_size;
            const int macrobatch_end = min(macrobatch_begin + macrobatch_size, num_tokens);
            int expert_begin = 0;
            int completed_lane = 0;
            for (int expert_idx = 0; expert_idx < num_local_experts; ++expert_idx) {
                const int expert_end = expert_begin + g.tokens_per_expert[{expert_idx}];
                const int segment_begin = max(expert_begin, macrobatch_begin);
                const int segment_end = min(expert_end, macrobatch_end);
                if (segment_begin < segment_end &&
                    (segment_end - 1) / g.minibatch_size == global_minibatch_idx) {
                    if (completed_lane == lane)
                        return expert_idx * wgrad_tile_tasks + tile;
                    ++completed_lane;
                }
                expert_begin = expert_end;
            }
            asm volatile("{trap;}");
            return -1;
        }
    };

    // Swiglu (forward and backward) tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        if (compute_cluster_idx < 0) return false;
        else if (compute_cluster_idx < shared_dgrad_down_tasks) return false; // shared dgrad down
        else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) return true; // shared swiglu bwd
        else if (compute_cluster_idx < shared_tasks) return false; // shared dgrad/wgrad
        else if (compute_cluster_idx >= true_num_clusters - comm_clusters) return false;

        int idx = compute_cluster_idx - shared_tasks;
        if (use_minibatch_pipeline) {
            int global_minibatch_idx, minibatch_task_idx;
            bool replayed = false;
            decode_minibatch_routed_task(idx, global_minibatch_idx, minibatch_task_idx, replayed);
            if constexpr (!FULL_CONTEXT) {
                if (replayed) {
                    if (minibatch_task_idx < minibatch_routed_replay_tasks)
                        return minibatch_task_idx >= 2 * minibatch_routed_gate_up_tasks;
                    minibatch_task_idx -= minibatch_routed_replay_tasks;
                }
            }
            if (minibatch_task_idx >= minibatch_routed_bwd_tasks)
                return false;
            return minibatch_task_idx >= minibatch_routed_dgrad_down_tasks &&
                   minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks;
        }
        int macrobatch_num_minibatches, macrobatch_task_idx;
        if constexpr (FULL_CONTEXT) {
            macrobatch_num_minibatches = saved_macrobatch_num_minibatches;
            macrobatch_task_idx = idx;
        } else {
            if (idx < saved_macrobatch_tasks) {
                macrobatch_num_minibatches = saved_macrobatch_num_minibatches;
                macrobatch_task_idx = idx;
            } else {
                idx -= saved_macrobatch_tasks;
                const int macrobatch_idx = 1 + idx / replayed_macrobatch_tasks;
                macrobatch_num_minibatches = num_minibatches_of(macrobatch_idx);
                macrobatch_task_idx = idx % replayed_macrobatch_tasks;
                if (macrobatch_task_idx < macrobatch_num_minibatches * minibatch_routed_replay_tasks)
                    return macrobatch_task_idx % minibatch_routed_replay_tasks >= 2 * minibatch_routed_gate_up_tasks; // swiglu fwd replay
                macrobatch_task_idx -= macrobatch_num_minibatches * minibatch_routed_replay_tasks;
            }
        }
        if (macrobatch_task_idx >= macrobatch_num_minibatches * minibatch_routed_bwd_tasks) return false; // wgrad
        const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_bwd_tasks;
        return minibatch_task_idx >= minibatch_routed_dgrad_down_tasks &&
               minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks; // swiglu bwd
    };

    const int d_gate_up_row_block_ready_required_count = (config::MLP_Mb / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const index_gl *buffer_done = num_macrobatches > 1 ? &g.routed_buffers_done : nullptr;
    bool current_is_cta_local = is_cta_local_task(cluster_idx - comm_clusters);
    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;

        if (compute_cluster_idx < shared_dgrad_down_tasks) {
            // Shared dgrad down: d_hidden_shared = d_y_shared @ w_shared_down
            const int task_idx = compute_cluster_idx;
            expert_grouped_gemm_kernel<true, false, true>(g.d_y_shared, g.w_shared_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.d_hidden_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, nullptr, nullptr, &g.d_hidden_ready, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) {
            // Shared Swiglu bwd: d_gate_shared, d_up_shared = swiglu_bwd(d_hidden_shared, gate_shared, up_shared)
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks;
            swiglu_bwd_kernel<true, IS_CLAMPED>(g.d_hidden_shared, g.gate_shared, g.up_shared, g.d_gate_shared, g.d_up_shared,
                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr,
                             g.d_hidden_ready, nullptr, g.d_gate_up_ready, nullptr,
                             swiglu_bwd_inputs_arrived, swiglu_bwd_bitfield,
                             g.gate_shared.rows(), g.swiglu_limit, macrobatch_size, g.minibatch_size,
                             0, 0, task_idx, cta_rank,
                             0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks) {
            // Shared dgrad gate+up: d_x_shared = d_gate_shared @ w_shared_gate + d_up_shared @ w_shared_up
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks;
            expert_grouped_gemm_kernel<true, false, true>(g.d_gate_shared, g.w_shared_gate, nullptr, nullptr, &g.d_up_shared, &g.w_shared_up, nullptr, nullptr,
                                      g.d_x_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + shared_wgrad_tasks) {
            // Shared wgrad down: d_w_shared_down += d_y_shared^T @ hidden_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks;
            expert_grouped_gemm_kernel<true, true>(g.d_y_shared, g.hidden_shared, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            g.d_w_shared_down, nullptr, nullptr,
                                            g.tokens_per_expert, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            d_tt, a_sc_tt, b_sc_tt,
                                            gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks - shared_wgrad_tasks) {
            // Shared wgrad gate: d_w_shared_gate += d_gate_shared^T @ x_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks - shared_wgrad_tasks;
            expert_grouped_gemm_kernel<true, true>(g.d_gate_shared, g.x_shared, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            g.d_w_shared_gate, nullptr, nullptr,
                                            g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr, nullptr,
                                            d_tt, a_sc_tt, b_sc_tt,
                                            gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared wgrad up: d_w_shared_up += d_up_shared^T @ x_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks - 2 * shared_wgrad_tasks;
            expert_grouped_gemm_kernel<true, true>(g.d_up_shared, g.x_shared, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            g.d_w_shared_up, nullptr, nullptr,
                                            g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr, nullptr,
                                            d_tt, a_sc_tt, b_sc_tt,
                                            gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else {
            // Routed / replay tasks
            const int global_routed_task_idx = compute_cluster_idx - shared_tasks;
            bool replayed = false;
            int global_minibatch_idx = 0;
            int macrobatch_idx = 0;
            int minibatch_idx = 0;
            int macrobatch_task_idx = 0;
            int macrobatch_num_minibatches = 0;
            int num_replay_tasks = 0;
            if (use_minibatch_pipeline) {
                decode_minibatch_routed_task(global_routed_task_idx, global_minibatch_idx,
                                             macrobatch_task_idx, replayed);
                macrobatch_idx = global_minibatch_idx / minibatches_per_macrobatch;
                minibatch_idx = global_minibatch_idx % minibatches_per_macrobatch;
                macrobatch_num_minibatches = num_minibatches_of(macrobatch_idx);
                if constexpr (!FULL_CONTEXT)
                    num_replay_tasks = replayed ? minibatch_routed_replay_tasks : 0;
            } else {
                if constexpr (FULL_CONTEXT) {
                    macrobatch_idx = 0;
                    macrobatch_task_idx = global_routed_task_idx;
                    macrobatch_num_minibatches = saved_macrobatch_num_minibatches;
                } else {
                    replayed = global_routed_task_idx >= saved_macrobatch_tasks;
                    const int replayed_task_idx = global_routed_task_idx - saved_macrobatch_tasks;
                    macrobatch_idx = replayed ? 1 + replayed_task_idx / replayed_macrobatch_tasks : 0;
                    macrobatch_task_idx = replayed ? replayed_task_idx % replayed_macrobatch_tasks : global_routed_task_idx;
                    macrobatch_num_minibatches = num_minibatches_of(macrobatch_idx);
                    num_replay_tasks = replayed ? macrobatch_num_minibatches * minibatch_routed_replay_tasks : 0;
                }
            }

            const index_gl *replayed_gate_up_ready_for_bwd = nullptr;
            const index_gl *replayed_hidden_ready_for_wgrad = nullptr;
            const index_gl *replayed_x_ready_for_wgrad = nullptr;
            if constexpr (!FULL_CONTEXT) {
                replayed_gate_up_ready_for_bwd =
                    replayed ? &g.replayed_gate_up_ready : nullptr;
                replayed_hidden_ready_for_wgrad =
                    replayed ? &g.replayed_hidden_ready : nullptr;
                replayed_x_ready_for_wgrad =
                    replayed ? &g.replayed_x_routed_ready : nullptr;
            }

            bool is_replay_task = false;
            if constexpr (!FULL_CONTEXT)
                is_replay_task = macrobatch_task_idx < num_replay_tasks;
            if constexpr (!FULL_CONTEXT) {
                if (is_replay_task) {
                    if (!use_minibatch_pipeline)
                        minibatch_idx = macrobatch_task_idx / minibatch_routed_replay_tasks;
                    const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_replay_tasks;
                    if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                        // Replay gate GEMM refreshes the routed activation.
                        const int task_idx = minibatch_task_idx;
                        expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_gate, &g.x_sc_routed, &g.w_routed_gate_sc, nullptr, nullptr, nullptr, nullptr,
                                                   g.gate_routed, &g.gate_fp8_routed, &g.gate_sc_routed,
                                                   g.tokens_per_expert, &g.replayed_x_routed_ready, nullptr, nullptr, &g.replayed_gate_up_ready, nullptr, nullptr,
                                                   d_tt, a_sc_tt, b_sc_tt,
                                                   gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                   num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                                   0, 0, 0, 0, 0, smem_base_addr);
                    } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                        // Replay up GEMM refreshes the routed activation.
                        const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                        expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_up, &g.x_sc_routed, &g.w_routed_up_sc, nullptr, nullptr, nullptr, nullptr,
                                                   g.up_routed, &g.up_fp8_routed, &g.up_sc_routed,
                                                   g.tokens_per_expert, &g.replayed_x_routed_ready, nullptr, nullptr, &g.replayed_gate_up_ready, nullptr, nullptr,
                                                   d_tt, a_sc_tt, b_sc_tt,
                                                   gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                   num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                                   0, 0, 0, 0, 0, smem_base_addr);
                    } else {
                        // Replay Swiglu refreshes the routed hidden activation.
                        const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                        swiglu_fwd_kernel<false, IS_CLAMPED>(g.gate_routed, g.up_routed, g.hidden_fp8_routed,
                                          &g.hidden_sc_routed, &g.hidden_fp8_t_routed, &g.hidden_sc_t_routed,
                                          g.replayed_gate_up_ready, g.replayed_hidden_ready,
                                          swiglu_fwd_inputs_arrived, swiglu_fwd_bitfield,
                                          num_tokens, g.swiglu_limit, macrobatch_size, g.minibatch_size,
                                          macrobatch_idx, minibatch_idx,
                                          task_idx, cta_rank, 0, 0, smem_base_addr);
                    }
                }
            }
            if (!is_replay_task) {
                const int num_routed_tasks = use_minibatch_pipeline
                    ? minibatch_routed_bwd_tasks
                    : macrobatch_num_minibatches * minibatch_routed_bwd_tasks;
                const int routed_task_idx = macrobatch_task_idx - num_replay_tasks;
                if (!use_minibatch_pipeline)
                    minibatch_idx = routed_task_idx / minibatch_routed_bwd_tasks;
                if (!use_minibatch_pipeline)
                    global_minibatch_idx =
                        macrobatch_idx * minibatches_per_macrobatch + minibatch_idx;
                const int minibatch_task_idx = routed_task_idx % minibatch_routed_bwd_tasks;
                const int current_wgrad_matrix_tasks = use_minibatch_pipeline
                    ? (g.completed_segment_offsets[{global_minibatch_idx + 1}] -
                       g.completed_segment_offsets[{global_minibatch_idx}]) *
                          wgrad_tile_tasks
                    : wgrad_matrix_tasks;
                const bool slot_has_successor =
                    global_minibatch_idx + minibatches_per_macrobatch < num_minibatches;
                const index_gl *current_buffer_done =
                    use_ep8_mxfp8_transitive_retire
                        ? nullptr
                        : (use_minibatch_pipeline
                               ? (slot_has_successor ? &g.routed_buffers_done
                                                     : nullptr)
                               : buffer_done);
                if (routed_task_idx < num_routed_tasks && minibatch_task_idx < minibatch_routed_dgrad_down_tasks) {
                    // Dgrad down: d_hidden_routed = d_y_routed @ w_routed_down
                    const int task_idx = minibatch_task_idx;
                    expert_grouped_gemm_kernel<false, false, !USE_MXFP8, false, false, false,
                                               FULL_CONTEXT && config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX>(g.d_y_fp8_routed, g.w_routed_down_T, &g.d_y_sc_routed, &g.w_routed_down_T_sc, nullptr, nullptr, nullptr, nullptr,
                                               g.d_hidden_routed, nullptr, nullptr,
                                               g.tokens_per_expert, &g.d_y_routed_ready, nullptr, nullptr, &g.d_hidden_ready, nullptr, current_buffer_done,
                                               d_tt, a_sc_tt, b_sc_tt,
                                               gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, 0, 0, shared_dgrad_down_tasks,
                                               use_minibatch_pipeline ? macrobatch_idx * minibatches_per_macrobatch + minibatch_idx : macrobatch_idx,
                                               smem_base_addr, nullptr,
                                               FULL_CONTEXT && config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX
                                                   ? &g.expert_row_offsets
                                                   : nullptr);
                } else if (routed_task_idx < num_routed_tasks &&
                           minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks) {
                    // Routed Swiglu backward
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks;
                    swiglu_bwd_kernel<false, IS_CLAMPED, FULL_CONTEXT>(g.d_hidden_routed, g.gate_fp8_routed, g.up_fp8_routed, g.d_gate_fp8_routed, g.d_up_fp8_routed,
                                      &g.gate_sc_routed, &g.up_sc_routed, &g.d_gate_sc_routed, &g.d_up_sc_routed,
                                      &g.d_gate_fp8_t_routed, &g.d_gate_sc_t_routed, &g.d_up_fp8_t_routed, &g.d_up_sc_t_routed,
                                      &g.router_weights, &g.d_router_weight_partials, &g.schedule_peer_rank,
                                      g.d_hidden_ready, replayed_gate_up_ready_for_bwd, g.d_gate_up_ready, current_buffer_done,
                                      swiglu_bwd_inputs_arrived, swiglu_bwd_bitfield,
                                      num_tokens, g.swiglu_limit, macrobatch_size, g.minibatch_size,
                                      macrobatch_idx, minibatch_idx,
                                      task_idx, cta_rank, shared_dgrad_down_tasks, 0, shared_row_blocks,
                                      use_minibatch_pipeline ? macrobatch_idx * minibatches_per_macrobatch + minibatch_idx : macrobatch_idx,
                                      smem_base_addr);
                } else if (routed_task_idx < num_routed_tasks) {
                    // Dgrad gate+up: d_x_routed = d_gate @ w_routed_gate + d_up @ w_routed_up
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks - minibatch_routed_swiglu_bwd_tasks;
                    expert_grouped_gemm_kernel<false, false, !USE_MXFP8, false, false, false,
                                               FULL_CONTEXT && config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX>(g.d_gate_fp8_routed, g.w_routed_gate_T, &g.d_gate_sc_routed, &g.w_routed_gate_T_sc,
                                               &g.d_up_fp8_routed, &g.w_routed_up_T, &g.d_up_sc_routed, &g.w_routed_up_T_sc,
                                               g.d_x_routed, nullptr, nullptr,
                                               g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, &g.d_x_routed_ready, current_buffer_done,
                                               d_tt, a_sc_tt, b_sc_tt,
                                               gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, shared_row_blocks, d_gate_up_row_block_ready_required_count, 0,
                                               use_minibatch_pipeline ? macrobatch_idx * minibatches_per_macrobatch + minibatch_idx : macrobatch_idx,
                                               smem_base_addr, nullptr,
                                               FULL_CONTEXT && config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX
                                                   ? &g.expert_row_offsets
                                                   : nullptr);
                } else if (routed_task_idx < num_routed_tasks + current_wgrad_matrix_tasks) {
                    // Wgrad down: d_w_routed_down += d_y_routed^T @ hidden_routed
                    const int lane_task_idx = routed_task_idx - num_routed_tasks;
                    const int task_idx = use_minibatch_pipeline
                        ? completed_segment_wgrad_dense_task(global_minibatch_idx, lane_task_idx)
                        : lane_task_idx;
                    expert_grouped_gemm_kernel<false, true, false, false, MINIBATCH_RELEASE, FULL_CONTEXT>(g.d_y_fp8_t_routed, g.hidden_fp8_t_routed, &g.d_y_sc_t_routed, &g.hidden_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
                                                     g.d_w_routed_down, nullptr, nullptr,
                                                     g.tokens_per_expert, &g.d_y_routed_ready, replayed_hidden_ready_for_wgrad,
                                                     nullptr, nullptr, nullptr, MINIBATCH_RELEASE ? nullptr : buffer_done,
                                                     d_tt, a_sc_tt, b_sc_tt,
                                                     gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), 0, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, smem_base_addr,
                                                     use_minibatch_pipeline ? &g.wgrad_read_consumed : nullptr,
                                                     FULL_CONTEXT && config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX
                                                         ? &g.expert_row_offsets
                                                         : nullptr);
                } else if (routed_task_idx < num_routed_tasks + 2 * current_wgrad_matrix_tasks) {
                    // Wgrad gate: d_w_routed_gate += d_gate_routed^T @ x_routed
                    const int lane_task_idx = routed_task_idx - num_routed_tasks - current_wgrad_matrix_tasks;
                    const int task_idx = use_minibatch_pipeline
                        ? completed_segment_wgrad_dense_task(global_minibatch_idx, lane_task_idx)
                        : lane_task_idx;
                    expert_grouped_gemm_kernel<false, true, false, false, MINIBATCH_RELEASE, FULL_CONTEXT>(g.d_gate_fp8_t_routed, g.x_fp8_t_routed, &g.d_gate_sc_t_routed, &g.x_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
                                                     g.d_w_routed_gate, nullptr, nullptr,
                                                     g.tokens_per_expert, replayed_x_ready_for_wgrad,
                                                     &g.d_gate_up_ready, nullptr, nullptr, nullptr, MINIBATCH_RELEASE ? nullptr : buffer_done,
                                                     d_tt, a_sc_tt, b_sc_tt,
                                                     gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), shared_row_blocks, d_gate_up_row_block_ready_required_count,
                                                     0, macrobatch_idx, smem_base_addr,
                                                     use_minibatch_pipeline ? &g.wgrad_read_consumed : nullptr,
                                                     FULL_CONTEXT && config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX
                                                         ? &g.expert_row_offsets
                                                         : nullptr);
                } else {
                    // Wgrad up: d_w_routed_up += d_up_routed^T @ x_routed
                    const int lane_task_idx = routed_task_idx - num_routed_tasks - 2 * current_wgrad_matrix_tasks;
                    const int task_idx = use_minibatch_pipeline
                        ? completed_segment_wgrad_dense_task(global_minibatch_idx, lane_task_idx)
                        : lane_task_idx;
                    expert_grouped_gemm_kernel<false, true, false, false, MINIBATCH_RELEASE, FULL_CONTEXT>(g.d_up_fp8_t_routed, g.x_fp8_t_routed, &g.d_up_sc_t_routed, &g.x_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
                                                     g.d_w_routed_up, nullptr, nullptr,
                                                     g.tokens_per_expert, replayed_x_ready_for_wgrad,
                                                     &g.d_gate_up_ready, nullptr, nullptr, nullptr, MINIBATCH_RELEASE ? nullptr : buffer_done,
                                                     d_tt, a_sc_tt, b_sc_tt,
                                                     gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), shared_row_blocks, d_gate_up_row_block_ready_required_count,
                                                     0, macrobatch_idx, smem_base_addr,
                                                     use_minibatch_pipeline ? &g.wgrad_read_consumed : nullptr,
                                                     FULL_CONTEXT && config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX
                                                         ? &g.expert_row_offsets
                                                         : nullptr);
                }
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        const bool next_is_cta_local =
            cluster_idx >= 0 && is_cta_local_task(next_compute_cluster_idx);
        if (current_is_cta_local && cluster_idx >= 0 && !next_is_cta_local)
            everyone::tma::cluster::sync();
        current_is_cta_local = next_is_cta_local;
    }

    if (use_minibatch_pipeline) {
        // Bulk async groups are tracked per issuing thread.  Drain each
        // thread's final outstanding reduce before the CTA enters the exit
        // barrier; this does not serialize Wgrad tasks or macrobatches.
        tma::store_async_wait();
    }
    everyone::tma::cluster::sync();

    // CLC drain for no-op threadblocks
    if (cluster_idx >= 0 && warp::laneid() == 0) {
        const int stage = warpid();
        int iter = 0;
        if (cta_rank == 0)
            clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
        tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
        while (true) {
            wait(drain_schedule_arrived[stage], iter % 2);
            const auto schedule = clc::query(clc_drain_handle[stage]);
            warp::tma::cluster::arrive(drain_schedule_finished[stage], 0);
            if (cta_rank == 0)
                wait(drain_schedule_finished[stage], iter % 2);
            if (!schedule.success)
                break;
            if (cta_rank == 0)
                clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
            tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
            ++iter;
        }
    }
}

// Keep the specialized kernel symbol explicit in profilers.  This scheduling
// policy is currently tuned and accepted for EP8; other EP sizes continue to
// use the generic macrobatch kernel unless selected explicitly.
template <bool IS_CLAMPED>
static __device__ __forceinline__ void
dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_pipeline_kernel(const globals_bwd &g) {
    dispatch_mlp_swiglu_combine_bwd_kernel<IS_CLAMPED, true, false>(g);
}

template <bool IS_CLAMPED>
static __device__ __forceinline__ void
dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_full_context_no_replay_kernel(
    const globals_bwd &g
) {
    dispatch_mlp_swiglu_combine_bwd_kernel<IS_CLAMPED, true, true>(g);
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_mxfp8(
    // Symmetric buffers (input/output gradients and router weights)
    const at::Tensor &d_y_buffer,               // (num_local_tokens, H)
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_x_routed_buffer,        // (num_local_tokens * topk, H)
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,     // (num_local_tokens, topk)
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,   // (num_local_tokens, topk)
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,

    // Weights (routed transposes pre-quantized to MXFP8)
    const at::Tensor &w_shared_gate,            // (I, H)
    const at::Tensor &w_routed_gate_T,          // (E, H, I) fp8
    const at::Tensor &w_routed_gate_T_sc,       // (E * H / 128, I / 128, 32, 16)
    const at::Tensor &w_shared_up,              // (I, H)
    const at::Tensor &w_routed_up_T,            // (E, H, I) fp8
    const at::Tensor &w_routed_up_T_sc,         // (E * H / 128, I / 128, 32, 16)
    const at::Tensor &w_shared_down,            // (H, I)
    const at::Tensor &w_routed_down_T,          // (E, I, H) fp8
    const at::Tensor &w_routed_down_T_sc,       // (E * I / 128, H / 128, 32, 16)

    // Activations saved from the forward (routed ones already MXFP8; replay overwrites them)
    const at::Tensor &x_fp8_t_routed,           // (H, context_size) fp8
    const at::Tensor &x_sc_t_routed,            // (H / 128, context_size / 128, 32, 16)
    const at::Tensor &gate_shared,              // (num_local_tokens, I)
    const at::Tensor &gate_fp8_routed,          // (context_size, I) fp8
    const at::Tensor &gate_sc_routed,           // (context_size / 128, I / 128, 32, 16)
    const at::Tensor &up_shared,                // (num_local_tokens, I)
    const at::Tensor &up_fp8_routed,            // (context_size, I) fp8
    const at::Tensor &up_sc_routed,             // (context_size / 128, I / 128, 32, 16)
    const at::Tensor &hidden_shared,            // (num_local_tokens, I)
    const at::Tensor &hidden_fp8_t_routed,      // (I, context_size) fp8
    const at::Tensor &hidden_sc_t_routed,       // (I / 128, context_size / 128, 32, 16)

    // Activations and weights for forward replay
    const at::Tensor &x,                        // (num_local_tokens, H)
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &w_routed_gate,            // (E, I, H) fp8
    const at::Tensor &w_routed_gate_sc,         // (E * I / 128, H / 128, 32, 16)
    const at::Tensor &w_routed_up,              // (E, I, H) fp8
    const at::Tensor &w_routed_up_sc,           // (E * I / 128, H / 128, 32, 16)

    // Dispatch/combine schedule saved from the forward
    const at::Tensor &schedule_peer_rank,       // (schedule_capacity,)
    const at::Tensor &schedule_peer_token_idx,  // (schedule_capacity,)
    const at::Tensor &num_tokens,               // (1,)
    const at::Tensor &tokens_per_expert,        // (E,)

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size,
    bool use_minibatch_release
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_local_experts = w_routed_gate.size(0);
    const int context_size = x_fp8_t_routed.size(1);
    TORCH_CHECK(macrobatch_size > 0 && minibatch_size > 0,
                "MXFP8 macrobatch_size and minibatch_size must be positive");
    TORCH_CHECK(macrobatch_size % minibatch_size == 0,
                "MXFP8 macrobatch_size must be divisible by minibatch_size");
    TORCH_CHECK(context_size >= macrobatch_size,
                "MXFP8 context_size must be at least macrobatch_size");
    TORCH_CHECK(context_size % config::MLP_FP8_Kb == 0,
                "MXFP8 context_size must be divisible by the 128-row scale block");
    if (context_size > macrobatch_size) {
        TORCH_CHECK(use_minibatch_release,
                    "MXFP8 full-context/no-replay requires minibatch_release=True");
        TORCH_CHECK(NUM_DEVICES == 8,
                    "MXFP8 full-context/no-replay proof currently requires EP8");
    }
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_macrobatches = (schedule_capacity + macrobatch_size - 1) / macrobatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int intermediate_dim_col_blocks = intermediate_dim / config::MLP_Nb;
    const int hidden_dim_col_blocks = hidden_dim / config::MLP_Nb;
    const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
    const int minibatch_routed_swiglu_tiles =
        (minibatch_size / config::SWIGLU_Mb) *
        (intermediate_dim / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_bwd_tasks =
        (minibatch_routed_swiglu_tiles +
         config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) /
        (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
    const int minibatch_routed_bwd_tasks =
        minibatch_routed_row_blocks * intermediate_dim_col_blocks +
        minibatch_routed_swiglu_bwd_tasks +
        minibatch_routed_row_blocks * hidden_dim_col_blocks;
    const int segment_wgrad_tasks =
        3 * intermediate_dim_col_blocks * hidden_dim_col_blocks;
    TORCH_CHECK(minibatch_routed_bwd_tasks > 0,
                "MXFP8 minibatch must contain at least one routed BWD task");
    const int minibatch_task_owner_bucket_shift =
        31 - __builtin_clz(minibatch_routed_bwd_tasks);
    const int minibatch_task_owner_bucket_width =
        1 << minibatch_task_owner_bucket_shift;
    const bool build_ep8_o1_decode =
        use_minibatch_release && context_size > macrobatch_size;
    const bool build_ep8_wgrad_expert_row_prefix =
        build_ep8_o1_decode &&
        config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX;
    const int64_t max_completed_segments_i64 =
        static_cast<int64_t>(num_macrobatches) +
        max(0, num_local_experts - 1);
    const int64_t max_routed_tasks_i64 =
        static_cast<int64_t>(num_global_minibatches) *
            minibatch_routed_bwd_tasks +
        max_completed_segments_i64 * segment_wgrad_tasks;
    const int64_t owner_buckets_i64 =
        (max_routed_tasks_i64 + minibatch_task_owner_bucket_width - 1) /
        minibatch_task_owner_bucket_width;
    const int64_t minibatch_task_owner_bucket_capacity_i64 =
        build_ep8_o1_decode ? (owner_buckets_i64 > 0 ? owner_buckets_i64 : 1) : 0;
    TORCH_CHECK(!build_ep8_o1_decode ||
                    (max_completed_segments_i64 <= 0x7fffffffLL &&
                     max_routed_tasks_i64 <= 0x7fffffffLL &&
                     minibatch_task_owner_bucket_capacity_i64 <= 0x7fffffffLL),
                "EP8 full-context task-decode metadata exceeds int32 capacity");
    const int max_completed_segments =
        build_ep8_o1_decode
            ? static_cast<int>(max_completed_segments_i64)
            : 0;
    const int minibatch_task_owner_bucket_capacity =
        static_cast<int>(minibatch_task_owner_bucket_capacity_i64);

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl d_y_buffer_data;
    activation_bf16_pgl d_x_routed_buffer_data;
    router_weight_pgl router_weight_buffer_data;
    router_weight_pgl d_router_weight_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        d_y_buffer_data[i] = reinterpret_cast<bf16*>(d_y_buffer_ptrs[i]);
        d_x_routed_buffer_data[i] = reinterpret_cast<bf16*>(d_x_routed_buffer_ptrs[i]);
        router_weight_buffer_data[i] = reinterpret_cast<float*>(router_weight_buffer_ptrs[i]);
        d_router_weight_buffer_data[i] = reinterpret_cast<float*>(d_router_weight_buffer_ptrs[i]);
    }

    // Replayed forward activations
    at::Tensor x_fp8_routed = at::empty({macrobatch_size, hidden_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_routed = at::empty({macrobatch_size / 128, hidden_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor hidden_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor router_weights = at::empty(
        {context_size > macrobatch_size ? context_size : macrobatch_size},
        router_weight_buffer.options());
    at::Tensor d_router_weight_partials = at::empty({macrobatch_size, intermediate_dim / config::SWIGLU_Nb}, router_weight_buffer.options());

    // Gradient tensors
    at::Tensor d_y_fp8_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_y_sc_routed = at::empty({macrobatch_size / 128, hidden_dim / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_y_fp8_t_routed = at::empty({hidden_dim, macrobatch_size}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_y_sc_t_routed = at::empty({hidden_dim / 128, macrobatch_size / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_hidden_shared = at::empty({num_local_tokens, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_hidden_routed = at::empty({macrobatch_size, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_gate_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_gate_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_gate_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_gate_fp8_t_routed = at::empty({intermediate_dim, macrobatch_size}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_gate_sc_t_routed = at::empty({intermediate_dim / 128, macrobatch_size / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_up_fp8_routed = at::empty_like(d_gate_fp8_routed);
    at::Tensor d_up_sc_routed = at::empty_like(d_gate_sc_routed);
    at::Tensor d_up_fp8_t_routed = at::empty_like(d_gate_fp8_t_routed);
    at::Tensor d_up_sc_t_routed = at::empty_like(d_gate_sc_t_routed);
    at::Tensor d_up_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_x_shared = at::empty({num_local_tokens, hidden_dim}, d_y_buffer.options());
    at::Tensor d_x_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_gate = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_gate = at::empty(
        {num_local_experts, intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_up = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_up = at::empty(
        {num_local_experts, intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_down = at::empty({hidden_dim, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_down = at::empty(
        {num_local_experts, hidden_dim, intermediate_dim}, d_y_buffer.options());

    if (use_minibatch_release) {
        utils::zero_empty_routed_wgrads::globals g_zerw {
            .d_w_routed_gate = reinterpret_cast<uint16_t *>(d_w_routed_gate.data_ptr<at::BFloat16>()),
            .d_w_routed_up = reinterpret_cast<uint16_t *>(d_w_routed_up.data_ptr<at::BFloat16>()),
            .d_w_routed_down = reinterpret_cast<uint16_t *>(d_w_routed_down.data_ptr<at::BFloat16>()),
            .tokens_per_expert = tokens_per_expert.data_ptr<int>(),
            .elements_per_expert = d_w_routed_gate.numel() / num_local_experts,
            .macrobatch_size = macrobatch_size,
            .zero_split_experts = true
        };
        utils::zero_empty_routed_wgrads::zero_empty_routed_wgrads_kernel<<<
            dim3(128, num_local_experts), 256, 0,
            at::cuda::getCurrentCUDAStream()>>>(g_zerw);
    }

    // Counters
    at::Tensor d_y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor d_hidden_ready = at::zeros({(shared_row_blocks + routed_row_blocks) * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor d_gate_up_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor d_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_gate_up_ready = at::zeros({routed_row_blocks * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor replayed_hidden_ready = at::zeros({routed_row_blocks}, tokens_per_expert.options());
    at::Tensor routed_buffers_done = at::zeros(
        {use_minibatch_release ? num_global_minibatches : num_macrobatches}, tokens_per_expert.options());
    at::Tensor router_weights_ready = at::zeros(
        {use_minibatch_release ? num_global_minibatches : num_macrobatches}, tokens_per_expert.options());
    at::Tensor wgrad_read_consumed = use_minibatch_release
        ? at::zeros({num_global_minibatches}, tokens_per_expert.options())
        : routed_buffers_done;
    at::Tensor minibatch_expert_offsets = use_minibatch_release
        ? at::empty({num_global_minibatches + 1}, tokens_per_expert.options())
        : num_tokens;
    at::Tensor completed_segment_offsets = use_minibatch_release
        ? at::empty({num_global_minibatches + 1}, tokens_per_expert.options())
        : num_tokens;
    at::Tensor expert_row_offsets = build_ep8_wgrad_expert_row_prefix
        ? at::empty({num_local_experts + 1}, tokens_per_expert.options())
        : num_tokens;
    at::Tensor completed_segment_experts = build_ep8_o1_decode
        ? at::empty({max(1, max_completed_segments)}, tokens_per_expert.options())
        : num_tokens;
    at::Tensor minibatch_task_owner_buckets = build_ep8_o1_decode
        ? at::empty({minibatch_task_owner_bucket_capacity}, tokens_per_expert.options())
        : num_tokens;
    if (use_minibatch_release) {
        minibatch_expert_offsets_globals offsets_g {
            .minibatch_expert_offsets = kittens::py::tensor_to_gl<index_gl>(minibatch_expert_offsets),
            .completed_segment_offsets = kittens::py::tensor_to_gl<index_gl>(completed_segment_offsets),
            .expert_row_offsets = kittens::py::tensor_to_gl<index_gl>(expert_row_offsets),
            .completed_segment_experts = kittens::py::tensor_to_gl<index_gl>(completed_segment_experts),
            .minibatch_task_owner_buckets = kittens::py::tensor_to_gl<index_gl>(minibatch_task_owner_buckets),
            .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
            .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
            .num_local_experts = num_local_experts,
            .macrobatch_size = macrobatch_size,
            .minibatch_size = minibatch_size,
            .capacity_minibatches = num_global_minibatches,
            .completed_segment_capacity = build_ep8_o1_decode ? max_completed_segments : 0,
            .minibatch_task_owner_bucket_capacity = minibatch_task_owner_bucket_capacity,
            .minibatch_routed_bwd_tasks = build_ep8_o1_decode ? minibatch_routed_bwd_tasks : 0,
            .minibatch_task_owner_bucket_width = build_ep8_o1_decode ? minibatch_task_owner_bucket_width : 0,
            .segment_wgrad_tasks = build_ep8_o1_decode ? segment_wgrad_tasks : 0,
        };
        if (build_ep8_o1_decode) {
            if constexpr (config::EP8_FULL_CONTEXT_WGRAD_EXPERT_ROW_PREFIX) {
                kittens::py::global_kernel<
                    minibatch_expert_offsets_config,
                    minibatch_expert_offsets_globals,
                    build_minibatch_expert_offsets_kernel<true, true>>
                    <<<1, minibatch_expert_offsets_config::NUM_THREADS, 0,
                       at::cuda::getCurrentCUDAStream()>>>(offsets_g);
            } else {
                kittens::py::global_kernel<
                    minibatch_expert_offsets_config,
                    minibatch_expert_offsets_globals,
                    build_minibatch_expert_offsets_kernel<true, false>>
                    <<<1, minibatch_expert_offsets_config::NUM_THREADS, 0,
                       at::cuda::getCurrentCUDAStream()>>>(offsets_g);
            }
        } else {
            kittens::py::global_kernel<
                minibatch_expert_offsets_config,
                minibatch_expert_offsets_globals,
                build_minibatch_expert_offsets_kernel<false, false>>
                <<<1, minibatch_expert_offsets_config::NUM_THREADS, 0,
                   at::cuda::getCurrentCUDAStream()>>>(offsets_g);
        }
    }

    globals_bwd g {
        .x_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_routed),
        .x_sc_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_routed),
        .x_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_t_routed),
        .x_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_t_routed),
        .gate_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(gate_fp8_routed),
        .gate_sc_routed = kittens::py::tensor_to_gl<sc_gl>(gate_sc_routed),
        .up_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(up_fp8_routed),
        .up_sc_routed = kittens::py::tensor_to_gl<sc_gl>(up_sc_routed),
        .hidden_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_routed),
        .hidden_sc_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_routed),
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_t_routed),
        .hidden_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_t_routed),
        .d_y_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_y_buffer),
        .d_y_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_y_fp8_routed),
        .d_y_sc_routed = kittens::py::tensor_to_gl<sc_gl>(d_y_sc_routed),
        .d_y_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_y_fp8_t_routed),
        .d_y_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(d_y_sc_t_routed),
        .d_hidden_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_shared),
        .d_hidden_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_routed),
        .d_gate_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_gate_shared),
        .d_gate_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_gate_fp8_routed),
        .d_gate_sc_routed = kittens::py::tensor_to_gl<sc_gl>(d_gate_sc_routed),
        .d_gate_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_gate_fp8_t_routed),
        .d_gate_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(d_gate_sc_t_routed),
        .d_up_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_up_shared),
        .d_up_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_up_fp8_routed),
        .d_up_sc_routed = kittens::py::tensor_to_gl<sc_gl>(d_up_sc_routed),
        .d_up_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_up_fp8_t_routed),
        .d_up_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(d_up_sc_t_routed),
        .d_x_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_shared),
        .d_x_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .d_y_buffer = d_y_buffer_data,
        .d_x_routed_buffer = d_x_routed_buffer_data,
        .router_weight_buffer = router_weight_buffer_data,
        .d_router_weight_buffer = d_router_weight_buffer_data,
        .router_weights = kittens::py::tensor_to_gl<router_weight_gl>(router_weights),
        .d_router_weight_partials = kittens::py::tensor_to_gl<d_router_weight_partials_gl>(d_router_weight_partials),
        .w_routed_gate = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_gate),
        .w_routed_gate_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_gate_sc),
        .w_routed_up = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_up),
        .w_routed_up_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_up_sc),
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate_T = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_gate_T),
        .w_routed_gate_T_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_gate_T_sc),
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up_T = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_up_T),
        .w_routed_up_T_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_up_T_sc),
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down_T = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_down_T),
        .w_routed_down_T_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_down_T_sc),
        .d_w_shared_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_gate),
        .d_w_routed_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_gate),
        .d_w_shared_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_up),
        .d_w_routed_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_up),
        .d_w_shared_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_down),
        .d_w_routed_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_down),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .minibatch_expert_offsets = kittens::py::tensor_to_gl<index_gl>(minibatch_expert_offsets),
        .completed_segment_offsets = kittens::py::tensor_to_gl<index_gl>(completed_segment_offsets),
        .expert_row_offsets = kittens::py::tensor_to_gl<index_gl>(expert_row_offsets),
        .completed_segment_experts = kittens::py::tensor_to_gl<index_gl>(completed_segment_experts),
        .minibatch_task_owner_buckets = kittens::py::tensor_to_gl<index_gl>(minibatch_task_owner_buckets),
        .router_weights_ready = kittens::py::tensor_to_gl<index_gl>(router_weights_ready),
        .d_y_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_y_routed_ready),
        .d_hidden_ready = kittens::py::tensor_to_gl<index_gl>(d_hidden_ready),
        .d_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(d_gate_up_ready),
        .d_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_x_routed_ready),
        .replayed_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(replayed_x_routed_ready),
        .replayed_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(replayed_gate_up_ready),
        .replayed_hidden_ready = kittens::py::tensor_to_gl<index_gl>(replayed_hidden_ready),
        .routed_buffers_done = kittens::py::tensor_to_gl<index_gl>(routed_buffers_done),
        .wgrad_read_consumed = kittens::py::tensor_to_gl<index_gl>(wgrad_read_consumed),
        .topk = topk,
        .swiglu_limit = swiglu_limit.value_or(0.0f),
        .num_comm_sms = num_comm_sms,
        .context_size = context_size,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size,
        .minibatch_task_owner_bucket_shift = build_ep8_o1_decode ? minibatch_task_owner_bucket_shift : 0,
        .minibatch_release = use_minibatch_release ? 1 : 0
    };

    if (swiglu_limit.has_value()) {
        if (use_minibatch_release) {
            if constexpr (NUM_DEVICES == 8) {
                if (context_size > macrobatch_size)
                    kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_full_context_no_replay_kernel<true>>(g);
                else
                    kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_pipeline_kernel<true>>(g);
            } else {
                kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<true, true, false>>(g);
            }
        } else {
            kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<true, false, false>>(g);
        }
    } else {
        if (use_minibatch_release) {
            if constexpr (NUM_DEVICES == 8) {
                if (context_size > macrobatch_size)
                    kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_full_context_no_replay_kernel<false>>(g);
                else
                    kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_pipeline_kernel<false>>(g);
            } else {
                kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<false, true, false>>(g);
            }
        } else {
            kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<false, false, false>>(g);
        }
    }

    if (!use_minibatch_release) {
        utils::zero_empty_routed_wgrads::globals g_zerw {
            .d_w_routed_gate = reinterpret_cast<uint16_t *>(d_w_routed_gate.data_ptr<at::BFloat16>()),
            .d_w_routed_up = reinterpret_cast<uint16_t *>(d_w_routed_up.data_ptr<at::BFloat16>()),
            .d_w_routed_down = reinterpret_cast<uint16_t *>(d_w_routed_down.data_ptr<at::BFloat16>()),
            .tokens_per_expert = tokens_per_expert.data_ptr<int>(),
            .elements_per_expert = d_w_routed_gate.numel() / num_local_experts,
            .macrobatch_size = macrobatch_size,
            .zero_split_experts = false
        };

        utils::zero_empty_routed_wgrads::zero_empty_routed_wgrads_kernel<<<dim3(128, num_local_experts), 256, 0, at::cuda::getCurrentCUDAStream()>>>(g_zerw);
    }

    return {d_x_shared, d_x_routed,
            d_gate_shared, d_gate_fp8_routed, d_gate_sc_routed,
            d_up_shared, d_up_fp8_routed, d_up_sc_routed,
            d_hidden_shared, d_hidden_routed, d_y_fp8_routed, d_y_sc_routed,
            d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down};
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_bf16(
    const at::Tensor &d_y_buffer,
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_x_routed_buffer,
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &x_routed,
    const at::Tensor &gate_shared,
    const at::Tensor &gate_routed,
    const at::Tensor &up_shared,
    const at::Tensor &up_routed,
    const at::Tensor &hidden_shared,
    const at::Tensor &hidden_routed,
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size,
    bool use_minibatch_release
) {
    static_assert(!USE_MXFP8);
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_local_experts = w_routed_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_macrobatches = (schedule_capacity + macrobatch_size - 1) / macrobatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int intermediate_dim_col_blocks = intermediate_dim / config::MLP_Nb;

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl d_y_buffer_data;
    activation_bf16_pgl d_x_routed_buffer_data;
    router_weight_pgl router_weight_buffer_data;
    router_weight_pgl d_router_weight_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        d_y_buffer_data[i] = reinterpret_cast<bf16*>(d_y_buffer_ptrs[i]);
        d_x_routed_buffer_data[i] = reinterpret_cast<bf16*>(d_x_routed_buffer_ptrs[i]);
        router_weight_buffer_data[i] = reinterpret_cast<float*>(router_weight_buffer_ptrs[i]);
        d_router_weight_buffer_data[i] = reinterpret_cast<float*>(d_router_weight_buffer_ptrs[i]);
    }

    at::Tensor router_weights = at::empty({macrobatch_size}, router_weight_buffer.options());
    at::Tensor d_router_weight_partials = at::empty({macrobatch_size, intermediate_dim / config::SWIGLU_Nb}, router_weight_buffer.options());
    at::Tensor d_y_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options());
    at::Tensor d_hidden_shared = at::empty({num_local_tokens, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_hidden_routed = at::empty({macrobatch_size, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_gate_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_gate_routed = at::empty_like(d_hidden_routed);
    at::Tensor d_up_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_up_routed = at::empty_like(d_hidden_routed);
    at::Tensor d_x_shared = at::empty({num_local_tokens, hidden_dim}, d_y_buffer.options());
    at::Tensor d_x_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_gate = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_gate = at::empty(w_routed_gate.sizes(), w_routed_gate.options());
    at::Tensor d_w_shared_up = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_up = at::empty(w_routed_up.sizes(), w_routed_up.options());
    at::Tensor d_w_shared_down = at::empty({hidden_dim, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_down = at::empty(w_routed_down.sizes(), w_routed_down.options());

    if (use_minibatch_release) {
        utils::zero_empty_routed_wgrads::globals g_zerw {
            .d_w_routed_gate = reinterpret_cast<uint16_t *>(d_w_routed_gate.data_ptr<at::BFloat16>()),
            .d_w_routed_up = reinterpret_cast<uint16_t *>(d_w_routed_up.data_ptr<at::BFloat16>()),
            .d_w_routed_down = reinterpret_cast<uint16_t *>(d_w_routed_down.data_ptr<at::BFloat16>()),
            .tokens_per_expert = tokens_per_expert.data_ptr<int>(),
            .elements_per_expert = d_w_routed_gate.numel() / num_local_experts,
            .macrobatch_size = macrobatch_size,
            .zero_split_experts = true
        };
        utils::zero_empty_routed_wgrads::zero_empty_routed_wgrads_kernel<<<
            dim3(128, num_local_experts), 256, 0,
            at::cuda::getCurrentCUDAStream()>>>(g_zerw);
    }

    at::Tensor d_y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor d_hidden_ready = at::zeros({(shared_row_blocks + routed_row_blocks) * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor d_gate_up_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor d_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_gate_up_ready = at::zeros({routed_row_blocks * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor replayed_hidden_ready = at::zeros({routed_row_blocks}, tokens_per_expert.options());
    at::Tensor routed_buffers_done = at::zeros(
        {use_minibatch_release ? num_global_minibatches : num_macrobatches}, tokens_per_expert.options());
    at::Tensor router_weights_ready = at::zeros(
        {use_minibatch_release ? num_global_minibatches : num_macrobatches}, tokens_per_expert.options());
    at::Tensor wgrad_read_consumed = use_minibatch_release
        ? at::zeros({num_global_minibatches}, tokens_per_expert.options())
        : routed_buffers_done;
    at::Tensor minibatch_expert_offsets = use_minibatch_release
        ? at::empty({num_global_minibatches + 1}, tokens_per_expert.options())
        : num_tokens;
    at::Tensor completed_segment_offsets = use_minibatch_release
        ? at::empty({num_global_minibatches + 1}, tokens_per_expert.options())
        : num_tokens;
    if (use_minibatch_release) {
        minibatch_expert_offsets_globals offsets_g {
            .minibatch_expert_offsets = kittens::py::tensor_to_gl<index_gl>(minibatch_expert_offsets),
            .completed_segment_offsets = kittens::py::tensor_to_gl<index_gl>(completed_segment_offsets),
            .expert_row_offsets = kittens::py::tensor_to_gl<index_gl>(num_tokens),
            .completed_segment_experts = kittens::py::tensor_to_gl<index_gl>(num_tokens),
            .minibatch_task_owner_buckets = kittens::py::tensor_to_gl<index_gl>(num_tokens),
            .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
            .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
            .num_local_experts = num_local_experts,
            .macrobatch_size = macrobatch_size,
            .minibatch_size = minibatch_size,
            .capacity_minibatches = num_global_minibatches,
            .completed_segment_capacity = 0,
            .minibatch_task_owner_bucket_capacity = 0,
            .minibatch_routed_bwd_tasks = 0,
            .minibatch_task_owner_bucket_width = 0,
            .segment_wgrad_tasks = 0,
        };
        kittens::py::global_kernel<
            minibatch_expert_offsets_config,
            minibatch_expert_offsets_globals,
            build_minibatch_expert_offsets_kernel<false, false>>
            <<<1, minibatch_expert_offsets_config::NUM_THREADS, 0,
               at::cuda::getCurrentCUDAStream()>>>(offsets_g);
    }

    globals_bwd g {
        .x_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(x_routed),
        .x_sc_routed = {},
        .x_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(x_routed),
        .x_sc_t_routed = {},
        .gate_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(gate_routed),
        .gate_sc_routed = {},
        .up_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(up_routed),
        .up_sc_routed = {},
        .hidden_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(hidden_routed),
        .hidden_sc_routed = {},
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(hidden_routed),
        .hidden_sc_t_routed = {},
        .d_y_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_y_buffer),
        .d_y_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(d_y_routed),
        .d_y_sc_routed = {},
        .d_y_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(d_y_routed),
        .d_y_sc_t_routed = {},
        .d_hidden_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_shared),
        .d_hidden_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_routed),
        .d_gate_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_gate_shared),
        .d_gate_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(d_gate_routed),
        .d_gate_sc_routed = {},
        .d_gate_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(d_gate_routed),
        .d_gate_sc_t_routed = {},
        .d_up_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_up_shared),
        .d_up_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(d_up_routed),
        .d_up_sc_routed = {},
        .d_up_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(d_up_routed),
        .d_up_sc_t_routed = {},
        .d_x_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_shared),
        .d_x_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .d_y_buffer = d_y_buffer_data,
        .d_x_routed_buffer = d_x_routed_buffer_data,
        .router_weight_buffer = router_weight_buffer_data,
        .d_router_weight_buffer = d_router_weight_buffer_data,
        .router_weights = kittens::py::tensor_to_gl<router_weight_gl>(router_weights),
        .d_router_weight_partials = kittens::py::tensor_to_gl<d_router_weight_partials_gl>(d_router_weight_partials),
        .w_routed_gate = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_routed_gate_sc = {},
        .w_routed_up = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_routed_up_sc = {},
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate_T = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_routed_gate_T_sc = {},
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up_T = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_routed_up_T_sc = {},
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down_T = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_down),
        .w_routed_down_T_sc = {},
        .d_w_shared_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_gate),
        .d_w_routed_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_gate),
        .d_w_shared_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_up),
        .d_w_routed_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_up),
        .d_w_shared_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_down),
        .d_w_routed_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_down),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .minibatch_expert_offsets = kittens::py::tensor_to_gl<index_gl>(minibatch_expert_offsets),
        .completed_segment_offsets = kittens::py::tensor_to_gl<index_gl>(completed_segment_offsets),
        .expert_row_offsets = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .completed_segment_experts = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .minibatch_task_owner_buckets = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .router_weights_ready = kittens::py::tensor_to_gl<index_gl>(router_weights_ready),
        .d_y_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_y_routed_ready),
        .d_hidden_ready = kittens::py::tensor_to_gl<index_gl>(d_hidden_ready),
        .d_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(d_gate_up_ready),
        .d_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_x_routed_ready),
        .replayed_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(replayed_x_routed_ready),
        .replayed_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(replayed_gate_up_ready),
        .replayed_hidden_ready = kittens::py::tensor_to_gl<index_gl>(replayed_hidden_ready),
        .routed_buffers_done = kittens::py::tensor_to_gl<index_gl>(routed_buffers_done),
        .wgrad_read_consumed = kittens::py::tensor_to_gl<index_gl>(wgrad_read_consumed),
        .topk = topk,
        .swiglu_limit = swiglu_limit.value_or(0.0f),
        .num_comm_sms = num_comm_sms,
        // BF16 full-context/no-replay is intentionally N/A in this proof.
        .context_size = macrobatch_size,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size,
        .minibatch_task_owner_bucket_shift = 0,
        .minibatch_release = use_minibatch_release ? 1 : 0
    };

    if (swiglu_limit.has_value()) {
        if (use_minibatch_release) {
            if constexpr (NUM_DEVICES == 8)
                kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_pipeline_kernel<true>>(g);
            else
                kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<true, true, false>>(g);
        } else {
            kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<true, false, false>>(g);
        }
    } else {
        if (use_minibatch_release) {
            if constexpr (NUM_DEVICES == 8)
                kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_ep8_minibatch_pipeline_kernel<false>>(g);
            else
                kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<false, true, false>>(g);
        } else {
            kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel<false, false, false>>(g);
        }
    }

    if (!use_minibatch_release) {
        utils::zero_empty_routed_wgrads::globals g_zerw {
            .d_w_routed_gate = reinterpret_cast<uint16_t *>(d_w_routed_gate.data_ptr<at::BFloat16>()),
            .d_w_routed_up = reinterpret_cast<uint16_t *>(d_w_routed_up.data_ptr<at::BFloat16>()),
            .d_w_routed_down = reinterpret_cast<uint16_t *>(d_w_routed_down.data_ptr<at::BFloat16>()),
            .tokens_per_expert = tokens_per_expert.data_ptr<int>(),
            .elements_per_expert = d_w_routed_gate.numel() / num_local_experts,
            .macrobatch_size = macrobatch_size,
            .zero_split_experts = false
        };

        utils::zero_empty_routed_wgrads::zero_empty_routed_wgrads_kernel<<<dim3(128, num_local_experts), 256, 0, at::cuda::getCurrentCUDAStream()>>>(g_zerw);
    }

    return {d_x_shared, d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed,
            d_hidden_shared, d_hidden_routed, d_y_routed,
            d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down};
}
