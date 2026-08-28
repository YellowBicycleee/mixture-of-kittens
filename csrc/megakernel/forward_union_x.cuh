// Parallel EP8 BF16 Forward globals for persistent rank-token Union-X.
// This type is intentionally separate from globals_fwd so the legacy ABI and
// dispatch/X-ring path remain byte-for-byte untouched.
struct globals_union_x_fwd {
    mlp_bf16_gl x_shared;                 // (T, H), shared-expert FC1 A
    routed_bf16_gl union_x;               // (8 * T, H), persistent through backward
    CUtensorMap union_x_gather4_tma;       // 2D [H, U], K64 x one-row box
    index_gl union_state;                 // flat (8 * T * ceil(H / 512),)
    index_gl route_to_union;               // (schedule_capacity,), padding = -1

    epi_bf16_gl gate_shared;              // (T, I)
    routed_gate_up_gl gate_routed;         // (macrobatch_size, I), macro-0 context
    epi_bf16_gl up_shared;                // (T, I)
    routed_gate_up_gl up_routed;           // (macrobatch_size, I), macro-0 context
    mlp_bf16_gl hidden_shared;            // (T, I)
    routed_activation_gl hidden_routed;   // (macrobatch_size, I)
    epi_bf16_gl y_shared;                 // (T, H)
    epi_bf16_gl y_routed;                 // (macrobatch_size, H)

    activation_bf16_pgl x_routed_send_buffer;
    activation_bf16_pgl y_routed_recv_buffer;

    weight_bf16_gl w_shared_gate;
    routed_weight_gl w_routed_gate;
    weight_bf16_gl w_shared_up;
    routed_weight_gl w_routed_up;
    weight_bf16_gl w_shared_down;
    routed_weight_gl w_routed_down;

    index_gl schedule_peer_rank;
    index_gl schedule_peer_token_idx;
    index_gl num_tokens;
    index_gl tokens_per_expert;

    index_gl hidden_row_block_ready;
    index_gl union_x_ready;                // coarse route-slice arrivals per minibatch
    index_gl y_routed_ready;
    index_gl y_routed_done;

    const int topk;
    const float swiglu_limit;
    const int num_comm_sms;
    const int macrobatch_size;
    const int minibatch_size;

    __host__ inline dim3 grid() const {
        const int num_minibatches =
            (schedule_peer_rank.cols() + minibatch_size - 1) / minibatch_size;
        const int shared_row_blocks = x_shared.rows() / config::MLP_Mb;
        const int minibatch_routed_row_blocks =
            minibatch_size / config::MLP_Mb;
        const int shared_gate_up_raw_tasks =
            shared_row_blocks * (w_shared_gate.rows() / config::SWIGLU_Nb);
        const int minibatch_routed_gate_up_raw_tasks =
            minibatch_routed_row_blocks
            * (w_routed_gate.rows() / UNION_X_RGU_PACKED_HIDDEN_N);
        const int shared_gate_up_tasks =
            (shared_gate_up_raw_tasks
                + config::FUSED_GATE_UP_TASK_GROUP_SIZE - 1)
            / config::FUSED_GATE_UP_TASK_GROUP_SIZE;
        const int minibatch_routed_gate_up_tasks =
            (minibatch_routed_gate_up_raw_tasks
                + config::FUSED_GATE_UP_TASK_GROUP_SIZE - 1)
            / config::FUSED_GATE_UP_TASK_GROUP_SIZE;
        const int shared_down_raw_tasks =
            shared_row_blocks * (w_shared_down.rows() / config::MLP_Nb);
        const int minibatch_routed_down_raw_tasks =
            minibatch_routed_row_blocks
            * (w_routed_down.rows() / config::MLP_Nb);
        const int shared_down_tasks =
            (shared_down_raw_tasks
                + config::FUSED_DOWN_TASK_GROUP_SIZE - 1)
            / config::FUSED_DOWN_TASK_GROUP_SIZE;
        const int minibatch_routed_down_tasks =
            (minibatch_routed_down_raw_tasks
                + config::FUSED_DOWN_TASK_GROUP_SIZE - 1)
            / config::FUSED_DOWN_TASK_GROUP_SIZE;
        const int shared_tasks = shared_gate_up_tasks + shared_down_tasks;
        const int minibatch_tasks =
            minibatch_routed_gate_up_tasks + minibatch_routed_down_tasks;
        return dim3(
            config::CLUSTER_SIZE
                * (shared_tasks + num_minibatches * minibatch_tasks)
            + num_comm_sms);
    }
};

// Keep Dispatch/readiness at the configured minibatch granularity, but order
// routed compute in stage-major windows of up to 16K rows.  This preserves all
// task identities and dependency counts while avoiding a Gate/Down role switch
// after every 4K communication minibatch.
static constexpr int UNION_X_COMPUTE_GROUP_ROWS = 16384;

template <bool IS_CLAMPED>
static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_fwd_union_x_kernel(
    const globals_union_x_fwd &g
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);
    static_assert(FWD_CLC_PIPE_DEPTH == 2);
    static_assert(FWD_GATE_GROUP_SIZE == 1);
    static_assert(FWD_DOWN_GROUP_SIZE == 1);
    // The EP8 BF16 forward selector may keep two CLC result slots so the
    // scheduler can obtain task t+2 without waiting for the consumer tail of
    // task t.  Every other path keeps its original single-result schedule.
    static_assert(FWD_CLC_PIPE_DEPTH > 0);
    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    // Shared raw tasks retain one MLP_Mb x SWIGLU_Nb hidden tile.  Routed
    // Union-X raw tasks pair two adjacent SWIGLU_Nb tiles so one gathered A
    // feeds two independent Gate|Up accumulators.  Each old hidden tile keeps
    // its own store and ready arrival.
    const int shared_gate_up_raw_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::SWIGLU_Nb);
    const int minibatch_routed_gate_up_raw_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / UNION_X_RGU_PACKED_HIDDEN_N);
    const int shared_gate_up_tasks =
        (shared_gate_up_raw_tasks + config::FUSED_GATE_UP_TASK_GROUP_SIZE - 1)
        / config::FUSED_GATE_UP_TASK_GROUP_SIZE;
    const int minibatch_routed_gate_up_tasks =
        (minibatch_routed_gate_up_raw_tasks + config::FUSED_GATE_UP_TASK_GROUP_SIZE - 1)
        / config::FUSED_GATE_UP_TASK_GROUP_SIZE;
    const int shared_down_raw_tasks = shared_row_blocks * (g.w_shared_down.rows() / config::MLP_Nb);
    const int minibatch_routed_down_raw_tasks = minibatch_routed_row_blocks * (g.w_routed_down.rows() / config::MLP_Nb);
    const int shared_down_tasks =
        (shared_down_raw_tasks + config::FUSED_DOWN_TASK_GROUP_SIZE - 1)
        / config::FUSED_DOWN_TASK_GROUP_SIZE;
    const int minibatch_routed_down_tasks =
        (minibatch_routed_down_raw_tasks + config::FUSED_DOWN_TASK_GROUP_SIZE - 1)
        / config::FUSED_DOWN_TASK_GROUP_SIZE;
    const int shared_tasks = shared_gate_up_tasks + shared_down_tasks;
    const int minibatch_tasks = minibatch_routed_gate_up_tasks + minibatch_routed_down_tasks;
    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int macrobatch_size = g.macrobatch_size;

    const int num_tokens = g.num_tokens[{0}];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
    const int compute_minibatches_per_group =
        UNION_X_COMPUTE_GROUP_ROWS % g.minibatch_size == 0
        ? min(
            minibatches_per_macrobatch,
            max(1, UNION_X_COMPUTE_GROUP_ROWS / g.minibatch_size))
        : 1;
    const int true_num_global_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int last_macrobatch_num_minibatches = true_num_global_minibatches - (num_macrobatches - 1) * minibatches_per_macrobatch;
    const int true_num_clusters = comm_clusters + shared_tasks + true_num_global_minibatches * minibatch_tasks;
    if (cluster_idx >= true_num_clusters) return;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t dispatch_bitfield = 0xFFFF0000;
    uint32_t combine_bitfield = 0xFFFF0000;
    uint32_t union_a_reuse_bitfield = 0xFFFF0000;
    uint32_t union_a_tma_bitfield = 0xFFFF0000;
    uint32_t union_a_gather_bitfield = 0xFFFF0000;
    uint32_t union_b_arm_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[FWD_CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[FWD_CLC_PIPE_DEPTH], schedule_finished[FWD_CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_finished[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore union_a_inputs_reusable[UNION_X_RGU_STAGES];
    __shared__ semaphore union_a_tma_arrived[UNION_X_RGU_STAGES];
    __shared__ semaphore union_a_arrived[UNION_X_RGU_STAGES];
    __shared__ semaphore union_b_inputs_armed[UNION_X_RGU_STAGES];
    __shared__ semaphore gemm_scales_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_inputs_arrived;
    __shared__ semaphore combine_inputs_arrived[config::COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
            init_semaphore(gemm_scales_arrived[i], 0, 1);
            init_semaphore(gemm_inputs_finished[i], 0, 1);
            init_semaphore(gemm_scales_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < UNION_X_RGU_STAGES; ++i)
            init_semaphore(union_a_inputs_reusable[i], 0, 1);
        #pragma unroll
        for (int i = 0; i < UNION_X_RGU_STAGES; ++i)
            init_semaphore(union_a_tma_arrived[i], 0, 1);
        #pragma unroll
        for (int i = 0; i < UNION_X_RGU_STAGES; ++i)
            init_semaphore(union_a_arrived[i], 0, config::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < UNION_X_RGU_STAGES; ++i)
            init_semaphore(union_b_inputs_armed[i], 0, 1);
        init_semaphore(gemm_outputs_arrived, 0, 1);
        init_semaphore(gemm_outputs_finished, 0, config::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < FWD_CLC_PIPE_DEPTH; ++i) {
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
    static_assert(
        2 * config::MLP_Nb <= decltype(tm_alloc)::cols,
        "routed Union-X N512 requires two BF16 accumulator halves");
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    // This specialization is BF16-only, so the MXFP8 scale half is unused and
    // can hold the paired logical-N accumulator, as in paired BF16 Replay.
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> paired_d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(256);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH>>(256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH>>(384);
    everyone::tma::cluster::sync();

    if (cluster_idx >= comm_clusters
        && threadIdx.x / WARP_THREADS == UNION_X_RGU_A_TMA_WARP)
        union_x_prefetch_gather4_tma(g.union_x_gather4_tma);

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        auto num_dispatch_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int dispatch_col_blocks = UNION_X_DISPATCH_HIDDEN_SLICES;
            return (macrobatch_tokens / config::DISPATCH_Mb) * dispatch_col_blocks;
        };
        auto num_combine_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int combine_col_blocks = (g.y_routed.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
            const int combine_tiles = (macrobatch_tokens / config::COMBINE_Mb) * combine_col_blocks;
            return (combine_tiles + config::COMBINE_PIPE_DEPTH - 1) / config::COMBINE_PIPE_DEPTH;
        };
        auto dispatch = [&](int macrobatch_idx, int task_idx) {
            union_x_dispatch_task(
                g.x_routed_send_buffer,
                g.union_x,
                g.union_state,
                g.route_to_union,
                g.schedule_peer_rank,
                g.schedule_peer_token_idx,
                g.union_x_ready,
                dispatch_inputs_arrived,
                dispatch_bitfield,
                num_tokens,
                macrobatch_size,
                g.minibatch_size,
                macrobatch_idx,
                task_idx,
                g.topk,
                smem_base_addr);
        };
        auto combine = [&](int macrobatch_idx, int task_idx) {
            combine_kernel<true>(g.y_routed_recv_buffer, g.y_routed, nullptr, nullptr,
                                 g.schedule_peer_rank, g.schedule_peer_token_idx,
                                 g.y_routed_ready, macrobatch_idx > 0 ? &g.y_routed_done : nullptr,
                                 combine_inputs_arrived, combine_bitfield,
                                 num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, smem_base_addr);
        };
        if (num_macrobatches == 0) return;
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks(num_macrobatches - 1); task_idx += g.num_comm_sms)
            dispatch(num_macrobatches - 1, task_idx);
        for (int macrobatch_idx = num_macrobatches - 1; macrobatch_idx >= 0; --macrobatch_idx) {
            // Union-X is persistent rather than a reusable X ring.  Resolve
            // every route-slice assigned to this COMM CTA for the next macro
            // before entering combine, whose y-ready wait can block.
            if (macrobatch_idx > 0) {
                for (int task_idx = comm_cta_idx;
                     task_idx < num_dispatch_tasks(macrobatch_idx - 1);
                     task_idx += g.num_comm_sms)
                    dispatch(macrobatch_idx - 1, task_idx);
            }
            for (int task_idx = comm_cta_idx;
                 task_idx < num_combine_tasks(macrobatch_idx);
                 task_idx += g.num_comm_sms)
                combine(macrobatch_idx, task_idx);
        }
        return;
    }

    // Fused Gate/Up/SwiGLU and Down are both cluster-cooperative tasks.
    auto is_cta_local_task = [&](int) { return false; };
    const int hidden_row_block_ready_required_count = (config::MLP_Mb / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % FWD_CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + FWD_CLC_PIPE_DEPTH) / FWD_CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);

        if (compute_cluster_idx < shared_gate_up_tasks) {
            // Shared Gate + Up + SwiGLU (BF16). Shared preactivations are
            // retained because shared-expert backward does not replay them.
            const int task_idx = compute_cluster_idx;
            expert_gate_up_swiglu_ep8_tuned_kernel<true, IS_CLAMPED>(
                g.x_shared, g.w_shared_gate, g.w_shared_up,
                nullptr, nullptr, nullptr,
                g.gate_shared, nullptr, g.up_shared, nullptr,
                g.hidden_shared, nullptr, nullptr, nullptr,
                g.tokens_per_expert, nullptr, g.hidden_row_block_ready,
                d_tt, a_sc_tt, b_sc_tt,
                gemm_inputs_arrived, gemm_scales_arrived,
                gemm_inputs_finished, gemm_scales_finished,
                gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                num_tokens, g.swiglu_limit, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared down (BF16)
            const int task_group_idx = compute_cluster_idx - shared_gate_up_tasks;
            #pragma unroll 1
            for (int task_group_offset = 0;
                 task_group_offset < config::FUSED_DOWN_TASK_GROUP_SIZE;
                 ++task_group_offset) {
                const int task_idx =
                    task_group_idx * config::FUSED_DOWN_TASK_GROUP_SIZE
                    + task_group_offset;
                expert_grouped_gemm_kernel<true, false, false, false, false, false, false,
                                           FUSED_GATE_UP_LOAD_PIPE_DEPTH>(g.hidden_shared, g.w_shared_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                          g.y_shared, nullptr, nullptr,
                                          g.tokens_per_expert, nullptr, &g.hidden_row_block_ready, nullptr, nullptr, nullptr, nullptr,
                                          d_tt, a_sc_tt, b_sc_tt,
                                          gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                          num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                          0, 0, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
            }
        } else {
            // Routed expert with macro/minibatching
            const int routed_task_order = compute_cluster_idx - shared_tasks;
            const int last_macrobatch_tasks =
                last_macrobatch_num_minibatches * minibatch_tasks;
            const int full_macrobatch_tasks =
                minibatches_per_macrobatch * minibatch_tasks;
            int macrobatch_idx;
            int macrobatch_num_minibatches;
            int macrobatch_task_order;
            if (routed_task_order < last_macrobatch_tasks) {
                macrobatch_idx = num_macrobatches - 1;
                macrobatch_num_minibatches =
                    last_macrobatch_num_minibatches;
                macrobatch_task_order = routed_task_order;
            } else {
                const int full_macrobatch_task_order =
                    routed_task_order - last_macrobatch_tasks;
                macrobatch_idx = num_macrobatches - 2
                    - full_macrobatch_task_order / full_macrobatch_tasks;
                macrobatch_num_minibatches = minibatches_per_macrobatch;
                macrobatch_task_order =
                    full_macrobatch_task_order % full_macrobatch_tasks;
            }

            const int full_group_tasks =
                compute_minibatches_per_group * minibatch_tasks;
            const int num_full_groups =
                macrobatch_num_minibatches / compute_minibatches_per_group;
            const int full_groups_tasks =
                num_full_groups * full_group_tasks;
            int group_minibatch_start;
            int group_num_minibatches;
            int group_task_order;
            if (macrobatch_task_order < full_groups_tasks) {
                const int group_idx =
                    macrobatch_task_order / full_group_tasks;
                group_minibatch_start =
                    group_idx * compute_minibatches_per_group;
                group_num_minibatches = compute_minibatches_per_group;
                group_task_order =
                    macrobatch_task_order - group_idx * full_group_tasks;
            } else {
                group_minibatch_start =
                    num_full_groups * compute_minibatches_per_group;
                group_num_minibatches =
                    macrobatch_num_minibatches - group_minibatch_start;
                group_task_order =
                    macrobatch_task_order - full_groups_tasks;
            }

            const int group_gate_up_tasks =
                group_num_minibatches * minibatch_routed_gate_up_tasks;
            const bool is_routed_gate_up =
                group_task_order < group_gate_up_tasks;
            int minibatch_idx;
            int minibatch_task_idx;
            if (is_routed_gate_up) {
                minibatch_idx = group_minibatch_start
                    + group_task_order / minibatch_routed_gate_up_tasks;
                minibatch_task_idx =
                    group_task_order % minibatch_routed_gate_up_tasks;
            } else {
                const int group_down_task_order =
                    group_task_order - group_gate_up_tasks;
                minibatch_idx = group_minibatch_start
                    + group_down_task_order / minibatch_routed_down_tasks;
                minibatch_task_idx = minibatch_routed_gate_up_tasks
                    + group_down_task_order % minibatch_routed_down_tasks;
            }

            if (is_routed_gate_up) {
                // Routed Gate + Up + SwiGLU. Only macrobatch 0 retains
                // preactivations/transpose context; hidden normal is written
                // for every macrobatch so Down can consume it immediately.
                const int task_idx = minibatch_task_idx;
                expert_gate_up_swiglu_union_x_bf16_kernel<IS_CLAMPED>(
                    g.union_x,
                    g.union_x_gather4_tma,
                    g.route_to_union,
                    g.w_routed_gate,
                    g.w_routed_up,
                    g.gate_routed,
                    g.up_routed,
                    g.hidden_routed,
                    g.tokens_per_expert,
                    g.union_x_ready,
                    g.hidden_row_block_ready,
                    d_tt,
                    paired_d_tt,
                    gemm_inputs_arrived,
                    union_b_inputs_armed,
                    union_a_inputs_reusable,
                    union_a_tma_arrived,
                    gemm_inputs_finished,
                    union_a_arrived,
                    gemm_outputs_arrived,
                    gemm_outputs_finished,
                    gemm_bitfield,
                    union_a_reuse_bitfield,
                    union_a_tma_bitfield,
                    union_a_gather_bitfield,
                    union_b_arm_bitfield,
                    num_tokens,
                    g.swiglu_limit,
                    macrobatch_size,
                    g.minibatch_size,
                    macrobatch_idx,
                    minibatch_idx,
                    task_idx,
                    cta_rank,
                    shared_row_blocks,
                    smem_base_addr);
            } else {
                // Routed down
                const int task_group_idx =
                    minibatch_task_idx - minibatch_routed_gate_up_tasks;
                #pragma unroll 1
                for (int task_group_offset = 0;
                     task_group_offset < config::FUSED_DOWN_TASK_GROUP_SIZE;
                     ++task_group_offset) {
                    const int task_idx =
                        task_group_idx * config::FUSED_DOWN_TASK_GROUP_SIZE
                        + task_group_offset;
                    expert_grouped_gemm_kernel<false, false, false, false, false, false, false,
                                               FUSED_GATE_UP_LOAD_PIPE_DEPTH>(g.hidden_routed, g.w_routed_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                               g.y_routed, nullptr, nullptr,
                                               g.tokens_per_expert, nullptr, &g.hidden_row_block_ready,
                                               macrobatch_idx + 1 < num_macrobatches ? &g.y_routed_done : nullptr,
                                               nullptr, &g.y_routed_ready, nullptr,
                                               d_tt, a_sc_tt, b_sc_tt,
                                               gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, shared_row_blocks, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
                }
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / FWD_CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_compute_cluster_idx))
            everyone::tma::cluster::sync();
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


static __host__ __forceinline__ std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_bf16_union_x(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &route_to_union,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);
    static_assert(FWD_CLC_PIPE_DEPTH == 2);
    static_assert(FWD_GATE_GROUP_SIZE == 1);
    static_assert(FWD_DOWN_GROUP_SIZE == 1);

    CHECK_INPUT(x);
    CHECK_INPUT(combine_buffer);
    CHECK_INPUT(w_shared_gate);
    CHECK_INPUT(w_routed_gate);
    CHECK_INPUT(w_shared_up);
    CHECK_INPUT(w_routed_up);
    CHECK_INPUT(w_shared_down);
    CHECK_INPUT(w_routed_down);
    CHECK_INPUT(schedule_peer_rank);
    CHECK_INPUT(schedule_peer_token_idx);
    CHECK_INPUT(route_to_union);
    CHECK_INPUT(num_tokens);
    CHECK_INPUT(tokens_per_expert);

    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_local_experts = w_routed_gate.size(0);
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 && x.dim() == 2
            && num_local_tokens >= 2 * config::MLP_Mb
            && num_local_tokens % config::MLP_Mb == 0
            && hidden_dim == UNION_X_DISPATCH_HIDDEN,
        "Union-X forward requires BF16 x [T, 4096], T >= 512 and "
        "divisible by 256");
    TORCH_CHECK(
        x_ptrs.size() == NUM_DEVICES
            && combine_buffer_ptrs.size() == NUM_DEVICES,
        "Union-X forward requires exactly 8 peer pointer entries");
    TORCH_CHECK(
        schedule_peer_rank.scalar_type() == at::kInt
            && schedule_peer_rank.dim() == 1
            && schedule_capacity > 0
            && schedule_capacity % config::MLP_Mb == 0
            && schedule_capacity
                >= static_cast<int64_t>(num_local_tokens) * topk
            && schedule_peer_token_idx.scalar_type() == at::kInt
            && schedule_peer_token_idx.sizes()
                == schedule_peer_rank.sizes()
            && route_to_union.scalar_type() == at::kInt
            && route_to_union.sizes() == schedule_peer_rank.sizes(),
        "Union-X schedule and route map must be equal-length int32 vectors "
        "with capacity divisible by 256 and at least T * topk");
    TORCH_CHECK(
        num_tokens.scalar_type() == at::kInt
            && num_tokens.numel() == 1
            && tokens_per_expert.scalar_type() == at::kInt
            && tokens_per_expert.dim() == 1
            && tokens_per_expert.size(0) == num_local_experts,
        "Union-X count tensors must be int32 [1] and [num_local_experts]");
    TORCH_CHECK(
        w_shared_gate.scalar_type() == at::kBFloat16
            && w_shared_gate.dim() == 2
            && w_shared_gate.size(1) == hidden_dim
            && intermediate_dim > 0
            && intermediate_dim % config::MLP_Nb == 0
            && w_shared_up.sizes() == w_shared_gate.sizes()
            && w_shared_up.scalar_type() == at::kBFloat16,
        "Union-X shared Gate/Up weights must be matching BF16 [I, 4096]");
    TORCH_CHECK(
        w_routed_gate.scalar_type() == at::kBFloat16
            && w_routed_gate.dim() == 3
            && num_local_experts > 0
            && w_routed_gate.size(1) == intermediate_dim
            && w_routed_gate.size(2) == hidden_dim
            && w_routed_up.sizes() == w_routed_gate.sizes()
            && w_routed_up.scalar_type() == at::kBFloat16,
        "Union-X routed Gate/Up weights must be matching BF16 [E, I, 4096]");
    TORCH_CHECK(
        w_shared_down.scalar_type() == at::kBFloat16
            && w_shared_down.dim() == 2
            && w_shared_down.size(0) == hidden_dim
            && w_shared_down.size(1) == intermediate_dim
            && w_routed_down.scalar_type() == at::kBFloat16
            && w_routed_down.dim() == 3
            && w_routed_down.size(0) == num_local_experts
            && w_routed_down.size(1) == hidden_dim
            && w_routed_down.size(2) == intermediate_dim,
        "Union-X Down weights must be BF16 [4096, I] and [E, 4096, I]");
    TORCH_CHECK(
        combine_buffer.scalar_type() == at::kBFloat16
            && combine_buffer.dim() == 2
            && combine_buffer.size(0)
                == static_cast<int64_t>(num_local_tokens) * topk
            && combine_buffer.size(1) == hidden_dim,
        "Union-X combine buffer must be BF16 [T * topk, 4096]");
    TORCH_CHECK(
        topk > 0 && topk <= 255
            && num_comm_sms > 0 && num_comm_sms % config::CLUSTER_SIZE == 0
            && minibatch_size > 0
            && minibatch_size % config::MLP_Mb == 0
            && macrobatch_size > 0
            && macrobatch_size % minibatch_size == 0,
        "invalid Union-X topk, COMM-SM, macro, or minibatch configuration");
    const auto *device_properties =
        at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(
        num_comm_sms < device_properties->multiProcessorCount,
        "Union-X num_comm_sms must leave at least one compute SM");

    const at::Device device = x.device();
    for (const at::Tensor *tensor : {
             &combine_buffer, &w_shared_gate, &w_routed_gate, &w_shared_up,
             &w_routed_up, &w_shared_down, &w_routed_down,
             &schedule_peer_rank, &schedule_peer_token_idx, &route_to_union,
             &num_tokens, &tokens_per_expert}) {
        TORCH_CHECK(tensor->device() == device,
                    "all Union-X forward tensors must share one CUDA device");
    }

    activation_bf16_pgl x_send;
    activation_bf16_pgl y_recv;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        TORCH_CHECK(x_ptrs[i] > 0 && combine_buffer_ptrs[i] > 0,
                    "Union-X peer pointers must be positive");
        x_send[i] = reinterpret_cast<bf16 *>(x_ptrs[i]);
        y_recv[i] = reinterpret_cast<bf16 *>(combine_buffer_ptrs[i]);
    }

    const int64_t union_capacity =
        static_cast<int64_t>(NUM_DEVICES) * num_local_tokens;
    const int num_global_minibatches =
        (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_global_row_blocks =
        schedule_capacity / (config::MLP_Mb / config::CLUSTER_SIZE);
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;

    at::Tensor union_x =
        at::empty({union_capacity, hidden_dim}, x.options());
    at::Tensor union_state = at::zeros(
        {union_capacity * UNION_X_DISPATCH_HIDDEN_SLICES},
        tokens_per_expert.options());
    at::Tensor gate_shared =
        at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed =
        at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_shared =
        at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed =
        at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor hidden_shared =
        at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor hidden_routed =
        at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed =
        at::empty({macrobatch_size, hidden_dim}, x.options());
    at::Tensor union_x_ready =
        at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros(
        {shared_row_blocks + routed_row_blocks},
        tokens_per_expert.options());
    at::Tensor y_routed_ready =
        at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor y_routed_done =
        at::zeros({num_global_row_blocks}, tokens_per_expert.options());

    globals_union_x_fwd g {
        .x_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(x),
        .union_x = kittens::py::tensor_to_gl<routed_bf16_gl>(union_x),
        .union_x_gather4_tma = union_x_make_gather4_tma(union_x),
        .union_state = kittens::py::tensor_to_gl<index_gl>(union_state),
        .route_to_union =
            kittens::py::tensor_to_gl<index_gl>(route_to_union),
        .gate_shared =
            kittens::py::tensor_to_gl<epi_bf16_gl>(gate_shared),
        .gate_routed =
            kittens::py::tensor_to_gl<routed_gate_up_gl>(gate_routed),
        .up_shared =
            kittens::py::tensor_to_gl<epi_bf16_gl>(up_shared),
        .up_routed =
            kittens::py::tensor_to_gl<routed_gate_up_gl>(up_routed),
        .hidden_shared =
            kittens::py::tensor_to_gl<mlp_bf16_gl>(hidden_shared),
        .hidden_routed =
            kittens::py::tensor_to_gl<routed_activation_gl>(hidden_routed),
        .y_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(y_routed),
        .x_routed_send_buffer = x_send,
        .y_routed_recv_buffer = y_recv,
        .w_shared_gate =
            kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate =
            kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_shared_up =
            kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up =
            kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_shared_down =
            kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down =
            kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_down),
        .schedule_peer_rank =
            kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx =
            kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert =
            kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .hidden_row_block_ready =
            kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .union_x_ready =
            kittens::py::tensor_to_gl<index_gl>(union_x_ready),
        .y_routed_ready =
            kittens::py::tensor_to_gl<index_gl>(y_routed_ready),
        .y_routed_done =
            kittens::py::tensor_to_gl<index_gl>(y_routed_done),
        .topk = topk,
        .swiglu_limit = swiglu_limit.value_or(0.0f),
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size,
    };

    if (swiglu_limit.has_value()) {
        kittens::py::launch_kernel<
            config,
            globals_union_x_fwd,
            dispatch_mlp_swiglu_combine_fwd_union_x_kernel<true>>(g);
    } else {
        kittens::py::launch_kernel<
            config,
            globals_union_x_fwd,
            dispatch_mlp_swiglu_combine_fwd_union_x_kernel<false>>(g);
    }
    return {
        union_x,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        y_shared,
        y_routed,
    };
}
