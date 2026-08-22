// Forward macro-0 routed MXFP8 Down shares the compact three-stage operand
// ring with fused Gate/Up.  grouped_gemm aligns its output scratch up to 1 KiB,
// so account for that 512-byte internal pad and the outer base's worst-case
// 1023-byte alignment shift explicitly here without changing the generic
// depth-6 Backward/Recompute instantiations.
static constexpr uint64_t FWD_MACRO0_MXFP8_DOWN_RING_BYTES =
    FUSED_GATE_UP_MACRO0_MXFP8_LOAD_PIPE_DEPTH
    * (2 * sizeof(mlp_fp8_tile) + 3 * sizeof(mlp_sc_tile));
static constexpr uint64_t FWD_MACRO0_MXFP8_DOWN_SCRATCH_OFFSET =
    (FWD_MACRO0_MXFP8_DOWN_RING_BYTES + 1023) & ~uint64_t(1023);
static constexpr uint64_t FWD_MACRO0_MXFP8_DOWN_ACTIVE_BYTES =
    FWD_MACRO0_MXFP8_DOWN_SCRATCH_OFFSET
    + config::MLP_NUM_BF16_D_TILES * sizeof(mlp_bf16_d_tile);
static_assert(FWD_MACRO0_MXFP8_DOWN_RING_BYTES == 102912);
static_assert(FWD_MACRO0_MXFP8_DOWN_SCRATCH_OFFSET == 103424);
static_assert(FWD_MACRO0_MXFP8_DOWN_ACTIVE_BYTES == 128000);
static_assert(
    FWD_MACRO0_MXFP8_DOWN_ACTIVE_BYTES + 1023
    <= config::DYNAMIC_SHARED_MEMORY);
static_assert(FWD_MACRO0_MXFP8_DOWN_ACTIVE_BYTES + 1023 <= 231424);

template <bool IS_CLAMPED>
static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_fwd_kernel(const globals_fwd &g) {
    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    // A fused task computes one MLP_Mb x SWIGLU_Nb hidden tile. CTA 0 supplies
    // Gate weights and CTA 1 supplies Up weights to the same cooperative MMA.
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::SWIGLU_Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / config::SWIGLU_Nb);
    const int shared_down_tasks = shared_row_blocks * (g.w_shared_down.rows() / config::MLP_Nb);
    const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (g.w_routed_down.rows() / config::MLP_Nb);
    const int shared_tasks = shared_gate_up_tasks + shared_down_tasks;
    const int minibatch_tasks = minibatch_routed_gate_up_tasks + minibatch_routed_down_tasks;
    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int macrobatch_size = g.macrobatch_size;

    const int num_tokens = g.num_tokens[{0}];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
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

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_finished[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
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
        auto num_dispatch_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int dispatch_col_blocks = (g.x_fp8_routed.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;
            return (macrobatch_tokens / config::DISPATCH_Mb) * dispatch_col_blocks;
        };
        auto num_combine_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int combine_col_blocks = (g.y_routed.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
            const int combine_tiles = (macrobatch_tokens / config::COMBINE_Mb) * combine_col_blocks;
            return (combine_tiles + config::COMBINE_PIPE_DEPTH - 1) / config::COMBINE_PIPE_DEPTH;
        };
        auto dispatch = [&](int macrobatch_idx, int task_idx) {
            dispatch_kernel<false>(g.x_routed_send_buffer, g.x_fp8_routed, &g.x_sc_routed, &g.x_fp8_t_routed, &g.x_sc_t_routed,
                                     nullptr, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                     macrobatch_idx + 1 < num_macrobatches ? &g.y_routed_ready : nullptr, nullptr, g.x_routed_ready,
                                     dispatch_inputs_arrived, dispatch_bitfield,
                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     macrobatch_idx + 1, 0, smem_base_addr);
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
            const int combine_tasks = num_combine_tasks(macrobatch_idx);
            const int dispatch_tasks = macrobatch_idx > 0 ? num_dispatch_tasks(macrobatch_idx - 1) : 0;
            for (int task_idx = comm_cta_idx; task_idx < max(combine_tasks, dispatch_tasks); task_idx += g.num_comm_sms) {
                if (task_idx < combine_tasks)
                    combine(macrobatch_idx, task_idx);
                if (task_idx < dispatch_tasks)
                    dispatch(macrobatch_idx - 1, task_idx);
            }
        }
        return;
    }

    // Fused Gate/Up/SwiGLU and Down are both cluster-cooperative tasks.
    auto is_cta_local_task = [&](int) { return false; };
    auto uses_compact_routed_mxfp8_smem = [&](int compute_cluster_idx) {
        if constexpr (!USE_MXFP8) {
            return false;
        } else {
            if (compute_cluster_idx < shared_tasks)
                return false;
            const int task_ordered_global_minibatch_idx =
                (compute_cluster_idx - shared_tasks) / minibatch_tasks;
            const int task_macrobatch_idx =
                task_ordered_global_minibatch_idx < last_macrobatch_num_minibatches
                    ? num_macrobatches - 1
                    : num_macrobatches - 2
                        - (task_ordered_global_minibatch_idx
                           - last_macrobatch_num_minibatches)
                            / minibatches_per_macrobatch;
            return task_macrobatch_idx == 0;
        }
    };
    const int hidden_row_block_ready_required_count = (config::MLP_Mb / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);

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
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);
        bool current_uses_compact_smem = false;

        if (compute_cluster_idx < shared_gate_up_tasks) {
            // Shared Gate + Up + SwiGLU (BF16). Shared preactivations are
            // retained because shared-expert backward does not replay them.
            const int task_idx = compute_cluster_idx;
            expert_gate_up_swiglu_kernel<
                true, IS_CLAMPED, FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH>(
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
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks;
            expert_grouped_gemm_kernel<
                true, false, false, FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH>(g.hidden_shared, g.w_shared_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.y_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, &g.hidden_row_block_ready, nullptr, nullptr, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else {
            // Routed expert with macro/minibatching
            const int task_ordered_global_minibatch_idx = (compute_cluster_idx - shared_tasks) / minibatch_tasks;
            const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) - task_ordered_global_minibatch_idx * minibatch_tasks;
            int macrobatch_idx, minibatch_idx;
            if (task_ordered_global_minibatch_idx < last_macrobatch_num_minibatches) {
                macrobatch_idx = num_macrobatches - 1;
                minibatch_idx = task_ordered_global_minibatch_idx;
            } else {
                const int idx = task_ordered_global_minibatch_idx - last_macrobatch_num_minibatches;
                macrobatch_idx = num_macrobatches - 2 - idx / minibatches_per_macrobatch;
                minibatch_idx = idx % minibatches_per_macrobatch;
            }
            current_uses_compact_smem = USE_MXFP8 && macrobatch_idx == 0;

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed Gate + Up + SwiGLU. Only macrobatch 0 retains
                // preactivations/transpose context; hidden normal is written
                // for every macrobatch so Down can consume it immediately.
                const int task_idx = minibatch_task_idx;
                auto run_routed_gate_up = [&](auto load_depth) {
                    constexpr int LOAD_PIPE_DEPTH = decltype(load_depth)::value;
                    expert_gate_up_swiglu_kernel<
                        false, IS_CLAMPED, LOAD_PIPE_DEPTH>(
                        g.x_fp8_routed, g.w_routed_gate, g.w_routed_up,
                        &g.x_sc_routed, &g.w_routed_gate_sc, &g.w_routed_up_sc,
                        g.gate_fp8_routed, &g.gate_sc_routed,
                        g.up_fp8_routed, &g.up_sc_routed,
                        g.hidden_fp8_routed, &g.hidden_sc_routed,
                        &g.hidden_fp8_t_routed, &g.hidden_sc_t_routed,
                        g.tokens_per_expert, &g.x_routed_ready,
                        g.hidden_row_block_ready,
                        d_tt, a_sc_tt, b_sc_tt,
                        gemm_inputs_arrived, gemm_scales_arrived,
                        gemm_inputs_finished, gemm_scales_finished,
                        gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                        num_tokens, g.swiglu_limit, macrobatch_size, g.minibatch_size,
                        macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                        shared_row_blocks, smem_base_addr);
                };
                if constexpr (USE_MXFP8) {
                    if (macrobatch_idx == 0)
                        run_routed_gate_up(std::integral_constant<
                            int, FUSED_GATE_UP_MACRO0_MXFP8_LOAD_PIPE_DEPTH>{});
                    else
                        run_routed_gate_up(std::integral_constant<
                            int, FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH>{});
                } else {
                    run_routed_gate_up(std::integral_constant<
                        int, FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH>{});
                }
            } else {
                // Routed down
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                auto run_routed_down = [&](auto load_depth) {
                    constexpr int LOAD_PIPE_DEPTH = decltype(load_depth)::value;
                    expert_grouped_gemm_kernel<
                        false, false, false, LOAD_PIPE_DEPTH>(g.hidden_fp8_routed, g.w_routed_down, &g.hidden_sc_routed, &g.w_routed_down_sc, nullptr, nullptr, nullptr, nullptr,
                                           g.y_routed, nullptr, nullptr,
                                           g.tokens_per_expert, nullptr, &g.hidden_row_block_ready,
                                           macrobatch_idx + 1 < num_macrobatches ? &g.y_routed_done : nullptr,
                                           nullptr, &g.y_routed_ready, nullptr,
                                           d_tt, a_sc_tt, b_sc_tt,
                                           gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                           0, shared_row_blocks, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
                };
                if constexpr (USE_MXFP8) {
                    if (macrobatch_idx == 0)
                        run_routed_down(std::integral_constant<
                            int, FUSED_GATE_UP_MACRO0_MXFP8_LOAD_PIPE_DEPTH>{});
                    else
                        run_routed_down(std::integral_constant<
                            int, FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH>{});
                } else {
                    run_routed_down(std::integral_constant<
                        int, FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH>{});
                }
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // The logical task order is last-macrobatch -> macro 0, so the normal
        // transition is depth 4 -> depth 3.  CLC itself does not promise an
        // index order, however.  If it ever returns a noncompact task after a
        // compact task, fence the whole cluster before depth-4 slot/layout
        // storage can overlap the prior depth-3 scratch.
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        const bool next_task_is_valid =
            cluster_idx >= 0 && cluster_idx < true_num_clusters;
        const bool needs_smem_layout_transition_fence =
            current_uses_compact_smem && next_task_is_valid
            && !uses_compact_routed_mxfp8_smem(next_compute_cluster_idx);
        if (next_task_is_valid
            && ((current_is_cta_local
                 && !is_cta_local_task(next_compute_cluster_idx))
                || needs_smem_layout_transition_fence))
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

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_mxfp8(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &w_routed_down_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_global_row_blocks = schedule_capacity / (config::MLP_Mb / config::CLUSTER_SIZE);
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.size(0) / config::SWIGLU_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (w_routed_gate.size(1) / config::SWIGLU_Nb);

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl y_routed_recv_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        y_routed_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_buffer_ptrs[i]);
    }

    at::Tensor x_fp8_routed = at::empty({macrobatch_size, hidden_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_routed = at::empty({macrobatch_size / 128, hidden_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor x_fp8_t_routed = at::empty({hidden_dim, macrobatch_size}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_t_routed = at::empty({hidden_dim / 128, macrobatch_size / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor gate_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor gate_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor gate_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor up_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor up_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor hidden_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor hidden_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor hidden_fp8_t_routed = at::empty({intermediate_dim, macrobatch_size}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_t_routed = at::empty({intermediate_dim / 128, macrobatch_size / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed = at::empty({macrobatch_size, hidden_dim}, x.options());
    at::Tensor x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor gate_up_tile_ready = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor y_routed_done = at::zeros({num_global_row_blocks}, tokens_per_expert.options());

    globals_fwd g {
        .x_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_routed),
        .x_sc_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_routed),
        .x_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_t_routed),
        .x_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_t_routed),
        .gate_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(gate_fp8_routed),
        .gate_sc_routed = kittens::py::tensor_to_gl<sc_gl>(gate_sc_routed),
        .up_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(up_fp8_routed),
        .up_sc_routed = kittens::py::tensor_to_gl<sc_gl>(up_sc_routed),
        .hidden_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_routed),
        .hidden_sc_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_routed),
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_t_routed),
        .hidden_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_t_routed),
        .y_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(y_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .y_routed_recv_buffer = y_routed_recv_buffer_data,
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_gate),
        .w_routed_gate_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_gate_sc),
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_up),
        .w_routed_up_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_up_sc),
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_down),
        .w_routed_down_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_down_sc),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .gate_up_tile_ready = kittens::py::tensor_to_gl<index_gl>(gate_up_tile_ready),
        .hidden_row_block_ready = kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .x_routed_ready = kittens::py::tensor_to_gl<index_gl>(x_routed_ready),
        .y_routed_ready = kittens::py::tensor_to_gl<index_gl>(y_routed_ready),
        .y_routed_done = kittens::py::tensor_to_gl<index_gl>(y_routed_done),
        .topk = topk,
        .swiglu_limit = swiglu_limit.value_or(0.0f),
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    if (swiglu_limit.has_value())
        kittens::py::launch_kernel<config, globals_fwd, dispatch_mlp_swiglu_combine_fwd_kernel<true>>(g);
    else
        kittens::py::launch_kernel<config, globals_fwd, dispatch_mlp_swiglu_combine_fwd_kernel<false>>(g);

    return {x_fp8_t_routed, x_sc_t_routed,
            gate_shared, gate_fp8_routed, gate_sc_routed,
            up_shared, up_fp8_routed, up_sc_routed,
            hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
            y_shared, y_routed};
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_bf16(
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
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(!USE_MXFP8);
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_global_row_blocks = schedule_capacity / (config::MLP_Mb / config::CLUSTER_SIZE);
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (intermediate_dim / config::SWIGLU_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (intermediate_dim / config::SWIGLU_Nb);

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl y_routed_recv_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        y_routed_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_buffer_ptrs[i]);
    }

    at::Tensor x_routed = at::empty({macrobatch_size, hidden_dim}, x.options());
    at::Tensor gate_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor hidden_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor hidden_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed = at::empty_like(x_routed);
    at::Tensor x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor gate_up_tile_ready = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor y_routed_done = at::zeros({num_global_row_blocks}, tokens_per_expert.options());

    globals_fwd g {
        .x_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(x_routed),
        .x_sc_routed = {},
        .x_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(x_routed),
        .x_sc_t_routed = {},
        .gate_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(gate_routed),
        .gate_sc_routed = {},
        .up_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(up_routed),
        .up_sc_routed = {},
        .hidden_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(hidden_routed),
        .hidden_sc_routed = {},
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(hidden_routed),
        .hidden_sc_t_routed = {},
        .y_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(y_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .y_routed_recv_buffer = y_routed_recv_buffer_data,
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_routed_gate_sc = {},
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_routed_up_sc = {},
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_down),
        .w_routed_down_sc = {},
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .gate_up_tile_ready = kittens::py::tensor_to_gl<index_gl>(gate_up_tile_ready),
        .hidden_row_block_ready = kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .x_routed_ready = kittens::py::tensor_to_gl<index_gl>(x_routed_ready),
        .y_routed_ready = kittens::py::tensor_to_gl<index_gl>(y_routed_ready),
        .y_routed_done = kittens::py::tensor_to_gl<index_gl>(y_routed_done),
        .topk = topk,
        .swiglu_limit = swiglu_limit.value_or(0.0f),
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    if (swiglu_limit.has_value())
        kittens::py::launch_kernel<config, globals_fwd, dispatch_mlp_swiglu_combine_fwd_kernel<true>>(g);
    else
        kittens::py::launch_kernel<config, globals_fwd, dispatch_mlp_swiglu_combine_fwd_kernel<false>>(g);
    return {x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed};
}
