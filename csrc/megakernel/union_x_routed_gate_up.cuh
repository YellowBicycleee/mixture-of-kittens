// Production-shape routed BF16 Gate+Up+SwiGLU for the parallel Union-X path.
//
// This header is called only by the parallel EP8 BF16 Union-X entrypoint.  It
// copies the routed BF16 task decode, tcgen05 mainloop, context/SwiGLU
// epilogue, and Hidden readiness semantics of the legacy tuned kernel.  Two
// adjacent logical N128 tiles for the same (expert, route-row block) are
// explicit one-task partners: every K64 stage gathers A once, loads both B
// partners, and updates two independent N256 Gate|Up accumulators.  The A
// operand is reconstructed with SM100 gather4: producer WG local warp 2 issues
// CTA-local A TMA, local warp 1 publishes its completion to CTA0, and local
// warp 3 retains the existing routed-weight TMA path.

// A stage now owns A + B0 + B1.  Two stages plus the unchanged three BF16
// epilogue tiles fit the launch SMEM envelope; three such stages do not.
static constexpr int UNION_X_RGU_STAGES = 2;
static constexpr int UNION_X_RGU_A_WAIT_WARP = 5;
static constexpr int UNION_X_RGU_A_TMA_WARP = 6;
static constexpr int UNION_X_RGU_B_WARP = 7;
static constexpr int UNION_X_RGU_MMA_WARP = 4;
static constexpr int UNION_X_RGU_CTA_ROWS = 128;
static constexpr int UNION_X_RGU_K = 64;
static constexpr int UNION_X_RGU_LOGICAL_N_TILES = 2;
static constexpr int UNION_X_RGU_PACKED_HIDDEN_N =
    UNION_X_RGU_LOGICAL_N_TILES * config::SWIGLU_Nb;

static_assert(UNION_X_RGU_STAGES <= config::MLP_LOAD_PIPE_DEPTH);
static_assert(UNION_X_RGU_CTA_ROWS == mlp_bf16_tile::rows);
static_assert(UNION_X_RGU_K == mlp_bf16_tile::cols);
static_assert(UNION_X_RGU_PACKED_HIDDEN_N == config::MLP_Nb);
static_assert(config::CLUSTER_SIZE == 2);
static_assert(config::NUM_THREADS == 256);

template <bool IS_CLAMPED>
static __device__ __forceinline__ void
expert_gate_up_swiglu_union_x_bf16_kernel(
    const routed_bf16_gl &union_x,
    const CUtensorMap &union_x_gather4_tma,
    const index_gl &route_to_union,
    const routed_weight_gl &gate_b_gmem,
    const routed_weight_gl &up_b_gmem,
    const routed_gate_up_gl &gate_context_gmem,
    const routed_gate_up_gl &up_context_gmem,
    const routed_activation_gl &hidden_gmem,
    const index_gl &tokens_per_expert,
    // Coarse readiness published by Union-X dispatch after every route-slice
    // task resolves.  A future caller must not pass the legacy X-ring counter.
    const index_gl &input_minibatch_ready,
    const index_gl &hidden_row_block_ready,
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> &d_tt,
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> &paired_d_tt,
    semaphore (&b_inputs_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&b_inputs_armed)[UNION_X_RGU_STAGES],
    semaphore (&a_inputs_reusable)[UNION_X_RGU_STAGES],
    semaphore (&a_tma_arrived)[UNION_X_RGU_STAGES],
    semaphore (&gemm_inputs_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&a_gather_arrived)[UNION_X_RGU_STAGES],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
    uint32_t &a_reuse_bitfield,
    uint32_t &a_tma_bitfield,
    uint32_t &a_gather_bitfield,
    uint32_t &b_arm_bitfield,
    const int num_tokens,
    const float swiglu_limit,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    int task_idx,
    const int cta_rank,
    const int hidden_row_block_ready_base_index,
    const uint64_t smem_base_addr
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);

    auto (&a_smem)[UNION_X_RGU_STAGES] = *reinterpret_cast<
        mlp_bf16_tile (*)[UNION_X_RGU_STAGES]>(smem_base_addr);
    auto (&b_smem)[UNION_X_RGU_STAGES] = *reinterpret_cast<
        mlp_bf16_tile (*)[UNION_X_RGU_STAGES]>(
            smem_base_addr + sizeof(a_smem));
    auto (&paired_b_smem)[UNION_X_RGU_STAGES] = *reinterpret_cast<
        mlp_bf16_tile (*)[UNION_X_RGU_STAGES]>(
            smem_base_addr + sizeof(a_smem) + sizeof(b_smem));
    // Producer warps can accept the next CLC task while this task's consumer
    // still uses its epilogue scratch.  Shared Gate/Up and routed Down both use
    // a four-stage generic A+B ring, so keep scratch above that 128 KiB region
    // even though this paired task's compact A+B0+B1 ring needs only 96 KiB.
    static constexpr uint64_t UNION_X_RGU_SCRATCH_OFFSET =
        2 * FUSED_GATE_UP_LOAD_PIPE_DEPTH * sizeof(mlp_bf16_tile);
    static_assert(
        sizeof(a_smem) + sizeof(b_smem) + sizeof(paired_b_smem)
        <= UNION_X_RGU_SCRATCH_OFFSET);
    const uint64_t scratch_base_addr =
        smem_base_addr + UNION_X_RGU_SCRATCH_OFFSET;
    auto &gate_raw_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr);
    auto &up_raw_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr + sizeof(quant_bf16_tile));
    auto &hidden_raw_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr + 2 * sizeof(quant_bf16_tile));
    static_assert(
        UNION_X_RGU_SCRATCH_OFFSET
            + 3 * sizeof(quant_bf16_tile) + 1023
        <= config::DYNAMIC_SHARED_MEMORY);

    const int paired_col_blocks =
        gate_b_gmem.rows() / UNION_X_RGU_PACKED_HIDDEN_N;
    const int global_minibatch_idx =
        macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset =
        macrobatch_idx * (macrobatch_size / config::MLP_Mb);

    // Preserve the existing grouped raw-task semantics, including G2
    // serialisation when the enclosing specialization requests it.
    const int task_group_idx = task_idx;
    #pragma unroll 1
    for (int task_group_offset = 0;
         task_group_offset < config::FUSED_GATE_UP_TASK_GROUP_SIZE;
         ++task_group_offset) {
    task_idx = task_group_idx * config::FUSED_GATE_UP_TASK_GROUP_SIZE
        + task_group_offset;

    int3 tile_coord = {-1, -1, -1};
    const int minibatch_routed_row_blocks =
        minibatch_size / config::MLP_Mb;
    const int global_minibatch_routed_first_row_block =
        global_minibatch_idx * minibatch_routed_row_blocks;
    int global_row_block_offset = 0;
    for (int expert_idx = 0; expert_idx < gate_b_gmem.depth();
         ++expert_idx) {
        const int expert_row_blocks =
            tokens_per_expert[{expert_idx}] / config::MLP_Mb;
        const int global_first_row_block = max(
            global_minibatch_routed_first_row_block,
            global_row_block_offset);
        const int row_blocks = max(
            0,
            min(global_minibatch_routed_first_row_block
                    + minibatch_routed_row_blocks,
                global_row_block_offset + expert_row_blocks)
                - global_first_row_block);
        const int num_tasks = row_blocks * paired_col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled =
                get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(
                    row_blocks, paired_col_blocks, task_idx);
            tile_coord = {
                global_first_row_block + swizzled.x
                    - macrobatch_row_block_offset,
                swizzled.y,
                expert_idx,
            };
            break;
        }
        task_idx -= num_tasks;
        global_row_block_offset += expert_row_blocks;
    }
    if (tile_coord.z < 0)
        return;

    const int iters_per_task = UNION_X_DISPATCH_HIDDEN / UNION_X_RGU_K;
    static_assert(UNION_X_DISPATCH_HIDDEN % UNION_X_RGU_K == 0);
    const int global_route_row_start =
        macrobatch_idx * macrobatch_size
        + tile_coord.x * config::MLP_Mb
        + cta_rank * UNION_X_RGU_CTA_ROWS;
    const int global_warp = threadIdx.x / WARP_THREADS;

    if (global_warp == UNION_X_RGU_A_WAIT_WARP
        && warp::elect_leader()) {
        int input_stage = 0;
        for (int k_block = 0; k_block < iters_per_task; ++k_block) {
            wait(
                a_tma_arrived[input_stage],
                get_phasebit<0>(a_tma_bitfield, input_stage));
            // The local complete-tx wait is acquire.  Publish one route-local
            // completion per CTA to the existing CTA0 cluster barrier only
            // after the full 16 KiB swizzled A tile is visible.
            warp::tma::cluster::arrive(
                a_gather_arrived[input_stage], 0);
            update_phasebit<0>(a_tma_bitfield, input_stage);
            input_stage = ring_advance<UNION_X_RGU_STAGES>(input_stage);
        }
    } else if (global_warp == UNION_X_RGU_A_TMA_WARP) {
        const int4 union_ids = union_x_load_route_ids4(
            route_to_union, global_route_row_start, union_x.rows());
        // The same route-row block is consumed by every FC1 N tile (eight for
        // Qwen I=1024), so retain its Union-X cache lines across nearby tasks.
        const uint64_t cache_hint =
            make_cache_policy<cache_policy::EVICT_LAST>();
        if (warp::elect_leader()) {
            const int minibatch_first_row =
                global_minibatch_idx * minibatch_size;
            const int minibatch_rows = max(
                0, min(minibatch_size, num_tokens - minibatch_first_row));
            const int required_count =
                ((minibatch_rows + config::DISPATCH_Mb - 1)
                    / config::DISPATCH_Mb)
                * ((UNION_X_DISPATCH_HIDDEN + config::DISPATCH_Nb - 1)
                    / config::DISPATCH_Nb);
            barrier_wait(
                input_minibatch_ready,
                global_minibatch_idx,
                required_count);
        }
        __syncwarp();
        // Dispatch publishes the persistent payload through an async TMA store
        // and exposes readiness through the generic proxy.  Every lane below
        // issues gather4, so every issuer bridges the acquire to async global.
        asm volatile("{fence.proxy.async.global;}" ::: "memory");

        int input_stage = 0;
        for (int k_block = 0; k_block < iters_per_task; ++k_block) {
            wait(
                a_inputs_reusable[input_stage],
                get_phasebit<0>(a_reuse_bitfield, input_stage));
            union_x_issue_gather4_a_tile(
                union_x_gather4_tma,
                a_tma_arrived[input_stage],
                a_smem[input_stage],
                k_block,
                union_ids,
                cache_hint);
            update_phasebit<0>(a_reuse_bitfield, input_stage);
            input_stage = ring_advance<UNION_X_RGU_STAGES>(input_stage);
        }
    } else if (
        global_warp == UNION_X_RGU_B_WARP
        && warp::elect_leader()) {
        int input_stage = 0;
        for (int k_block = 0; k_block < iters_per_task; ++k_block) {
            wait(
                gemm_inputs_finished[input_stage],
                get_phasebit<1>(gemm_bitfield, input_stage));
            // Warp 7 owns this producer phase across legacy and Union-X CLC
            // tasks.  Hand stage reuse to warp 6 through a private semaphore;
            // warp 6 must not infer the shared barrier phase from its stale
            // per-thread gemm bitfield.
            warp::tma::cluster::arrive(
                a_inputs_reusable[input_stage], cta_rank);
            // CTA0 must arm the shared transaction barrier before either
            // cluster CTA can complete bytes into it.
            wait(
                b_inputs_armed[input_stage],
                get_phasebit<0>(b_arm_bitfield, input_stage));
            update_phasebit<0>(b_arm_bitfield, input_stage);
            const routed_weight_gl &weight =
                cta_rank == 0 ? gate_b_gmem : up_b_gmem;
            tma::cluster::load_async(
                b_smem[input_stage],
                weight,
                {
                    tile_coord.z,
                    tile_coord.y * UNION_X_RGU_LOGICAL_N_TILES,
                    k_block,
                },
                b_inputs_arrived[input_stage],
                static_cast<uint16_t>(1 << cta_rank),
                0);
            tma::cluster::load_async(
                paired_b_smem[input_stage],
                weight,
                {
                    tile_coord.z,
                    tile_coord.y * UNION_X_RGU_LOGICAL_N_TILES + 1,
                    k_block,
                },
                b_inputs_arrived[input_stage],
                static_cast<uint16_t>(1 << cta_rank),
                0);
            update_phasebit<1>(gemm_bitfield, input_stage);
            input_stage = ring_advance<UNION_X_RGU_STAGES>(input_stage);
        }
    } else if (
        cta_rank == 0
        && global_warp == UNION_X_RGU_MMA_WARP
        && warp::elect_leader()) {
        wait(
            gemm_outputs_finished,
            get_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
        update_phasebit<1>(
            gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
        tensor_after_thread_sync();

        int input_stage = 0;
        for (int k_block = 0; k_block < iters_per_task; ++k_block) {
            tma::expect_bytes(
                b_inputs_arrived[input_stage],
                config::CLUSTER_SIZE
                    * UNION_X_RGU_LOGICAL_N_TILES
                    * sizeof(mlp_bf16_tile));
            // Publish the arm to the local B warp in both CTAs only after
            // expect_bytes has established this stage's transaction phase.
            warp::tma::cluster::arrive(
                b_inputs_armed[input_stage], 0);
            warp::tma::cluster::arrive(
                b_inputs_armed[input_stage], 1);
            tma::cluster::wait(
                a_gather_arrived[input_stage],
                get_phasebit<0>(a_gather_bitfield, input_stage));
            update_phasebit<0>(a_gather_bitfield, input_stage);
            wait(
                b_inputs_arrived[input_stage],
                get_phasebit<0>(gemm_bitfield, input_stage));
            if (k_block == 0) {
                mm2_ABt(
                    d_tt,
                    a_smem[input_stage],
                    b_smem[input_stage]);
                mm2_ABt(
                    paired_d_tt,
                    a_smem[input_stage],
                    paired_b_smem[input_stage],
                    gemm_inputs_finished[input_stage]);
            } else {
                mma2_ABt(
                    d_tt,
                    a_smem[input_stage],
                    b_smem[input_stage]);
                mma2_ABt(
                    paired_d_tt,
                    a_smem[input_stage],
                    paired_b_smem[input_stage],
                    gemm_inputs_finished[input_stage]);
            }
            update_phasebit<0>(gemm_bitfield, input_stage);
            input_stage = ring_advance<UNION_X_RGU_STAGES>(input_stage);
        }
        detail::tcgen05::commit<config::CLUSTER_SIZE>(
            gemm_outputs_arrived);
    } else if (warpgroup::groupid() == 0) {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(
            gemm_outputs_arrived,
            get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
        update_phasebit<0>(
            gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);

        const int tile_row = epilogue_group::laneid();
        const bool save_context = macrobatch_idx == 0;
        const int output_row =
            tile_coord.x * config::CLUSTER_SIZE + cta_rank;
        #pragma unroll 1
        for (int logical_n_offset = 0;
             logical_n_offset < UNION_X_RGU_LOGICAL_N_TILES;
             ++logical_n_offset) {
        auto &output_tt = logical_n_offset == 0 ? d_tt : paired_d_tt;
        const int logical_col =
            tile_coord.y * UNION_X_RGU_LOGICAL_N_TILES + logical_n_offset;
        #pragma unroll 1
        for (int block = 0; block < config::MLP_Nb / 32; ++block) {
            float2 tmp[16];
            asm volatile(R"(
                tcgen05.ld.sync.aligned.32x32b.x32.b32
                {%0, %1, %2, %3, %4, %5, %6, %7,
                 %8, %9, %10, %11, %12, %13, %14, %15,
                 %16, %17, %18, %19, %20, %21, %22, %23,
                 %24, %25, %26, %27, %28, %29, %30, %31}, [%32];
                )"
                : "=f"(tmp[0].x), "=f"(tmp[0].y),
                  "=f"(tmp[1].x), "=f"(tmp[1].y),
                  "=f"(tmp[2].x), "=f"(tmp[2].y),
                  "=f"(tmp[3].x), "=f"(tmp[3].y),
                  "=f"(tmp[4].x), "=f"(tmp[4].y),
                  "=f"(tmp[5].x), "=f"(tmp[5].y),
                  "=f"(tmp[6].x), "=f"(tmp[6].y),
                  "=f"(tmp[7].x), "=f"(tmp[7].y),
                  "=f"(tmp[8].x), "=f"(tmp[8].y),
                  "=f"(tmp[9].x), "=f"(tmp[9].y),
                  "=f"(tmp[10].x), "=f"(tmp[10].y),
                  "=f"(tmp[11].x), "=f"(tmp[11].y),
                  "=f"(tmp[12].x), "=f"(tmp[12].y),
                  "=f"(tmp[13].x), "=f"(tmp[13].y),
                  "=f"(tmp[14].x), "=f"(tmp[14].y),
                  "=f"(tmp[15].x), "=f"(tmp[15].y)
                : "r"(output_tt.addr
                      + ((warpgroup::warpid() * 32) << 16)
                      + block * 32));
            tensor_load_wait();

            bf16_2 d_reg[16];
            #pragma unroll
            for (int pair = 0; pair < 16; ++pair)
                d_reg[pair] = __float22bfloat162_rn(tmp[pair]);
            const uint32_t gate_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&gate_raw_smem));
            const uint32_t up_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&up_raw_smem));
            const uint32_t dst_addr = block < 4 ? gate_addr : up_addr;
            const int dst_col = (block & 3) * 32;
            const uint32_t *d_words =
                reinterpret_cast<const uint32_t *>(d_reg);
            #pragma unroll
            for (int vector = 0; vector < 4; ++vector) {
                move<float4>::sts(
                    quant_bf16_tile::idx(
                        dst_addr, {tile_row, dst_col + vector * 8}),
                    float4{
                        __uint_as_float(d_words[vector * 4]),
                        __uint_as_float(d_words[vector * 4 + 1]),
                        __uint_as_float(d_words[vector * 4 + 2]),
                        __uint_as_float(d_words[vector * 4 + 3]),
                    });
            }
        }
        tensor_before_thread_sync();
        epilogue_group::sync(1);
        if (logical_n_offset == UNION_X_RGU_LOGICAL_N_TILES - 1)
            warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);

        if (save_context) {
            if (epilogue_group::laneid() == 0) {
                tma::store_async(
                    gate_context_gmem,
                    gate_raw_smem,
                    {output_row, logical_col});
                tma::store_async(
                    up_context_gmem,
                    up_raw_smem,
                    {output_row, logical_col});
            }
            epilogue_group::sync(1);
        }

        const auto *gate_pairs =
            reinterpret_cast<const bf16_2 *>(gate_raw_smem.data);
        const auto *up_pairs =
            reinterpret_cast<const bf16_2 *>(up_raw_smem.data);
        auto *hidden_pairs =
            reinterpret_cast<bf16_2 *>(hidden_raw_smem.data);
        constexpr int NUM_PAIRS =
            config::SWIGLU_Mb * config::SWIGLU_Nb / 2;
        constexpr int EPILOGUE_THREADS =
            WARPGROUP_WARPS * WARP_THREADS;
        #pragma unroll 1
        for (int pair = epilogue_group::laneid(); pair < NUM_PAIRS;
             pair += EPILOGUE_THREADS) {
            float2 gate = __bfloat1622float2(gate_pairs[pair]);
            float2 up = __bfloat1622float2(up_pairs[pair]);
            if constexpr (IS_CLAMPED) {
                gate = {
                    fminf(gate.x, swiglu_limit),
                    fminf(gate.y, swiglu_limit),
                };
                up = {
                    fminf(fmaxf(up.x, -swiglu_limit), swiglu_limit),
                    fminf(fmaxf(up.y, -swiglu_limit), swiglu_limit),
                };
            }
            float2 denominator = base_ops::mul::op<float2>(
                gate, float2{-1.0f, -1.0f});
            denominator = base_ops::exp::op<float2>(denominator);
            denominator = base_ops::sum::op<float2>(
                denominator, float2{1.0f, 1.0f});
            gate = base_ops::div::op<float2>(gate, denominator);
            gate = base_ops::mul::op<float2>(gate, up);
            hidden_pairs[pair] =
                __floats2bfloat162_rn(gate.x, gate.y);
        }
        epilogue_group::sync(1);

        if (epilogue_group::laneid() == 0) {
            tma::store_async(
                hidden_gmem,
                hidden_raw_smem,
                {output_row, logical_col});
            tma::store_async_wait();
            // Preserve one publication for each old N128 hidden subtile.  A
            // paired task therefore contributes two arrivals per CTA, leaving
            // the existing Down required count byte-for-byte unchanged.
            barrier_arrive(
                hidden_row_block_ready,
                hidden_row_block_ready_base_index
                    + macrobatch_row_block_offset + tile_coord.x);
        }
        epilogue_group::sync(1);
        }
    }
    }
}
