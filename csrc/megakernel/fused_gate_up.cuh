// Forward-only grouped Gate+Up GEMM with an in-CTA SwiGLU epilogue.
//
// This file is included inside dispatch_mlp_swiglu_combiner, like the other
// megakernel implementation headers.  The cooperative 2-CTA MMA keeps the
// public Gate and Up weight tensors separate: CTA rank 0 loads a 128-row Gate
// weight tile and CTA rank 1 loads the matching 128-row Up weight tile.  The
// local TMEM accumulator is therefore [Gate(128) | Up(128)] for each CTA's
// 128 activation rows, and the epilogue can apply SwiGLU without an HBM
// Gate/Up round trip.

static constexpr int FUSED_GATE_UP_Nb = config::MLP_Nb / config::CLUSTER_SIZE;
static constexpr int FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH = 4;
static constexpr int FUSED_GATE_UP_MACRO0_MXFP8_LOAD_PIPE_DEPTH = 3;
static_assert(config::CLUSTER_SIZE == 2);
static_assert(FUSED_GATE_UP_Nb == config::SWIGLU_Nb);
static_assert(config::MLP_Nb == 2 * config::SWIGLU_Nb);
static_assert(FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH <= config::MLP_LOAD_PIPE_DEPTH);
static_assert(FUSED_GATE_UP_MACRO0_MXFP8_LOAD_PIPE_DEPTH <= config::MLP_LOAD_PIPE_DEPTH);
static_assert(
    config::MLP_LOAD_PIPE_DEPTH + 1
        + FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH
    <= 16);

// This topology is tuned and benchmarked for EP8.  It remains instantiable for
// the other supported world sizes so the generic functional matrix is kept,
// but no performance claim is implied outside EP8.
template <bool IS_SHARED, bool IS_CLAMPED, int LOAD_PIPE_DEPTH>
static __device__ __forceinline__ void expert_gate_up_swiglu_ep8_tuned_kernel(
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &a_gmem,
    const std::conditional_t<IS_SHARED, weight_bf16_gl, routed_weight_gl> &gate_b_gmem,
    const std::conditional_t<IS_SHARED, weight_bf16_gl, routed_weight_gl> &up_b_gmem,
    const routed_sc_gl *a_sc_gmem,                    // routed MXFP8 only
    const routed_sc_gl *gate_b_sc_gmem,               // routed MXFP8 only
    const routed_sc_gl *up_b_sc_gmem,                 // routed MXFP8 only
    const std::conditional_t<IS_SHARED, epi_bf16_gl, routed_gate_up_gl> &gate_context_gmem,
    const routed_sc_gl *gate_context_sc_gmem,         // routed MXFP8 only
    const std::conditional_t<IS_SHARED, epi_bf16_gl, routed_gate_up_gl> &up_context_gmem,
    const routed_sc_gl *up_context_sc_gmem,           // routed MXFP8 only
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &hidden_gmem,
    const routed_sc_gl *hidden_sc_gmem,               // routed MXFP8 only
    const routed_transposed_gl *hidden_t_gmem,         // routed MXFP8 only; saved macrobatch only
    const routed_sc_gl *hidden_sc_t_gmem,             // routed MXFP8 only; saved macrobatch only
    const index_gl &tokens_per_expert,
    const index_gl *input_minibatch_ready,             // dispatch -> Gate+Up
    const index_gl &hidden_row_block_ready,            // fused epilogue -> Down
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> &d_tt,
    const full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> &a_sc_tt,
    const full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> &b_sc_tt,
    semaphore (&gemm_inputs_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_scales_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_inputs_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_scales_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
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
    static constexpr bool USE_ROUTED_MXFP8 = !IS_SHARED && USE_MXFP8;
    static constexpr bool SAVE_ROUTED_MXFP8_CONTEXT =
        USE_ROUTED_MXFP8
        && LOAD_PIPE_DEPTH == FUSED_GATE_UP_MACRO0_MXFP8_LOAD_PIPE_DEPTH;
    static_assert(
        LOAD_PIPE_DEPTH == FUSED_GATE_UP_V3A_LOAD_PIPE_DEPTH
        || (USE_ROUTED_MXFP8
            && LOAD_PIPE_DEPTH == FUSED_GATE_UP_MACRO0_MXFP8_LOAD_PIPE_DEPTH));
    using a_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile, mlp_bf16_tile>;
    using b_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile, mlp_bf16_tile>;
    constexpr int MLP_Kb = USE_ROUTED_MXFP8 ? config::MLP_FP8_Kb : config::MLP_BF16_Kb;

    auto (&a_smem)[LOAD_PIPE_DEPTH] =
        *reinterpret_cast<a_tile (*)[LOAD_PIPE_DEPTH]>(smem_base_addr);
    auto (&b_smem)[LOAD_PIPE_DEPTH] =
        *reinterpret_cast<b_tile (*)[LOAD_PIPE_DEPTH]>(
            smem_base_addr + sizeof(a_smem));
    auto (&a_sc_smem)[LOAD_PIPE_DEPTH] =
        *reinterpret_cast<mlp_sc_tile (*)[LOAD_PIPE_DEPTH]>(
            smem_base_addr + sizeof(a_smem) + sizeof(b_smem));
    auto (&b_sc_smem)[LOAD_PIPE_DEPTH][2] =
        *reinterpret_cast<mlp_sc_tile (*)[LOAD_PIPE_DEPTH][2]>(
            smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem));

    // The active task's scratch is disjoint from its input ring.  Macro-0
    // routed MXFP8 uses the compact three-stage ring to make room for three
    // independent q/sc pairs: Gate context, Up context, and hidden normal.
    // All other paths retain V3a's four-stage ring and one-q footprint.
    static constexpr uint64_t FUSED_GATE_UP_RING_BYTES =
        sizeof(a_smem) + sizeof(b_smem)
        + sizeof(a_sc_smem) + sizeof(b_sc_smem);
    const uint64_t scratch_base_addr =
        smem_base_addr + FUSED_GATE_UP_RING_BYTES;

    static constexpr uint64_t GATE_RAW_OFFSET = 0;
    static constexpr uint64_t UP_RAW_OFFSET =
        GATE_RAW_OFFSET + sizeof(quant_bf16_tile);
    static constexpr uint64_t GATE_Q_FP8_OFFSET =
        UP_RAW_OFFSET + sizeof(quant_bf16_tile);
    static constexpr uint64_t GATE_Q_SC_OFFSET =
        GATE_Q_FP8_OFFSET + sizeof(quant_fp8_tile);
    static constexpr uint64_t UP_Q_FP8_OFFSET =
        GATE_Q_SC_OFFSET + sizeof(quant_sc_tile);
    static constexpr uint64_t UP_Q_SC_OFFSET =
        UP_Q_FP8_OFFSET + sizeof(quant_fp8_tile);
    static constexpr uint64_t HIDDEN_Q_FP8_OFFSET =
        UP_Q_SC_OFFSET + sizeof(quant_sc_tile);
    static constexpr uint64_t HIDDEN_Q_SC_OFFSET =
        HIDDEN_Q_FP8_OFFSET + sizeof(quant_fp8_tile);
    static constexpr uint64_t ROUTED_MXFP8_SCRATCH_BYTES =
        HIDDEN_Q_SC_OFFSET + sizeof(quant_sc_tile);
    static constexpr uint64_t V3A_SCRATCH_BYTES =
        GATE_Q_SC_OFFSET + sizeof(quant_sc_tile);
    static constexpr uint64_t ACTIVE_SCRATCH_BYTES =
        SAVE_ROUTED_MXFP8_CONTEXT
            ? ROUTED_MXFP8_SCRATCH_BYTES
            : V3A_SCRATCH_BYTES;

    auto &gate_raw_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr + GATE_RAW_OFFSET);
    auto &up_raw_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr + UP_RAW_OFFSET);
    auto &gate_q_fp8_smem = *reinterpret_cast<quant_fp8_tile *>(
        scratch_base_addr + GATE_Q_FP8_OFFSET);
    auto &gate_q_sc_smem = *reinterpret_cast<quant_sc_tile *>(
        scratch_base_addr + GATE_Q_SC_OFFSET);
    // Compile-time-dead extra-q aliases collapse onto Gate q in V3a paths, so
    // every constructed SMEM reference remains inside ACTIVE_SCRATCH_BYTES.
    auto &up_q_fp8_smem = *reinterpret_cast<quant_fp8_tile *>(
        scratch_base_addr
        + (SAVE_ROUTED_MXFP8_CONTEXT ? UP_Q_FP8_OFFSET : GATE_Q_FP8_OFFSET));
    auto &up_q_sc_smem = *reinterpret_cast<quant_sc_tile *>(
        scratch_base_addr
        + (SAVE_ROUTED_MXFP8_CONTEXT ? UP_Q_SC_OFFSET : GATE_Q_SC_OFFSET));
    auto &hidden_q_fp8_smem = *reinterpret_cast<quant_fp8_tile *>(
        scratch_base_addr
        + (SAVE_ROUTED_MXFP8_CONTEXT ? HIDDEN_Q_FP8_OFFSET : GATE_Q_FP8_OFFSET));
    auto &hidden_q_sc_smem = *reinterpret_cast<quant_sc_tile *>(
        scratch_base_addr
        + (SAVE_ROUTED_MXFP8_CONTEXT ? HIDDEN_Q_SC_OFFSET : GATE_Q_SC_OFFSET));

    // Keep every TMA shared-memory operand on a 128-byte boundary.  The kernel
    // base is 1024-byte aligned and the compact ring ends at +102912.
    static_assert(sizeof(a_smem) % 128 == 0);
    static_assert((sizeof(a_smem) + sizeof(b_smem)) % 128 == 0);
    static_assert(
        (sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem)) % 128 == 0);
    static_assert(FUSED_GATE_UP_RING_BYTES % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + GATE_RAW_OFFSET) % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + UP_RAW_OFFSET) % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + GATE_Q_FP8_OFFSET) % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + GATE_Q_SC_OFFSET) % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + UP_Q_FP8_OFFSET) % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + UP_Q_SC_OFFSET) % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + HIDDEN_Q_FP8_OFFSET) % 128 == 0);
    static_assert((FUSED_GATE_UP_RING_BYTES + HIDDEN_Q_SC_OFFSET) % 128 == 0);
    static_assert(V3A_SCRATCH_BYTES == 82432);
    static_assert(!SAVE_ROUTED_MXFP8_CONTEXT || FUSED_GATE_UP_RING_BYTES == 102912);
    static_assert(!SAVE_ROUTED_MXFP8_CONTEXT || ROUTED_MXFP8_SCRATCH_BYTES == 116224);
    static_assert(
        !SAVE_ROUTED_MXFP8_CONTEXT
        || FUSED_GATE_UP_RING_BYTES + ROUTED_MXFP8_SCRATCH_BYTES == 219136);
    static_assert(
        SAVE_ROUTED_MXFP8_CONTEXT || FUSED_GATE_UP_RING_BYTES == 137216);
    static_assert(
        SAVE_ROUTED_MXFP8_CONTEXT
        || FUSED_GATE_UP_RING_BYTES + V3A_SCRATCH_BYTES == 219648);
    static_assert(
        FUSED_GATE_UP_RING_BYTES + ACTIVE_SCRATCH_BYTES + 1023
        <= config::DYNAMIC_SHARED_MEMORY);
    static_assert(
        FUSED_GATE_UP_RING_BYTES + ACTIVE_SCRATCH_BYTES + 1023
        <= 231424);

    const int col_blocks = gate_b_gmem.rows() / FUSED_GATE_UP_Nb;
    const int global_minibatch_idx =
        macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::MLP_Mb);

    // Each logical task produces 256 token rows x 128 intermediate columns.
    int3 tile_coord = {-1, -1, -1};
    if constexpr (IS_SHARED) {
        const int row_blocks = a_gmem.rows() / config::MLP_Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled =
                get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {swizzled.x, swizzled.y, 0};
        }
    } else {
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int global_minibatch_routed_first_row_block =
            global_minibatch_idx * minibatch_routed_row_blocks;
        int global_row_block_offset = 0;
        for (int expert_idx = 0; expert_idx < gate_b_gmem.depth(); ++expert_idx) {
            const int expert_row_blocks = tokens_per_expert[{expert_idx}] / config::MLP_Mb;
            const int global_first_row_block =
                max(global_minibatch_routed_first_row_block, global_row_block_offset);
            const int row_blocks = max(
                0,
                min(global_minibatch_routed_first_row_block + minibatch_routed_row_blocks,
                    global_row_block_offset + expert_row_blocks) - global_first_row_block);
            const int num_tasks = row_blocks * col_blocks;
            if (task_idx < num_tasks) {
                const int2 swizzled =
                    get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
                tile_coord = {
                    global_first_row_block + swizzled.x - macrobatch_row_block_offset,
                    swizzled.y,
                    expert_idx,
                };
                break;
            }
            task_idx -= num_tasks;
            global_row_block_offset += expert_row_blocks;
        }
    }
    if (tile_coord.z < 0)
        return;

    const int iters_per_task = a_gmem.cols() / MLP_Kb;
    auto wait_for_a_operand = [&]() {
        if (input_minibatch_ready != nullptr) {
            const int minibatch_first_row = global_minibatch_idx * minibatch_size;
            const int minibatch_rows = max(0, min(minibatch_size, num_tokens - minibatch_first_row));
            const int required_count =
                ((minibatch_rows + config::DISPATCH_Mb - 1) / config::DISPATCH_Mb)
                * ((a_gmem.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb);
            barrier_wait(*input_minibatch_ready, global_minibatch_idx, required_count);
        }
    };

    if (warpgroup::groupid() == config::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            wait_for_a_operand();
            int input_ring = 0;
            for (int k_block = 0; k_block < iters_per_task; ++k_block) {
                wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                tma::cluster::load_async(
                    a_smem[input_ring], a_gmem,
                    {tile_coord.x * 2 + cta_rank, k_block},
                    gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);

                // Rank 0 contributes Gate columns and rank 1 contributes the
                // matching Up columns.  mm2_ABt exposes both rank-contributed
                // N halves in each CTA's local 128x256 TMEM accumulator.
                if (cta_rank == 0) {
                    tma::cluster::load_async(
                        b_smem[input_ring], gate_b_gmem,
                        {tile_coord.z, tile_coord.y, k_block},
                        gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                } else {
                    tma::cluster::load_async(
                        b_smem[input_ring], up_b_gmem,
                        {tile_coord.z, tile_coord.y, k_block},
                        gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                }
                update_phasebit<1>(gemm_bitfield, input_ring);
                input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            if constexpr (USE_ROUTED_MXFP8) {
                wait_for_a_operand();
                int input_ring = 0;
                for (int k_block = 0; k_block < iters_per_task; ++k_block) {
                    wait(gemm_scales_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                    tma::cluster::load_async(
                        a_sc_smem[input_ring], *a_sc_gmem,
                        {tile_coord.x * 2 + cta_rank, k_block, 0, 0},
                        gemm_scales_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    if (cta_rank == 0) {
                        tma::cluster::load_async(
                            b_sc_smem[input_ring][cta_rank], *gate_b_sc_gmem,
                            {tile_coord.z * (gate_b_gmem.rows() / config::QUANT_Mb) + tile_coord.y,
                             k_block, 0, 0},
                            gemm_scales_arrived[input_ring], (uint16_t)(0b11), 0);
                    } else {
                        tma::cluster::load_async(
                            b_sc_smem[input_ring][cta_rank], *up_b_sc_gmem,
                            {tile_coord.z * (up_b_gmem.rows() / config::QUANT_Mb) + tile_coord.y,
                             k_block, 0, 0},
                            gemm_scales_arrived[input_ring], (uint16_t)(0b11), 0);
                    }
                    update_phasebit<1>(gemm_bitfield, input_ring);
                    input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
                }
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            int input_ring = 0;
            wait(gemm_outputs_finished,
                 get_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
            update_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
            tensor_after_thread_sync();
            for (int idx = 0; idx < iters_per_task; ++idx) {
                if constexpr (USE_ROUTED_MXFP8) {
                    tma::expect_bytes(
                        gemm_scales_arrived[input_ring],
                        config::CLUSTER_SIZE * 3 * sizeof(mlp_sc_tile));
                    wait(gemm_scales_arrived[input_ring],
                         get_phasebit<0>(gemm_bitfield,
                                         config::MLP_LOAD_PIPE_DEPTH + 1 + input_ring));
                    update_phasebit<0>(gemm_bitfield,
                                       config::MLP_LOAD_PIPE_DEPTH + 1 + input_ring);
                    auto a_sc_tt_subtile =
                        a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16);
                    auto b_sc_tt_subtile_0 =
                        b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32);
                    auto b_sc_tt_subtile_1 =
                        b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32 + 16);
                    load_mxnv_scale_async2(a_sc_tt_subtile, a_sc_smem[input_ring]);
                    load_mxnv_scale_async2(b_sc_tt_subtile_0, b_sc_smem[input_ring][0]);
                    load_mxnv_scale_async2(
                        b_sc_tt_subtile_1, b_sc_smem[input_ring][1],
                        gemm_scales_finished[input_ring]);
                    tma::expect_bytes(
                        gemm_inputs_arrived[input_ring],
                        config::CLUSTER_SIZE * (sizeof(a_tile) + sizeof(b_tile)));
                    wait(gemm_inputs_arrived[input_ring],
                         get_phasebit<0>(gemm_bitfield, input_ring));
                    if (idx == 0) {
                        mm2_ABt(
                            d_tt, a_smem[input_ring], b_smem[input_ring],
                            a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                            b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                            gemm_inputs_finished[input_ring]);
                    } else {
                        mma2_ABt(
                            d_tt, a_smem[input_ring], b_smem[input_ring],
                            a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                            b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                            gemm_inputs_finished[input_ring]);
                    }
                } else {
                    tma::expect_bytes(
                        gemm_inputs_arrived[input_ring],
                        config::CLUSTER_SIZE * (sizeof(a_tile) + sizeof(b_tile)));
                    wait(gemm_inputs_arrived[input_ring],
                         get_phasebit<0>(gemm_bitfield, input_ring));
                    if (idx == 0)
                        mm2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring],
                                gemm_inputs_finished[input_ring]);
                    else
                        mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring],
                                 gemm_inputs_finished[input_ring]);
                }
                update_phasebit<0>(gemm_bitfield, input_ring);
                input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<config::CLUSTER_SIZE>(gemm_outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(gemm_outputs_arrived,
             get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
        update_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);

        // Materialize the accumulator once in BF16 SMEM.  This matches the
        // old HBM-separated path's BF16 rounding point before SwiGLU.
        const int tile_row = epilogue_group::laneid();
        #pragma unroll 1
        for (int block = 0; block < config::MLP_Nb / 32; ++block) {
            float2 tmp[16];
            asm volatile(R"(
                tcgen05.ld.sync.aligned.32x32b.x32.b32
                {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,
                 %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];
                )"
                : "=f"(tmp[0].x), "=f"(tmp[0].y), "=f"(tmp[1].x), "=f"(tmp[1].y),
                  "=f"(tmp[2].x), "=f"(tmp[2].y), "=f"(tmp[3].x), "=f"(tmp[3].y),
                  "=f"(tmp[4].x), "=f"(tmp[4].y), "=f"(tmp[5].x), "=f"(tmp[5].y),
                  "=f"(tmp[6].x), "=f"(tmp[6].y), "=f"(tmp[7].x), "=f"(tmp[7].y),
                  "=f"(tmp[8].x), "=f"(tmp[8].y), "=f"(tmp[9].x), "=f"(tmp[9].y),
                  "=f"(tmp[10].x), "=f"(tmp[10].y), "=f"(tmp[11].x), "=f"(tmp[11].y),
                  "=f"(tmp[12].x), "=f"(tmp[12].y), "=f"(tmp[13].x), "=f"(tmp[13].y),
                  "=f"(tmp[14].x), "=f"(tmp[14].y), "=f"(tmp[15].x), "=f"(tmp[15].y)
                : "r"(d_tt.addr + ((warpgroup::warpid() * 32) << 16) + block * 32));
            tensor_load_wait();

            bf16_2 d_reg[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                d_reg[j] = __float22bfloat162_rn(tmp[j]);
            const uint32_t gate_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&gate_raw_smem));
            const uint32_t up_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&up_raw_smem));
            const uint32_t dst_addr = block < 4 ? gate_addr : up_addr;
            const int dst_col = (block & 3) * 32;
            const uint32_t *d_words = reinterpret_cast<const uint32_t *>(d_reg);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                move<float4>::sts(
                    quant_bf16_tile::idx(dst_addr, {tile_row, dst_col + j * 8}),
                    float4{
                        __uint_as_float(d_words[j * 4]),
                        __uint_as_float(d_words[j * 4 + 1]),
                        __uint_as_float(d_words[j * 4 + 2]),
                        __uint_as_float(d_words[j * 4 + 3]),
                    });
            }
        }
        tensor_before_thread_sync();
        epilogue_group::sync(1);
        warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);

        const bool save_context = IS_SHARED || macrobatch_idx == 0;
        const int output_row = tile_coord.x * config::CLUSTER_SIZE + cta_rank;

        // Routed MXFP8 macro 0 owns independent Gate/Up q sources.  Issue
        // both context streams before SwiGLU so their TMA source reads can
        // overlap the activation.  The depth-4 routed instantiation is the
        // macro>0 fast path and compiles this entire context block away.
        if constexpr (USE_ROUTED_MXFP8) {
            if constexpr (SAVE_ROUTED_MXFP8_CONTEXT) {
                mxfp8::quantize_tile<
                    true, false, config::SWIGLU_Nb, false, false, false>(
                    gate_raw_smem,
                    gate_q_fp8_smem, gate_q_sc_smem,
                    gate_q_fp8_smem, gate_q_sc_smem, nullptr,
                    epilogue_group::laneid(), 1);
                epilogue_group::sync(1);
                if (epilogue_group::laneid() == 0) {
                    tma::store_async(gate_context_gmem, gate_q_fp8_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(*gate_context_sc_gmem, gate_q_sc_smem,
                                     {output_row, tile_coord.y, 0, 0});
                }

                mxfp8::quantize_tile<
                    true, false, config::SWIGLU_Nb, false, false, false>(
                    up_raw_smem,
                    up_q_fp8_smem, up_q_sc_smem,
                    up_q_fp8_smem, up_q_sc_smem, nullptr,
                    epilogue_group::laneid(), 1);
                epilogue_group::sync(1);
                if (epilogue_group::laneid() == 0) {
                    tma::store_async(up_context_gmem, up_q_fp8_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(*up_context_sc_gmem, up_q_sc_smem,
                                     {output_row, tile_coord.y, 0, 0});
                }
            }
        } else {
            // Shared Gate/Up are always saved; routed BF16 saves macro 0.  The
            // read wait makes Gate safe for the in-place hidden overwrite.
            if (save_context) {
                if (epilogue_group::laneid() == 0) {
                    tma::store_async(gate_context_gmem, gate_raw_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(up_context_gmem, up_raw_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async_read_wait();
                }
                epilogue_group::sync(1);
            }
        }

        // Apply SwiGLU in place: each thread reads one Gate/Up pair and
        // overwrites exactly the corresponding Gate pair with hidden.
        const auto *gate_pairs =
            reinterpret_cast<const bf16_2 *>(gate_raw_smem.data);
        const auto *up_pairs =
            reinterpret_cast<const bf16_2 *>(up_raw_smem.data);
        auto *hidden_pairs = reinterpret_cast<bf16_2 *>(gate_raw_smem.data);
        constexpr int NUM_PAIRS = config::SWIGLU_Mb * config::SWIGLU_Nb / 2;
        constexpr int EPILOGUE_THREADS = WARPGROUP_WARPS * WARP_THREADS;
        #pragma unroll 1
        for (int i = epilogue_group::laneid(); i < NUM_PAIRS;
             i += EPILOGUE_THREADS) {
            float2 gate = __bfloat1622float2(gate_pairs[i]);
            float2 up = __bfloat1622float2(up_pairs[i]);
            if constexpr (IS_CLAMPED) {
                gate = {fminf(gate.x, swiglu_limit), fminf(gate.y, swiglu_limit)};
                up = {
                    fminf(fmaxf(up.x, -swiglu_limit), swiglu_limit),
                    fminf(fmaxf(up.y, -swiglu_limit), swiglu_limit),
                };
            }
            float2 denominator = base_ops::mul::op<float2>(gate, float2{-1.0f, -1.0f});
            denominator = base_ops::exp::op<float2>(denominator);
            denominator = base_ops::sum::op<float2>(denominator, float2{1.0f, 1.0f});
            gate = base_ops::div::op<float2>(gate, denominator);
            gate = base_ops::mul::op<float2>(gate, up);
            hidden_pairs[i] = __floats2bfloat162_rn(gate.x, gate.y);
        }
        // All hidden BF16 elements must be visible before quantization.  On
        // macro 0, both context streams continue reading their independent q
        // sources across this synchronization and the hidden-normal pass.
        epilogue_group::sync(1);

        if constexpr (USE_ROUTED_MXFP8) {
            if constexpr (SAVE_ROUTED_MXFP8_CONTEXT) {
                // The third q pair is disjoint from both context sources, so
                // hidden-normal quantization overlaps both TMA source reads.
                mxfp8::quantize_tile<
                    true, false, config::SWIGLU_Nb, false, false, false>(
                    gate_raw_smem,
                    hidden_q_fp8_smem, hidden_q_sc_smem,
                    hidden_q_fp8_smem, hidden_q_sc_smem, nullptr,
                    epilogue_group::laneid(), 1);

                // Only the elected leader issued the four context stores.  Its
                // read wait closes both source lifetimes; the warpgroup sync
                // then makes Gate q reusable by every epilogue thread.
                if (epilogue_group::laneid() == 0)
                    tma::store_async_read_wait();
                epilogue_group::sync(1);

                mxfp8::quantize_tile<
                    false, true, config::SWIGLU_Nb, false, false, false>(
                    gate_raw_smem,
                    gate_q_fp8_smem, gate_q_sc_smem,
                    gate_q_fp8_smem, gate_q_sc_smem, nullptr,
                    epilogue_group::laneid(), 1);
                epilogue_group::sync(1);

                if (epilogue_group::laneid() == 0) {
                    tma::store_async(hidden_gmem, hidden_q_fp8_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(*hidden_sc_gmem, hidden_q_sc_smem,
                                     {output_row, tile_coord.y, 0, 0});
                    tma::store_async(*hidden_t_gmem, gate_q_fp8_smem,
                                     {tile_coord.y, output_row});
                    tma::store_async(*hidden_sc_t_gmem, gate_q_sc_smem,
                                     {tile_coord.y, output_row, 0, 0});
                    // Full completion covers context and hidden writes before
                    // Down observes this row-block's ready counter.
                    tma::store_async_wait();
                    barrier_arrive(
                        hidden_row_block_ready,
                        hidden_row_block_ready_base_index
                            + macrobatch_row_block_offset + tile_coord.x);
                }
            } else {
                // Routed MXFP8 macro>0: V3a's one-q hidden-normal fast path.
                mxfp8::quantize_tile<
                    true, false, config::SWIGLU_Nb, false, false, false>(
                    gate_raw_smem,
                    gate_q_fp8_smem, gate_q_sc_smem,
                    gate_q_fp8_smem, gate_q_sc_smem, nullptr,
                    epilogue_group::laneid(), 1);
                epilogue_group::sync(1);
                if (epilogue_group::laneid() == 0) {
                    tma::store_async(hidden_gmem, gate_q_fp8_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(*hidden_sc_gmem, gate_q_sc_smem,
                                     {output_row, tile_coord.y, 0, 0});
                    tma::store_async_wait();
                    barrier_arrive(
                        hidden_row_block_ready,
                        hidden_row_block_ready_base_index
                            + macrobatch_row_block_offset + tile_coord.x);
                }
            }
        } else {
            if (epilogue_group::laneid() == 0) {
                tma::store_async(hidden_gmem, gate_raw_smem,
                                 {output_row, tile_coord.y});
                tma::store_async_wait();
                barrier_arrive(
                    hidden_row_block_ready,
                    hidden_row_block_ready_base_index
                        + macrobatch_row_block_offset + tile_coord.x);
            }
        }
        // The leader's TMA wait protects the scratch lifetime.  Collect the
        // whole consumer warpgroup before any thread accepts the next task.
        epilogue_group::sync(1);
    }
}
