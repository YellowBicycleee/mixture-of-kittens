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
static constexpr int FUSED_GATE_UP_LOAD_PIPE_DEPTH = 4;
static_assert(config::CLUSTER_SIZE == 2);
static_assert(FUSED_GATE_UP_Nb == config::SWIGLU_Nb);
static_assert(config::MLP_Nb == 2 * config::SWIGLU_Nb);
static_assert(FUSED_GATE_UP_LOAD_PIPE_DEPTH <= config::MLP_LOAD_PIPE_DEPTH);
static_assert(config::FUSED_GATE_UP_TASK_GROUP_SIZE > 0);

template <bool IS_SHARED, bool IS_CLAMPED>
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
    static constexpr bool PURE_BF16 = !USE_MXFP8;
    using a_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile, mlp_bf16_tile>;
    using b_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile, mlp_bf16_tile>;
    constexpr int MLP_Kb = USE_ROUTED_MXFP8 ? config::MLP_FP8_Kb : config::MLP_BF16_Kb;

    auto (&a_smem)[FUSED_GATE_UP_LOAD_PIPE_DEPTH] =
        *reinterpret_cast<a_tile (*)[FUSED_GATE_UP_LOAD_PIPE_DEPTH]>(smem_base_addr);
    auto (&b_smem)[FUSED_GATE_UP_LOAD_PIPE_DEPTH] =
        *reinterpret_cast<b_tile (*)[FUSED_GATE_UP_LOAD_PIPE_DEPTH]>(
            smem_base_addr + sizeof(a_smem));
    auto (&a_sc_smem)[FUSED_GATE_UP_LOAD_PIPE_DEPTH] =
        *reinterpret_cast<mlp_sc_tile (*)[FUSED_GATE_UP_LOAD_PIPE_DEPTH]>(
            smem_base_addr + sizeof(a_smem) + sizeof(b_smem));
    auto (&b_sc_smem)[FUSED_GATE_UP_LOAD_PIPE_DEPTH][2] =
        *reinterpret_cast<mlp_sc_tile (*)[FUSED_GATE_UP_LOAD_PIPE_DEPTH][2]>(
            smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem));

    // Scratch is disjoint from the four-stage input ring.  Routed MXFP8 keeps
    // one BF16 hidden tile plus independent Gate, Up, and hidden-normal q
    // buffers; shared/routed-BF16 retain the V2.1 Gate/Up raw layout.
    // A pure-BF16 instantiation never touches the scale ring. Reclaim it for
    // a disjoint Hidden tile so the Gate/Up context stores can read their raw
    // tiles while the consumer warpgroup executes SwiGLU. Keep the scale-ring
    // layout intact for an MXFP8 instantiation: its shared-BF16 task can overlap
    // a following routed-MXFP8 producer that already writes these scale stages.
    static constexpr uint64_t FUSED_GATE_UP_RING_BYTES =
        sizeof(a_smem) + sizeof(b_smem)
        + (PURE_BF16 ? 0 : sizeof(a_sc_smem) + sizeof(b_sc_smem));
    static_assert(FUSED_GATE_UP_RING_BYTES % 1024 == 0);
    const uint64_t scratch_base_addr =
        smem_base_addr + FUSED_GATE_UP_RING_BYTES;

    static constexpr uint64_t ROUTED_HIDDEN_BF16_OFFSET = 0;
    static constexpr uint64_t ROUTED_GATE_Q_FP8_OFFSET =
        ROUTED_HIDDEN_BF16_OFFSET + sizeof(quant_bf16_tile);
    static constexpr uint64_t ROUTED_UP_Q_FP8_OFFSET =
        ROUTED_GATE_Q_FP8_OFFSET + sizeof(quant_fp8_tile);
    static constexpr uint64_t ROUTED_HIDDEN_Q_FP8_OFFSET =
        ROUTED_UP_Q_FP8_OFFSET + sizeof(quant_fp8_tile);
    static constexpr uint64_t ROUTED_GATE_Q_SC_OFFSET =
        ROUTED_HIDDEN_Q_FP8_OFFSET + sizeof(quant_fp8_tile);
    static constexpr uint64_t ROUTED_UP_Q_SC_OFFSET =
        ROUTED_GATE_Q_SC_OFFSET + sizeof(quant_sc_tile);
    static constexpr uint64_t ROUTED_HIDDEN_Q_SC_OFFSET =
        ROUTED_UP_Q_SC_OFFSET + sizeof(quant_sc_tile);
    static constexpr uint64_t ROUTED_SCRATCH_BYTES =
        ROUTED_HIDDEN_Q_SC_OFFSET + sizeof(quant_sc_tile);
    static constexpr uint64_t BF16_SCRATCH_BYTES =
        PURE_BF16
            ? 3 * sizeof(quant_bf16_tile)
            : 2 * sizeof(quant_bf16_tile)
                + sizeof(quant_fp8_tile) + sizeof(quant_sc_tile);
    static constexpr uint64_t ACTIVE_SCRATCH_BYTES =
        USE_ROUTED_MXFP8 ? ROUTED_SCRATCH_BYTES : BF16_SCRATCH_BYTES;

    auto &gate_raw_smem = *reinterpret_cast<quant_bf16_tile *>(scratch_base_addr);
    auto &up_raw_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr + sizeof(gate_raw_smem));
    auto &hidden_raw_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr + 2 * sizeof(quant_bf16_tile));
    auto &hidden_bf16_smem = *reinterpret_cast<quant_bf16_tile *>(
        scratch_base_addr + ROUTED_HIDDEN_BF16_OFFSET);
    auto &gate_q_fp8_smem = *reinterpret_cast<quant_fp8_tile *>(
        scratch_base_addr + ROUTED_GATE_Q_FP8_OFFSET);
    auto &gate_q_sc_smem = *reinterpret_cast<quant_sc_tile *>(
        scratch_base_addr + ROUTED_GATE_Q_SC_OFFSET);
    auto &up_q_fp8_smem = *reinterpret_cast<quant_fp8_tile *>(
        scratch_base_addr + ROUTED_UP_Q_FP8_OFFSET);
    auto &up_q_sc_smem = *reinterpret_cast<quant_sc_tile *>(
        scratch_base_addr + ROUTED_UP_Q_SC_OFFSET);
    auto &hidden_q_fp8_smem = *reinterpret_cast<quant_fp8_tile *>(
        scratch_base_addr + ROUTED_HIDDEN_Q_FP8_OFFSET);
    auto &hidden_q_sc_smem = *reinterpret_cast<quant_sc_tile *>(
        scratch_base_addr + ROUTED_HIDDEN_Q_SC_OFFSET);
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

    // A CLC task owns a short consecutive group of raw 256-row x 128-column
    // Gate+Up tiles. Keep every raw tile as a complete producer/epilogue
    // transaction so Down's existing per-row ready count remains unchanged.
    const int task_group_idx = task_idx;
    #pragma unroll 1
    for (int task_group_offset = 0;
         task_group_offset < config::FUSED_GATE_UP_TASK_GROUP_SIZE;
         ++task_group_offset) {
    task_idx = task_group_idx * config::FUSED_GATE_UP_TASK_GROUP_SIZE
        + task_group_offset;

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
    // Both CTAs derive the same tail predicate. Returning here is therefore
    // cluster-uniform, after all earlier raw tiles in this group completed.
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
                input_ring = ring_advance<FUSED_GATE_UP_LOAD_PIPE_DEPTH>(input_ring);
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
                    input_ring = ring_advance<FUSED_GATE_UP_LOAD_PIPE_DEPTH>(input_ring);
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
                input_ring = ring_advance<FUSED_GATE_UP_LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<config::CLUSTER_SIZE>(gemm_outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(gemm_outputs_arrived,
             get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
        update_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);

        const int tile_row = epilogue_group::laneid();
        const bool save_context = IS_SHARED || macrobatch_idx == 0;
        const int output_row = tile_coord.x * config::CLUSTER_SIZE + cta_rank;
        if constexpr (USE_ROUTED_MXFP8) {
            // Load Gate/Up as four 32-column pairs.  Gate is retained in the
            // hidden tile; each Up block stays in registers for its SwiGLU.
            // Macro 0 rereads Up after issuing context stores, while later
            // macrobatches remain single-pass.  Both preserve the original
            // F32-to-BF16 rounding point without two full BF16 raw tiles.
            const uint32_t hidden_bf16_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&hidden_bf16_smem));
            const uint32_t gate_q_fp8_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&gate_q_fp8_smem));
            const uint32_t gate_q_sc_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&gate_q_sc_smem));
            const uint32_t up_q_fp8_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&up_q_fp8_smem));
            const uint32_t up_q_sc_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&up_q_sc_smem));
            const uint32_t hidden_q_fp8_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&hidden_q_fp8_smem));
            const uint32_t hidden_q_sc_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&hidden_q_sc_smem));

            auto load_tmem_bf16_block = [&](const int block, bf16_2 (&values)[16]) {
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
                    : "r"(d_tt.addr + ((warpgroup::warpid() * 32) << 16)
                          + block * 32));
                tensor_load_wait();
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    values[j] = __float22bfloat162_rn(tmp[j]);
            };

            auto store_hidden_block = [&](const bf16_2 (&values)[16], const int block) {
                const uint32_t *words = reinterpret_cast<const uint32_t *>(values);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    move<float4>::sts(
                        quant_bf16_tile::idx(
                            hidden_bf16_addr, {tile_row, block * 32 + j * 8}),
                        float4{
                            __uint_as_float(words[j * 4]),
                            __uint_as_float(words[j * 4 + 1]),
                            __uint_as_float(words[j * 4 + 2]),
                            __uint_as_float(words[j * 4 + 3]),
                        });
                }
            };

            auto quantize_row_block = [&] (
                const bf16_2 (&values)[16], const uint32_t fp8_addr,
                const int block, uint32_t &scale_word
            ) {
                uint32_t values_fp8[8];
                uint32_t scale_byte;
                mxfp8::quantize_single_block(values, values_fp8, scale_byte);
                scale_word |= scale_byte << (block * 8);
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    move<float4>::sts(
                        quant_fp8_tile::idx(
                            fp8_addr, {tile_row, block * 32 + k * 16}),
                        float4{
                            __uint_as_float(values_fp8[k * 4]),
                            __uint_as_float(values_fp8[k * 4 + 1]),
                            __uint_as_float(values_fp8[k * 4 + 2]),
                            __uint_as_float(values_fp8[k * 4 + 3]),
                        });
                }
            };

            auto swiglu_block = [&] (
                bf16_2 (&up_values)[16], const int block,
                uint32_t &hidden_scale_word
            ) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    bf16_2 gate_bf16;
                    move<bf16_2>::lds(
                        gate_bf16,
                        quant_bf16_tile::idx(
                            hidden_bf16_addr, {tile_row, block * 32 + j * 2}));
                    float2 gate = __bfloat1622float2(gate_bf16);
                    float2 up = __bfloat1622float2(up_values[j]);
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
                    up_values[j] = __floats2bfloat162_rn(gate.x, gate.y);
                }
                // Quantize the already-rounded BF16 hidden values while they
                // are still in registers.  Only macro 0 needs the BF16 tile
                // in shared memory later for transposed context generation.
                quantize_row_block(
                    up_values, hidden_q_fp8_addr, block, hidden_scale_word);
                if (save_context)
                    store_hidden_block(up_values, block);
            };

            const uint32_t scale_offset =
                (tile_row % 32) * 16 + (tile_row / 32) * 4;

            if (save_context) {
                uint32_t gate_scale_word = 0;
                uint32_t up_scale_word = 0;
                // Macro 0 first pass: materialize Gate in hidden scratch and
                // build both independent context q/sc sources.  Up is loaded
                // again below, avoiding a full-tile register lifetime.
                #pragma unroll 1
                for (int block = 0; block < config::SWIGLU_Nb / 32; ++block) {
                    bf16_2 values[16];
                    load_tmem_bf16_block(block, values);
                    quantize_row_block(
                        values, gate_q_fp8_addr, block, gate_scale_word);
                    store_hidden_block(values, block);

                    load_tmem_bf16_block(
                        block + config::SWIGLU_Nb / 32, values);
                    quantize_row_block(
                        values, up_q_fp8_addr, block, up_scale_word);
                }

                move<int>::sts(
                    gate_q_sc_addr + scale_offset,
                    std::bit_cast<int>(gate_scale_word));
                move<int>::sts(
                    up_q_sc_addr + scale_offset,
                    std::bit_cast<int>(up_scale_word));

                // Publish every q/sc source before the elected lane issues
                // both context streams.  TMEM remains live for the Up reread.
                tensor_before_thread_sync();
                epilogue_group::sync(1);
                if (epilogue_group::laneid() == 0) {
                    tma::store_async(gate_context_gmem, gate_q_fp8_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(*gate_context_sc_gmem, gate_q_sc_smem,
                                     {output_row, tile_coord.y, 0, 0});
                    tma::store_async(up_context_gmem, up_q_fp8_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(*up_context_sc_gmem, up_q_sc_smem,
                                     {output_row, tile_coord.y, 0, 0});
                }
                // All epilogue warps enter SwiGLU only after both context
                // payload/scale streams have been issued.
                epilogue_group::sync(1);

                // The context stores now overlap the second-pass Up loads and
                // SwiGLU.  Release TMEM immediately after the final Up block
                // reaches registers, before computing that block.
                uint32_t hidden_scale_word = 0;
                #pragma unroll 1
                for (int block = 0; block < config::SWIGLU_Nb / 32; ++block) {
                    bf16_2 up_values[16];
                    load_tmem_bf16_block(
                        block + config::SWIGLU_Nb / 32, up_values);
                    if (block == config::SWIGLU_Nb / 32 - 1) {
                        tensor_before_thread_sync();
                        epilogue_group::sync(1);
                        warpgroup::tma::cluster::arrive(
                            gemm_outputs_finished, 0);
                    }
                    swiglu_block(up_values, block, hidden_scale_word);
                }
                move<int>::sts(
                    hidden_q_sc_addr + scale_offset,
                    std::bit_cast<int>(hidden_scale_word));
            } else {
                // Macro > 0 keeps V3b's single-pass path: no context is saved,
                // so rereading Up would provide no overlap benefit.
                uint32_t hidden_scale_word = 0;
                #pragma unroll 1
                for (int block = 0; block < config::SWIGLU_Nb / 32; ++block) {
                    bf16_2 values[16];
                    load_tmem_bf16_block(block, values);
                    store_hidden_block(values, block);

                    load_tmem_bf16_block(
                        block + config::SWIGLU_Nb / 32, values);
                    if (block == config::SWIGLU_Nb / 32 - 1) {
                        tensor_before_thread_sync();
                        epilogue_group::sync(1);
                        warpgroup::tma::cluster::arrive(
                            gemm_outputs_finished, 0);
                    }
                    swiglu_block(values, block, hidden_scale_word);
                }
                move<int>::sts(
                    hidden_q_sc_addr + scale_offset,
                    std::bit_cast<int>(hidden_scale_word));
            }

            if (save_context && epilogue_group::laneid() == 0)
                tma::store_async_read_wait();
            // Context q/sc cannot be overwritten for hidden transpose until
            // the elected lane has observed source-read completion.  The sync
            // publishes hidden-normal q/sc for TMA and, when context is saved,
            // every hidden BF16 row for the transposed quantization pass.
            epilogue_group::sync(1);

            if (save_context) {
                if constexpr (NUM_DEVICES == 8) {
                    // EP8 hidden normal uses disjoint q/sc scratch, so issue
                    // its TMA stores before transposed quantization and
                    // overlap the copy with that pass.  The final full wait
                    // still covers every store before publishing Down ready.
                    if (epilogue_group::laneid() == 0) {
                        tma::store_async(hidden_gmem, hidden_q_fp8_smem,
                                         {output_row, tile_coord.y});
                        tma::store_async(*hidden_sc_gmem, hidden_q_sc_smem,
                                         {output_row, tile_coord.y, 0, 0});
                    }
                }
                mxfp8::quantize_tile<
                    false, true, config::SWIGLU_Nb, false, false, false>(
                    hidden_bf16_smem,
                    gate_q_fp8_smem, gate_q_sc_smem,
                    gate_q_fp8_smem, gate_q_sc_smem, nullptr,
                    epilogue_group::laneid(), 1);
                epilogue_group::sync(1);
            }

            if (epilogue_group::laneid() == 0) {
                if constexpr (NUM_DEVICES == 8) {
                    if (!save_context) {
                        tma::store_async(hidden_gmem, hidden_q_fp8_smem,
                                         {output_row, tile_coord.y});
                        tma::store_async(*hidden_sc_gmem, hidden_q_sc_smem,
                                         {output_row, tile_coord.y, 0, 0});
                    }
                } else {
                    tma::store_async(hidden_gmem, hidden_q_fp8_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(*hidden_sc_gmem, hidden_q_sc_smem,
                                     {output_row, tile_coord.y, 0, 0});
                }
                if (save_context) {
                    tma::store_async(*hidden_t_gmem, gate_q_fp8_smem,
                                     {tile_coord.y, output_row});
                    tma::store_async(*hidden_sc_t_gmem, gate_q_sc_smem,
                                     {tile_coord.y, output_row, 0, 0});
                }
                tma::store_async_wait();
                barrier_arrive(
                    hidden_row_block_ready,
                    hidden_row_block_ready_base_index
                        + macrobatch_row_block_offset + tile_coord.x);
            }
        } else {
            // Shared/routed-BF16 path: materialize Gate and Up. A pure-BF16
            // instantiation writes SwiGLU to a disjoint Hidden tile, allowing
            // the context TMA stores to read Gate/Up concurrently. The shared
            // BF16 task inside an MXFP8 instantiation retains the old in-place
            // layout because its next routed task can already use the scale ring.
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
                    : "r"(d_tt.addr + ((warpgroup::warpid() * 32) << 16)
                          + block * 32));
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

            if (save_context) {
                if (epilogue_group::laneid() == 0) {
                    tma::store_async(gate_context_gmem, gate_raw_smem,
                                     {output_row, tile_coord.y});
                    tma::store_async(up_context_gmem, up_raw_smem,
                                     {output_row, tile_coord.y});
                    if constexpr (!PURE_BF16)
                        tma::store_async_read_wait();
                }
                epilogue_group::sync(1);
            }

            const auto *gate_pairs =
                reinterpret_cast<const bf16_2 *>(gate_raw_smem.data);
            const auto *up_pairs =
                reinterpret_cast<const bf16_2 *>(up_raw_smem.data);
            auto *hidden_pairs = reinterpret_cast<bf16_2 *>(
                PURE_BF16 ? hidden_raw_smem.data : gate_raw_smem.data);
            constexpr int NUM_PAIRS = config::SWIGLU_Mb * config::SWIGLU_Nb / 2;
            constexpr int EPILOGUE_THREADS = WARPGROUP_WARPS * WARP_THREADS;
            #pragma unroll 1
            for (int i = epilogue_group::laneid(); i < NUM_PAIRS;
                 i += EPILOGUE_THREADS) {
                float2 gate = __bfloat1622float2(gate_pairs[i]);
                float2 up = __bfloat1622float2(up_pairs[i]);
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
                hidden_pairs[i] = __floats2bfloat162_rn(gate.x, gate.y);
            }
            epilogue_group::sync(1);

            if (epilogue_group::laneid() == 0) {
                if constexpr (PURE_BF16)
                    tma::store_async(hidden_gmem, hidden_raw_smem,
                                     {output_row, tile_coord.y});
                else
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
}
