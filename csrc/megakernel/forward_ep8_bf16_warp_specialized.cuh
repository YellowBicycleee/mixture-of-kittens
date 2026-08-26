// BF16-only EP8 persistent forward with same-CTA communication and compute.
//
// This header is included inside dispatch_mlp_swiglu_combiner, after types.cuh.
// DeepGEMM MegaMoE is used only for the role split and lifetime discipline:
// WG0 pulls routed activations, WG1 owns TMA/tcgen05, and WG2 owns the
// training epilogue.  Gate, Up, and Hidden remain BF16 and their saved
// backward context is committed with TMA stores.

struct ep8_bf16_ws_launch_config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_WARPGROUPS = EP8_BF16_WS_NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = EP8_BF16_WS_NUM_WARPS;
    static constexpr int NUM_THREADS = EP8_BF16_WS_NUM_THREADS;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int NUM_CLUSTERS = NUM_BLOCKS / CLUSTER_SIZE;
    static constexpr int MIN_BLOCKS_PER_SM = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        EP8_BF16_WS_SMEM_CAPACITY_BYTES;
};

static_assert(ep8_bf16_ws_launch_config::NUM_THREADS == 384);
static_assert(ep8_bf16_ws_launch_config::NUM_BLOCKS == 148);
static_assert(ep8_bf16_ws_launch_config::NUM_CLUSTERS == 74);
static_assert(
    ep8_bf16_ws_launch_config::NUM_BLOCKS
        % ep8_bf16_ws_launch_config::CLUSTER_SIZE
    == 0);

using ep8_bf16_ws_accumulator =
    tt<float, config::MLP_Mb / config::CLUSTER_SIZE, config::MLP_Nb>;
static_assert(ep8_bf16_ws_accumulator::rows == 128);
static_assert(ep8_bf16_ws_accumulator::cols == 256);
static_assert(
    EP8_BF16_WS_TMEM_STAGES * ep8_bf16_ws_accumulator::cols
    == tensor_allocator<1, config::CLUSTER_SIZE>::cols);

static constexpr int EP8_BF16_WS_TMEM_LOAD_COLS = 8;
static constexpr int EP8_BF16_WS_EPI_BARRIER = 1;
static constexpr int EP8_BF16_WS_COMM_REGISTERS = 64;
static constexpr int EP8_BF16_WS_MAINLOOP_REGISTERS = 64;
static constexpr int EP8_BF16_WS_EPI_REGISTERS = 160;

static __device__ __forceinline__ uint32_t
ep8_bf16_ws_pair_as_uint(const bf16_2 value) {
    return static_cast<uint32_t>(
               __bfloat16_as_ushort(__low2bfloat16(value)))
        | (static_cast<uint32_t>(
               __bfloat16_as_ushort(__high2bfloat16(value)))
           << 16);
}

static __device__ __forceinline__ void ep8_bf16_ws_load_tmem_8(
    const ep8_bf16_ws_accumulator &d_tt,
    const int block,
    bf16_2 (&values)[EP8_BF16_WS_TMEM_LOAD_COLS / 2]
) {
    float2 tmp[EP8_BF16_WS_TMEM_LOAD_COLS / 2];
    asm volatile(R"(
        tcgen05.ld.sync.aligned.32x32b.x8.b32
        {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];
        )"
        : "=f"(tmp[0].x), "=f"(tmp[0].y),
          "=f"(tmp[1].x), "=f"(tmp[1].y),
          "=f"(tmp[2].x), "=f"(tmp[2].y),
          "=f"(tmp[3].x), "=f"(tmp[3].y)
        : "r"(d_tt.addr
              + ((warpgroup::warpid() * 32) << 16)
              + block * EP8_BF16_WS_TMEM_LOAD_COLS));
    tensor_load_wait();
    #pragma unroll
    for (int i = 0; i < EP8_BF16_WS_TMEM_LOAD_COLS / 2; ++i)
        values[i] = __float22bfloat162_rn(tmp[i]);
}

template <typename GL>
static __device__ __forceinline__ void ep8_bf16_ws_store_8(
    const GL &dst,
    const int row,
    const int col,
    const bf16_2 (&values)[EP8_BF16_WS_TMEM_LOAD_COLS / 2]
) {
    auto *out = reinterpret_cast<float4 *>(
        dst.raw_ptr + static_cast<size_t>(row) * dst.cols() + col);
    out[0] = float4{
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[0])),
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[1])),
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[2])),
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[3]))};
}

static __device__ __forceinline__ void ep8_bf16_ws_store_peer_8(
    bf16 *dst,
    const int cols,
    const int row,
    const int col,
    const bf16_2 (&values)[EP8_BF16_WS_TMEM_LOAD_COLS / 2]
) {
    auto *out = reinterpret_cast<float4 *>(
        dst + static_cast<size_t>(row) * cols + col);
    out[0] = float4{
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[0])),
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[1])),
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[2])),
        __uint_as_float(ep8_bf16_ws_pair_as_uint(values[3]))};
}

// One COMM warp owns one 32-row x 64-column (4 KiB) buffer.  Four warps
// therefore make independent progress without a CTA-wide synchronization.
static __device__ __forceinline__ void ep8_bf16_ws_dispatch_pull(
    const globals_fwd &g,
    ep8_bf16_ws_smem_storage &storage,
    uint32_t &comm_bitfield,
    const int num_tokens,
    const int macrobatch_idx,
    const int previous_macrobatch_idx,
    const int warp_task_idx
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);
    if (warpgroup::groupid() != 0)
        return;

    constexpr int ROWS_PER_TASK = 32;
    constexpr int COLS_PER_TASK = 64;
    constexpr int BYTES_PER_ROW = COLS_PER_TASK * sizeof(bf16);
    static_assert(ROWS_PER_TASK * BYTES_PER_ROW
                  == EP8_BF16_WS_COMM_BUFFER_BYTES);

    const int comm_warp = warpgroup::warpid();
    const int lane = warp::laneid();
    const int macro_first_row = macrobatch_idx * g.macrobatch_size;
    const int macro_rows = max(
        0, min(g.macrobatch_size, num_tokens - macro_first_row));
    const int row_chunks =
        (macro_rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
    const int col_chunks =
        (g.x_fp8_routed.cols() + COLS_PER_TASK - 1) / COLS_PER_TASK;
    if (warp_task_idx >= row_chunks * col_chunks)
        return;

    const int row_chunk = warp_task_idx / col_chunks;
    const int col_chunk = warp_task_idx - row_chunk * col_chunks;
    const int local_row = row_chunk * ROWS_PER_TASK + lane;
    const int global_row = macro_first_row + local_row;
    const int col = col_chunk * COLS_PER_TASK;
    const int chunk_cols = min(COLS_PER_TASK, g.x_fp8_routed.cols() - col);
    const uint32_t chunk_bytes = chunk_cols * sizeof(bf16);
    const bool valid_row = local_row < macro_rows;
    const int peer_rank = valid_row
        ? g.schedule_peer_rank[{global_row}]
        : -1;
    const int peer_token_idx = peer_rank >= 0
        ? g.schedule_peer_token_idx[{global_row}]
        : 0;
    const uint32_t valid_mask = __ballot_sync(0xffffffff, peer_rank >= 0);

    auto *warp_buffer = reinterpret_cast<bf16 *>(
        storage.comm_smem[comm_warp]);
    auto *lane_buffer = warp_buffer + lane * COLS_PER_TASK;

    // The next reverse macro may reuse its X ring only as the corresponding
    // current-macro output minibatch becomes globally visible.  This keeps
    // COMM live during Down without relying on a host-side phase decision.
    if (lane == 0 && previous_macrobatch_idx >= 0) {
        const int previous_macro_first =
            previous_macrobatch_idx * g.macrobatch_size;
        const int previous_macro_rows = max(
            0, min(g.macrobatch_size,
                   num_tokens - previous_macro_first));
        const int previous_local_row = row_chunk * ROWS_PER_TASK;
        if (previous_local_row < previous_macro_rows) {
            const int previous_global_row =
                previous_macro_first + previous_local_row;
            const int previous_minibatch_idx =
                previous_global_row / g.minibatch_size;
            const int previous_minibatch_first =
                previous_minibatch_idx * g.minibatch_size;
            const int previous_minibatch_rows = max(
                0, min(g.minibatch_size,
                       num_tokens - previous_minibatch_first));
            const int required_count =
                ((previous_minibatch_rows + config::MLP_Mb - 1)
                 / config::MLP_Mb)
                * (g.y_routed.cols() / config::MLP_Nb)
                * config::CLUSTER_SIZE;
            barrier_wait(
                g.y_routed_ready,
                previous_minibatch_idx,
                required_count);
        }
    }
    __syncwarp();

    wait(
        storage.comm_empty[comm_warp],
        get_phasebit<1>(comm_bitfield, comm_warp));
    update_phasebit<1>(comm_bitfield, comm_warp);
    if (lane == 0)
        tma::expect_bytes(
            storage.comm_full[comm_warp],
            __popc(valid_mask) * chunk_bytes);
    __syncwarp();

    if (peer_rank >= 0) {
        tma::load_async(
            lane_buffer,
            &g.x_routed_send_buffer[peer_rank][
                static_cast<size_t>(peer_token_idx / g.topk)
                    * g.x_fp8_routed.cols()
                + col],
            chunk_bytes,
            storage.comm_full[comm_warp]);
    } else if (valid_row) {
        auto *words = reinterpret_cast<float4 *>(lane_buffer);
        #pragma unroll
        for (int i = 0; i < BYTES_PER_ROW / sizeof(float4); ++i)
            words[i] = float4{0.0f, 0.0f, 0.0f, 0.0f};
    }

    wait(
        storage.comm_full[comm_warp],
        get_phasebit<0>(comm_bitfield, comm_warp));
    update_phasebit<0>(comm_bitfield, comm_warp);
    __syncwarp();
    if (valid_row) {
        tma::store_async(
            &g.x_fp8_routed.raw_ptr[
                static_cast<size_t>(local_row) * g.x_fp8_routed.cols()
                + col],
            lane_buffer,
            chunk_bytes);
        tma::store_async_wait();
    }
    __syncwarp();
    group<1>::arrive(storage.comm_empty[comm_warp]);
    if (lane == 0) {
        const int global_minibatch_idx = global_row / g.minibatch_size;
        barrier_arrive(g.x_routed_ready, global_minibatch_idx);
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ int3 ep8_bf16_ws_decode_task(
    const globals_fwd &g,
    const int macrobatch_idx,
    int task_idx,
    const int col_blocks
) {
    if constexpr (IS_SHARED) {
        const int row_blocks = g.x_shared.rows() / config::MLP_Mb;
        if (task_idx >= row_blocks * col_blocks)
            return {-1, -1, -1};
        const int2 swizzled =
            get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(
                row_blocks, col_blocks, task_idx);
        return {swizzled.x, swizzled.y, 0};
    } else {
        const int macro_first_row_block =
            macrobatch_idx * (g.macrobatch_size / config::MLP_Mb);
        const int macro_row_blocks =
            (min(g.macrobatch_size,
                 max(0, g.num_tokens[{0}]
                            - macrobatch_idx * g.macrobatch_size))
             + config::MLP_Mb - 1)
            / config::MLP_Mb;
        int expert_first_row_block = 0;
        for (int expert_idx = 0;
             expert_idx < g.w_routed_gate.depth();
             ++expert_idx) {
            const int expert_row_blocks =
                (g.tokens_per_expert[{expert_idx}] + config::MLP_Mb - 1)
                / config::MLP_Mb;
            const int first = max(
                macro_first_row_block, expert_first_row_block);
            const int end = min(
                macro_first_row_block + macro_row_blocks,
                expert_first_row_block + expert_row_blocks);
            const int row_blocks = max(0, end - first);
            const int tasks = row_blocks * col_blocks;
            if (task_idx < tasks) {
                const int2 swizzled =
                    get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(
                        row_blocks, col_blocks, task_idx);
                return {
                    first + swizzled.x - macro_first_row_block,
                    swizzled.y,
                    expert_idx};
            }
            task_idx -= tasks;
            expert_first_row_block += expert_row_blocks;
        }
        return {-1, -1, -1};
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void ep8_bf16_ws_wait_for_x(
    const globals_fwd &g,
    const int num_tokens,
    const int macrobatch_idx,
    const int local_row_block
) {
    if constexpr (!IS_SHARED) {
        const int global_row =
            macrobatch_idx * g.macrobatch_size
            + local_row_block * config::MLP_Mb;
        const int global_minibatch_idx = global_row / g.minibatch_size;
        const int minibatch_first_row =
            global_minibatch_idx * g.minibatch_size;
        const int minibatch_rows = max(
            0, min(g.minibatch_size, num_tokens - minibatch_first_row));
        constexpr int ROWS_PER_COMM_TASK = 32;
        constexpr int COLS_PER_COMM_TASK = 64;
        const int required_count =
            (minibatch_rows + ROWS_PER_COMM_TASK - 1)
                / ROWS_PER_COMM_TASK
            * ((g.x_fp8_routed.cols() + COLS_PER_COMM_TASK - 1)
               / COLS_PER_COMM_TASK);
        barrier_wait(
            g.x_routed_ready, global_minibatch_idx, required_count);
    }
}

template <int NUM_PAIRS>
static __device__ __forceinline__ void ep8_bf16_ws_store_tmem_block(
    quant_bf16_tile &dst,
    const int row,
    const int col,
    const bf16_2 (&values)[NUM_PAIRS]
) {
    static_assert(NUM_PAIRS > 0 && NUM_PAIRS % 4 == 0);
    const uint32_t dst_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&dst));
    #pragma unroll
    for (int i = 0; i < NUM_PAIRS / 4; ++i) {
        move<float4>::sts(
            quant_bf16_tile::idx(dst_addr, {row, col + i * 8}),
            float4{
                __uint_as_float(
                    ep8_bf16_ws_pair_as_uint(values[i * 4])),
                __uint_as_float(
                    ep8_bf16_ws_pair_as_uint(values[i * 4 + 1])),
                __uint_as_float(
                    ep8_bf16_ws_pair_as_uint(values[i * 4 + 2])),
                __uint_as_float(
                    ep8_bf16_ws_pair_as_uint(values[i * 4 + 3]))});
    }
}

template <bool IS_SHARED, bool IS_CLAMPED>
static __device__ __forceinline__ void ep8_bf16_ws_gate_up_stage(
    const globals_fwd &g,
    ep8_bf16_ws_smem_storage &storage,
    ep8_bf16_ws_accumulator (&d_tt)[EP8_BF16_WS_TMEM_STAGES],
    uint32_t &load_bitfield,
    uint32_t &tmem_bitfield,
    const int num_tokens,
    const int macrobatch_idx,
    const int task_idx,
    const int output_sequence,
    const int cta_rank,
    const int hidden_row_block_ready_base_index
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);
    const int col_blocks = (IS_SHARED
        ? g.w_shared_gate.rows()
        : g.w_routed_gate.rows()) / config::SWIGLU_Nb;
    const int3 tile_coord = ep8_bf16_ws_decode_task<IS_SHARED>(
        g, macrobatch_idx, task_idx, col_blocks);
    if (tile_coord.z < 0)
        return;

    const int global_warp = threadIdx.x / WARP_THREADS;
    const int iters = (IS_SHARED
        ? g.x_shared.cols()
        : g.x_fp8_routed.cols()) / config::MLP_BF16_Kb;
    const int output_stage =
        output_sequence % EP8_BF16_WS_TMEM_STAGES;

    if (global_warp == EP8_BF16_WS_TMA_A_WARP
        && warp::elect_leader()) {
        ep8_bf16_ws_wait_for_x<IS_SHARED>(
            g, num_tokens, macrobatch_idx, tile_coord.x);
        int input_stage = 0;
        for (int k_block = 0; k_block < iters; ++k_block) {
            wait(
                storage.load_empty[input_stage],
                get_phasebit<1>(load_bitfield, input_stage));
            if constexpr (IS_SHARED) {
                tma::cluster::load_async(
                    storage.a_smem[input_stage], g.x_shared,
                    {tile_coord.x * config::CLUSTER_SIZE + cta_rank,
                     k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            } else {
                tma::cluster::load_async(
                    storage.a_smem[input_stage], g.x_fp8_routed,
                    {tile_coord.x * config::CLUSTER_SIZE + cta_rank,
                     k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            }
            update_phasebit<1>(load_bitfield, input_stage);
            input_stage = ring_advance<EP8_BF16_WS_LOAD_STAGES>(
                input_stage);
        }
    } else if (
        global_warp == EP8_BF16_WS_TMA_B_WARP
        && warp::elect_leader()) {
        int input_stage = 0;
        for (int k_block = 0; k_block < iters; ++k_block) {
            wait(
                storage.load_empty[input_stage],
                get_phasebit<1>(load_bitfield, input_stage));
            if constexpr (IS_SHARED) {
                const weight_bf16_gl &weight =
                    cta_rank == 0 ? g.w_shared_gate : g.w_shared_up;
                tma::cluster::load_async(
                    storage.b_smem[input_stage], weight,
                    {tile_coord.y, k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            } else {
                const routed_weight_gl &weight =
                    cta_rank == 0 ? g.w_routed_gate : g.w_routed_up;
                tma::cluster::load_async(
                    storage.b_smem[input_stage], weight,
                    {tile_coord.z, tile_coord.y, k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            }
            update_phasebit<1>(load_bitfield, input_stage);
            input_stage = ring_advance<EP8_BF16_WS_LOAD_STAGES>(
                input_stage);
        }
    } else if (
        global_warp == EP8_BF16_WS_MMA_WARP
        && cta_rank == 0
        && warp::elect_leader()) {
        wait(
            storage.tmem_empty[output_stage],
            get_phasebit<1>(tmem_bitfield, output_stage));
        update_phasebit<1>(tmem_bitfield, output_stage);
        tensor_after_thread_sync();
        int input_stage = 0;
        for (int k_block = 0; k_block < iters; ++k_block) {
            tma::expect_bytes(
                storage.load_full[input_stage],
                config::CLUSTER_SIZE
                    * (sizeof(mlp_bf16_tile)
                       + sizeof(mlp_bf16_tile)));
            wait(
                storage.load_full[input_stage],
                get_phasebit<0>(load_bitfield, input_stage));
            if (k_block == 0) {
                mm2_ABt(
                    d_tt[output_stage],
                    storage.a_smem[input_stage],
                    storage.b_smem[input_stage],
                    storage.load_empty[input_stage]);
            } else {
                mma2_ABt(
                    d_tt[output_stage],
                    storage.a_smem[input_stage],
                    storage.b_smem[input_stage],
                    storage.load_empty[input_stage]);
            }
            update_phasebit<0>(load_bitfield, input_stage);
            input_stage = ring_advance<EP8_BF16_WS_LOAD_STAGES>(
                input_stage);
        }
        detail::tcgen05::commit<config::CLUSTER_SIZE>(
            storage.tmem_full[output_stage]);
    } else if (
        global_warp >= EP8_BF16_WS_EPI_WARP_BEGIN
        && global_warp < EP8_BF16_WS_EPI_WARP_END) {
        using epilogue_group = group<EP8_BF16_WS_EPI_WARPS>;
        wait(
            storage.tmem_full[output_stage],
            get_phasebit<0>(tmem_bitfield, output_stage));
        update_phasebit<0>(tmem_bitfield, output_stage);

        const int tile_row = epilogue_group::laneid();
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
                : "r"(d_tt[output_stage].addr
                      + ((warpgroup::warpid() * 32) << 16)
                      + block * 32));
            tensor_load_wait();
            bf16_2 values[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                values[i] = __float22bfloat162_rn(tmp[i]);
            const bool is_gate = block < config::SWIGLU_Nb / 32;
            ep8_bf16_ws_store_tmem_block(
                is_gate ? storage.gate_smem : storage.up_smem,
                tile_row,
                (block % (config::SWIGLU_Nb / 32)) * 32,
                values);
        }

        // The complete Gate/Up tile is now in SMEM.  Release this exact TMEM
        // stage before any SFU work so tcgen05(task+1) can run concurrently.
        tensor_before_thread_sync();
        epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);
        warpgroup::tma::cluster::arrive(
            storage.tmem_empty[output_stage], 0);

        const bool save_context = IS_SHARED || macrobatch_idx == 0;
        const int output_row =
            tile_coord.x * config::CLUSTER_SIZE + cta_rank;
        const auto &gate_context_gmem = [&]() -> const auto & {
            if constexpr (IS_SHARED)
                return g.gate_shared;
            else
                return g.gate_fp8_routed;
        }();
        const auto &up_context_gmem = [&]() -> const auto & {
            if constexpr (IS_SHARED)
                return g.up_shared;
            else
                return g.up_fp8_routed;
        }();
        const auto &hidden_gmem = [&]() -> const auto & {
            if constexpr (IS_SHARED)
                return g.hidden_shared;
            else
                return g.hidden_fp8_routed;
        }();
        const index_gl &hidden_row_block_ready =
            g.hidden_row_block_ready;
        if (save_context && epilogue_group::laneid() == 0) {
            tma::store_async(gate_context_gmem, storage.gate_smem,
                             {output_row, tile_coord.y});
            tma::store_async(up_context_gmem, storage.up_smem,
                             {output_row, tile_coord.y});
        }
        epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);

        const auto *gate_pairs =
            reinterpret_cast<const bf16_2 *>(storage.gate_smem.data);
        const auto *up_pairs =
            reinterpret_cast<const bf16_2 *>(storage.up_smem.data);
        auto *hidden_pairs =
            reinterpret_cast<bf16_2 *>(storage.hidden_smem.data);
        constexpr int NUM_PAIRS =
            config::SWIGLU_Mb * config::SWIGLU_Nb / 2;
        constexpr int EPI_THREADS =
            EP8_BF16_WS_EPI_WARPS * WARP_THREADS;
        #pragma unroll 1
        for (int i = epilogue_group::laneid(); i < NUM_PAIRS;
             i += EPI_THREADS) {
            float2 gate = __bfloat1622float2(gate_pairs[i]);
            float2 up = __bfloat1622float2(up_pairs[i]);
            if constexpr (IS_CLAMPED) {
                gate = {
                    fminf(gate.x, g.swiglu_limit),
                    fminf(gate.y, g.swiglu_limit)};
                up = {
                    fminf(fmaxf(up.x, -g.swiglu_limit), g.swiglu_limit),
                    fminf(fmaxf(up.y, -g.swiglu_limit), g.swiglu_limit)};
            }
            float2 denominator = base_ops::mul::op<float2>(
                gate, float2{-1.0f, -1.0f});
            denominator = base_ops::exp::op<float2>(denominator);
            denominator = base_ops::sum::op<float2>(
                denominator, float2{1.0f, 1.0f});
            gate = base_ops::div::op<float2>(gate, denominator);
            gate = base_ops::mul::op<float2>(gate, up);
            hidden_pairs[i] =
                __floats2bfloat162_rn(gate.x, gate.y);
        }
        epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);

        if (epilogue_group::laneid() == 0) {
            tma::store_async(hidden_gmem, storage.hidden_smem,
                             {output_row, tile_coord.y});
            // This full group wait covers Gate, Up, and Hidden.  Publishing
            // Hidden earlier would let Down race a backward-context source.
            tma::store_async_wait();
            barrier_arrive(hidden_row_block_ready,
                hidden_row_block_ready_base_index
                    + (IS_SHARED
                       ? tile_coord.x
                       : macrobatch_idx
                             * (g.macrobatch_size / config::MLP_Mb)
                             + tile_coord.x));
        }
        epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void ep8_bf16_ws_down_stage(
    const globals_fwd &g,
    ep8_bf16_ws_smem_storage &storage,
    ep8_bf16_ws_accumulator (&d_tt)[EP8_BF16_WS_TMEM_STAGES],
    uint32_t &load_bitfield,
    uint32_t &tmem_bitfield,
    const int num_tokens,
    const int macrobatch_idx,
    const int task_idx,
    const int output_sequence,
    const int cta_rank,
    const int hidden_row_block_ready_base_index,
    const int hidden_ready_required_count
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);
    const int col_blocks = (IS_SHARED
        ? g.w_shared_down.rows()
        : g.w_routed_down.rows()) / config::MLP_Nb;
    const int3 tile_coord = ep8_bf16_ws_decode_task<IS_SHARED>(
        g, macrobatch_idx, task_idx, col_blocks);
    if (tile_coord.z < 0)
        return;

    const int global_warp = threadIdx.x / WARP_THREADS;
    const int iters = (IS_SHARED
        ? g.hidden_shared.cols()
        : g.hidden_fp8_routed.cols()) / config::MLP_BF16_Kb;
    const int output_stage =
        output_sequence % EP8_BF16_WS_TMEM_STAGES;

    if (global_warp == EP8_BF16_WS_TMA_A_WARP
        && warp::elect_leader()) {
        barrier_wait(
            g.hidden_row_block_ready,
            hidden_row_block_ready_base_index
                + (IS_SHARED
                   ? tile_coord.x
                   : macrobatch_idx
                         * (g.macrobatch_size / config::MLP_Mb)
                         + tile_coord.x),
            hidden_ready_required_count);
        int input_stage = 0;
        for (int k_block = 0; k_block < iters; ++k_block) {
            wait(
                storage.load_empty[input_stage],
                get_phasebit<1>(load_bitfield, input_stage));
            if constexpr (IS_SHARED) {
                tma::cluster::load_async(
                    storage.a_smem[input_stage], g.hidden_shared,
                    {tile_coord.x * config::CLUSTER_SIZE + cta_rank,
                     k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            } else {
                tma::cluster::load_async(
                    storage.a_smem[input_stage], g.hidden_fp8_routed,
                    {tile_coord.x * config::CLUSTER_SIZE + cta_rank,
                     k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            }
            update_phasebit<1>(load_bitfield, input_stage);
            input_stage = ring_advance<EP8_BF16_WS_LOAD_STAGES>(
                input_stage);
        }
    } else if (
        global_warp == EP8_BF16_WS_TMA_B_WARP
        && warp::elect_leader()) {
        int input_stage = 0;
        for (int k_block = 0; k_block < iters; ++k_block) {
            wait(
                storage.load_empty[input_stage],
                get_phasebit<1>(load_bitfield, input_stage));
            if constexpr (IS_SHARED) {
                tma::cluster::load_async(
                    storage.b_smem[input_stage], g.w_shared_down,
                    {tile_coord.y * config::CLUSTER_SIZE + cta_rank,
                     k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            } else {
                tma::cluster::load_async(
                    storage.b_smem[input_stage], g.w_routed_down,
                    {tile_coord.z,
                     tile_coord.y * config::CLUSTER_SIZE + cta_rank,
                     k_block},
                    storage.load_full[input_stage],
                    static_cast<uint16_t>(1 << cta_rank), 0);
            }
            update_phasebit<1>(load_bitfield, input_stage);
            input_stage = ring_advance<EP8_BF16_WS_LOAD_STAGES>(
                input_stage);
        }
    } else if (
        global_warp == EP8_BF16_WS_MMA_WARP
        && cta_rank == 0
        && warp::elect_leader()) {
        wait(
            storage.tmem_empty[output_stage],
            get_phasebit<1>(tmem_bitfield, output_stage));
        update_phasebit<1>(tmem_bitfield, output_stage);
        tensor_after_thread_sync();
        int input_stage = 0;
        for (int k_block = 0; k_block < iters; ++k_block) {
            tma::expect_bytes(
                storage.load_full[input_stage],
                config::CLUSTER_SIZE
                    * (sizeof(mlp_bf16_tile)
                       + sizeof(mlp_bf16_tile)));
            wait(
                storage.load_full[input_stage],
                get_phasebit<0>(load_bitfield, input_stage));
            if (k_block == 0) {
                mm2_ABt(
                    d_tt[output_stage],
                    storage.a_smem[input_stage],
                    storage.b_smem[input_stage],
                    storage.load_empty[input_stage]);
            } else {
                mma2_ABt(
                    d_tt[output_stage],
                    storage.a_smem[input_stage],
                    storage.b_smem[input_stage],
                    storage.load_empty[input_stage]);
            }
            update_phasebit<0>(load_bitfield, input_stage);
            input_stage = ring_advance<EP8_BF16_WS_LOAD_STAGES>(
                input_stage);
        }
        detail::tcgen05::commit<config::CLUSTER_SIZE>(
            storage.tmem_full[output_stage]);
    } else if (
        global_warp >= EP8_BF16_WS_EPI_WARP_BEGIN
        && global_warp < EP8_BF16_WS_EPI_WARP_END) {
        using epilogue_group = group<EP8_BF16_WS_EPI_WARPS>;
        wait(
            storage.tmem_full[output_stage],
            get_phasebit<0>(tmem_bitfield, output_stage));
        update_phasebit<0>(tmem_bitfield, output_stage);
        const int tid = epilogue_group::laneid();
        const int output_row =
            tile_coord.x * config::MLP_Mb
            + cta_rank * (config::MLP_Mb / config::CLUSTER_SIZE)
            + tid;

        if constexpr (IS_SHARED) {
            #pragma unroll 1
            for (int block = 0;
                 block < config::MLP_Nb / EP8_BF16_WS_TMEM_LOAD_COLS;
                 ++block) {
                bf16_2 values[EP8_BF16_WS_TMEM_LOAD_COLS / 2];
                ep8_bf16_ws_load_tmem_8(
                    d_tt[output_stage], block, values);
                ep8_bf16_ws_store_tmem_block(
                    block < config::SWIGLU_Nb
                                / EP8_BF16_WS_TMEM_LOAD_COLS
                        ? storage.gate_smem
                        : storage.up_smem,
                    tid,
                    (block
                     % (config::SWIGLU_Nb
                        / EP8_BF16_WS_TMEM_LOAD_COLS))
                        * EP8_BF16_WS_TMEM_LOAD_COLS,
                    values);
            }
            tensor_before_thread_sync();
            epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);
            warpgroup::tma::cluster::arrive(
                storage.tmem_empty[output_stage], 0);
            if (tid == 0) {
                const int output_tile_row =
                    tile_coord.x * config::CLUSTER_SIZE + cta_rank;
                tma::store_async(
                    g.y_shared, storage.gate_smem,
                    {output_tile_row, tile_coord.y * 2});
                tma::store_async(
                    g.y_shared, storage.up_smem,
                    {output_tile_row, tile_coord.y * 2 + 1});
                tma::store_async_wait();
            }
            epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);
        } else {
            const int global_schedule_row =
                macrobatch_idx * g.macrobatch_size + output_row;
            const bool valid_row = global_schedule_row < num_tokens;
            const int peer_rank = valid_row
                ? g.schedule_peer_rank[{global_schedule_row}]
                : -1;
            const int peer_token_idx = peer_rank >= 0
                ? g.schedule_peer_token_idx[{global_schedule_row}]
                : 0;
            #pragma unroll 1
            for (int block = 0;
                 block < config::MLP_Nb / EP8_BF16_WS_TMEM_LOAD_COLS;
                 ++block) {
                bf16_2 values[EP8_BF16_WS_TMEM_LOAD_COLS / 2];
                ep8_bf16_ws_load_tmem_8(
                    d_tt[output_stage], block, values);
                if (block
                    == config::MLP_Nb / EP8_BF16_WS_TMEM_LOAD_COLS
                           - 1) {
                    tensor_before_thread_sync();
                    epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);
                    warpgroup::tma::cluster::arrive(
                        storage.tmem_empty[output_stage], 0);
                }
                const int col = tile_coord.y * config::MLP_Nb
                    + block * EP8_BF16_WS_TMEM_LOAD_COLS;
                if (macrobatch_idx == 0 && valid_row)
                    ep8_bf16_ws_store_8(
                        g.y_routed, output_row, col, values);
                if (peer_rank >= 0)
                    ep8_bf16_ws_store_peer_8(
                        g.y_routed_recv_buffer[peer_rank],
                        g.y_routed.cols(), peer_token_idx, col, values);
            }
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __threadfence_system();
            epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);
            if (tid == 0) {
                const int global_minibatch_idx =
                    global_schedule_row / g.minibatch_size;
                barrier_arrive(g.y_routed_ready, global_minibatch_idx);
            }
            epilogue_group::sync(EP8_BF16_WS_EPI_BARRIER);
        }
    }
}

template <bool IS_CLAMPED>
static __device__ __forceinline__ void
dispatch_mlp_swiglu_combine_fwd_bf16_ep8_warp_specialized_kernel(
    const globals_fwd &g
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);
    static_assert(EP8_BF16_WS_NUM_THREADS == 384);

    const int warpgroup_idx = warpgroup::groupid();
    if (warpgroup_idx == 0) {
        warpgroup::decrease_registers<EP8_BF16_WS_COMM_REGISTERS>();
    } else if (warpgroup_idx == 1) {
        warpgroup::decrease_registers<EP8_BF16_WS_MAINLOOP_REGISTERS>();
    }
    __syncthreads();
    if (warpgroup_idx == 2)
        warpgroup::increase_registers<EP8_BF16_WS_EPI_REGISTERS>();
    __syncthreads();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr =
        (reinterpret_cast<uint64_t>(&__shm[0])
         + EP8_BF16_WS_SMEM_ALIGNMENT_SLACK_BYTES)
        & ~uint64_t(EP8_BF16_WS_SMEM_ALIGNMENT_BYTES - 1);
    auto &storage = *reinterpret_cast<ep8_bf16_ws_smem_storage *>(
        smem_base_addr);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < EP8_BF16_WS_LOAD_STAGES; ++stage) {
            init_semaphore(storage.load_full[stage], 0, 1);
            init_semaphore(storage.load_empty[stage], 0, 1);
        }
        #pragma unroll
        for (int warp = 0; warp < EP8_BF16_WS_COMM_WARPS; ++warp) {
            init_semaphore(storage.comm_full[warp], 0, 1);
            init_semaphore(storage.comm_empty[warp], 0, 1);
        }
        #pragma unroll
        for (int stage = 0; stage < EP8_BF16_WS_TMEM_STAGES; ++stage) {
            init_semaphore(storage.tmem_full[stage], 0, 1);
            init_semaphore(
                storage.tmem_empty[stage], 0, config::CLUSTER_SIZE);
        }
    }
    __syncthreads();

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    ep8_bf16_ws_accumulator d_tt[EP8_BF16_WS_TMEM_STAGES] = {
        tm_alloc.template allocate<ep8_bf16_ws_accumulator>(
            EP8_BF16_WS_TMEM_STAGE_0_OFFSET),
        tm_alloc.template allocate<ep8_bf16_ws_accumulator>(
            EP8_BF16_WS_TMEM_STAGE_1_OFFSET)};
    everyone::tma::cluster::sync();

    uint32_t load_bitfield = 0xFFFF0000;
    uint32_t tmem_bitfield = 0xFFFF0000;
    uint32_t comm_bitfield = 0xFFFF0000;
    int output_sequence = 0;

    const int num_tokens = g.num_tokens[{0}];
    const int num_macrobatches =
        (num_tokens + g.macrobatch_size - 1) / g.macrobatch_size;
    const int cta_rank = cluster_ctarank();
    const int cluster_idx = clusterIdx().x;
    constexpr int CLUSTER_STRIDE = ep8_bf16_ws_launch_config::NUM_CLUSTERS;
    constexpr int CTA_STRIDE = ep8_bf16_ws_launch_config::NUM_BLOCKS;
    const int shared_row_blocks =
        g.x_shared.rows() / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks
        * (g.w_shared_gate.rows() / config::SWIGLU_Nb);
    const int shared_down_tasks = shared_row_blocks
        * (g.w_shared_down.rows() / config::MLP_Nb);
    const int shared_hidden_ready_count =
        config::CLUSTER_SIZE
        * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int routed_hidden_ready_count =
        config::CLUSTER_SIZE
        * (g.hidden_fp8_routed.cols() / config::SWIGLU_Nb);

    auto dispatch_macro = [&] (
        const int macrobatch_idx,
        const int previous_macrobatch_idx
    ) {
        if (macrobatch_idx < 0)
            return;
        constexpr int ROWS_PER_TASK = 32;
        constexpr int COLS_PER_TASK = 64;
        const int macro_rows = max(
            0, min(g.macrobatch_size,
                   num_tokens - macrobatch_idx * g.macrobatch_size));
        const int num_tasks =
            ((macro_rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK)
            * ((g.x_fp8_routed.cols() + COLS_PER_TASK - 1)
               / COLS_PER_TASK);
        const int warp = warpgroup::warpid();
        for (int task_idx = blockIdx.x * EP8_BF16_WS_COMM_WARPS + warp;
             task_idx < num_tasks;
             task_idx += CTA_STRIDE * EP8_BF16_WS_COMM_WARPS) {
            ep8_bf16_ws_dispatch_pull(
                g, storage, comm_bitfield, num_tokens,
                macrobatch_idx, previous_macrobatch_idx, task_idx);
        }
    };

    auto run_gate_up = [&]<bool IS_SHARED>(
        const int macrobatch_idx, const int num_tasks) {
        for (int task_idx = cluster_idx;
             task_idx < num_tasks;
             task_idx += CLUSTER_STRIDE, ++output_sequence) {
            ep8_bf16_ws_gate_up_stage<IS_SHARED, IS_CLAMPED>(
                g, storage, d_tt, load_bitfield, tmem_bitfield,
                num_tokens, macrobatch_idx, task_idx,
                output_sequence, cta_rank,
                IS_SHARED ? 0 : shared_row_blocks);
        }
    };
    auto run_down = [&]<bool IS_SHARED>(
        const int macrobatch_idx, const int num_tasks) {
        for (int task_idx = cluster_idx;
             task_idx < num_tasks;
             task_idx += CLUSTER_STRIDE, ++output_sequence) {
            ep8_bf16_ws_down_stage<IS_SHARED>(
                g, storage, d_tt, load_bitfield, tmem_bitfield,
                num_tokens, macrobatch_idx, task_idx,
                output_sequence, cta_rank,
                IS_SHARED ? 0 : shared_row_blocks,
                IS_SHARED
                    ? shared_hidden_ready_count
                    : routed_hidden_ready_count);
        }
    };

    auto grid = cooperative_groups::this_grid();
    // Initial remote pull overlaps shared Gate/Up/SwiGLU.  The first grid
    // barrier is the ownership handoff for the routed X ring.
    if (warpgroup_idx == 0) {
        if (num_macrobatches > 0)
            dispatch_macro(num_macrobatches - 1, -1);
    } else {
        run_gate_up.template operator()<true>(0, shared_gate_up_tasks);
    }
    grid.sync();

    if (warpgroup_idx != 0)
        run_down.template operator()<true>(0, shared_down_tasks);
    grid.sync();

    // Reverse macro order preserves the one-macrobatch activation ring.  At
    // each iteration, WG0 fills macro-1 only after Gate/Up has stopped reading
    // the current X ring, while WG1/WG2 execute current Down.
    for (int macrobatch_idx = num_macrobatches - 1;
         macrobatch_idx >= 0;
         --macrobatch_idx) {
        const int macro_rows = max(
            0, min(g.macrobatch_size,
                   num_tokens - macrobatch_idx * g.macrobatch_size));
        const int macro_row_blocks =
            (macro_rows + config::MLP_Mb - 1) / config::MLP_Mb;
        const int gate_up_tasks = macro_row_blocks
            * (g.w_routed_gate.rows() / config::SWIGLU_Nb);
        const int down_tasks = macro_row_blocks
            * (g.w_routed_down.rows() / config::MLP_Nb);
        if (warpgroup_idx != 0)
            run_gate_up.template operator()<false>(
                macrobatch_idx, gate_up_tasks);
        grid.sync();

        if (warpgroup_idx == 0) {
            if (macrobatch_idx > 0)
                dispatch_macro(macrobatch_idx - 1, macrobatch_idx);
        } else {
            run_down.template operator()<false>(
                macrobatch_idx, down_tasks);
        }
        grid.sync();
    }
}

template <bool IS_CLAMPED>
static __host__ __forceinline__ void
launch_fwd_ep8_bf16_warp_specialized(const globals_fwd &g) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);
    using launch_config = ep8_bf16_ws_launch_config;
    constexpr int DYNAMIC_SMEM = launch_config::DYNAMIC_SHARED_MEMORY;
    static_assert(DYNAMIC_SMEM == 231424);
    static_assert(
        sizeof(ep8_bf16_ws_smem_storage)
            + EP8_BF16_WS_SMEM_ALIGNMENT_SLACK_BYTES
        <= DYNAMIC_SMEM);

    // These are shape gates, not a host scheduling decision: num_tokens and
    // every reverse-macro tail remain device-resident and device-decoded.
    TORCH_CHECK(
        g.macrobatch_size > 0
            && g.macrobatch_size % config::MLP_Mb == 0,
        "EP8 BF16 warp specialization requires macrobatch_size divisible by ",
        config::MLP_Mb);
    TORCH_CHECK(
        g.minibatch_size > 0
            && g.minibatch_size % config::MLP_Mb == 0
            && g.macrobatch_size % g.minibatch_size == 0,
        "EP8 BF16 warp specialization requires 256-row-aligned minibatches "
        "that divide macrobatch_size");
    TORCH_CHECK(
        g.x_fp8_routed.cols() % config::MLP_BF16_Kb == 0
            && g.hidden_fp8_routed.cols() % config::MLP_BF16_Kb == 0,
        "EP8 BF16 warp specialization requires K divisible by 64");
    TORCH_CHECK(
        g.w_routed_gate.rows() % config::SWIGLU_Nb == 0
            && g.w_shared_gate.rows() % config::SWIGLU_Nb == 0
            && g.w_routed_down.rows() % config::MLP_Nb == 0
            && g.w_shared_down.rows() % config::MLP_Nb == 0,
        "EP8 BF16 warp specialization requires I divisible by 128 and "
        "H divisible by 256");

    cudaLaunchAttribute launch_attributes[2]{};
    launch_attributes[0].id = cudaLaunchAttributeClusterDimension;
    launch_attributes[0].val.clusterDim.x = launch_config::CLUSTER_SIZE;
    launch_attributes[0].val.clusterDim.y = 1;
    launch_attributes[0].val.clusterDim.z = 1;
    launch_attributes[1].id = cudaLaunchAttributeCooperative;
    launch_attributes[1].val.cooperative = 1;

    cudaLaunchConfig_t cuda_config{};
    cuda_config.gridDim = dim3(launch_config::NUM_BLOCKS, 1, 1);
    cuda_config.blockDim = dim3(launch_config::NUM_THREADS, 1, 1);
    cuda_config.dynamicSmemBytes = DYNAMIC_SMEM;
    cuda_config.attrs = launch_attributes;
    cuda_config.numAttrs = 2;
    cuda_config.stream = at::cuda::getCurrentCUDAStream();

    int device = -1;
    CUDACHECK(cudaGetDevice(&device));
    static thread_local int validated_device = -1;
    if (validated_device != device) {
        int sm_count = 0;
        int cooperative_launch = 0;
        int cluster_launch = 0;
        int major = 0;
        int minor = 0;
        CUDACHECK(cudaDeviceGetAttribute(
            &sm_count, cudaDevAttrMultiProcessorCount, device));
        CUDACHECK(cudaDeviceGetAttribute(
            &cooperative_launch, cudaDevAttrCooperativeLaunch, device));
        CUDACHECK(cudaDeviceGetAttribute(
            &cluster_launch, cudaDevAttrClusterLaunch, device));
        CUDACHECK(cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device));
        CUDACHECK(cudaDeviceGetAttribute(
            &minor, cudaDevAttrComputeCapabilityMinor, device));
        TORCH_CHECK(
            major == 10 && minor == 3,
            "EP8 BF16 warp specialization requires SM103, found sm_",
            major, minor);
        TORCH_CHECK(
            sm_count == launch_config::NUM_BLOCKS,
            "EP8 BF16 warp specialization requires exactly ",
            launch_config::NUM_BLOCKS, " SMs, found ", sm_count);
        TORCH_CHECK(
            cooperative_launch != 0 && cluster_launch != 0,
            "EP8 BF16 warp specialization requires cooperative cluster "
            "launch");

        CUDACHECK(cudaFuncSetAttribute(
            kittens::py::global_kernel<
                launch_config, globals_fwd,
                dispatch_mlp_swiglu_combine_fwd_bf16_ep8_warp_specialized_kernel<
                    IS_CLAMPED>>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            DYNAMIC_SMEM));
        int active_blocks_per_sm = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            kittens::py::global_kernel<
                launch_config, globals_fwd,
                dispatch_mlp_swiglu_combine_fwd_bf16_ep8_warp_specialized_kernel<
                    IS_CLAMPED>>,
            launch_config::NUM_THREADS,
            DYNAMIC_SMEM));
        TORCH_CHECK(
            active_blocks_per_sm >= 1,
            "EP8 BF16 warp specialization cannot place one CTA per SM");
        int active_clusters = 0;
        CUDACHECK(cudaOccupancyMaxActiveClusters(
            &active_clusters,
            kittens::py::global_kernel<
                launch_config, globals_fwd,
                dispatch_mlp_swiglu_combine_fwd_bf16_ep8_warp_specialized_kernel<
                    IS_CLAMPED>>,
            &cuda_config));
        TORCH_CHECK(
            active_clusters >= launch_config::NUM_CLUSTERS,
            "EP8 BF16 warp specialization requires ",
            launch_config::NUM_CLUSTERS,
            " resident clusters, occupancy reports ", active_clusters);
        validated_device = device;
    }

    CUDACHECK(cudaLaunchKernelEx(
        &cuda_config,
        kittens::py::global_kernel<
            launch_config, globals_fwd,
            dispatch_mlp_swiglu_combine_fwd_bf16_ep8_warp_specialized_kernel<
                IS_CLAMPED>>,
        g));
}
