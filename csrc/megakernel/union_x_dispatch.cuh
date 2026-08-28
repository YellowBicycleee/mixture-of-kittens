// Private production-aligned EP8 BF16 Union-X dispatch helper.
//
// The generic combiner template includes this private helper, but no legacy
// entrypoint calls it.  A future parallel Union-X forward instantiates it only
// for EP8 BF16.  One work item is the
// legacy dispatch shape: 128 route rows x one 512-BF16 hidden slice.  All CTA
// threads call this helper; the first 128 threads own route rows and every
// thread participates in the CTA barriers and the shared transaction barrier.

static constexpr int UNION_X_DISPATCH_EP_SIZE = 8;
static constexpr int UNION_X_DISPATCH_HIDDEN = 4096;
static constexpr int UNION_X_DISPATCH_ROWS = 128;
static constexpr int UNION_X_DISPATCH_SLICE_COLS = 512;
static constexpr int UNION_X_DISPATCH_SLICE_BYTES =
    UNION_X_DISPATCH_SLICE_COLS * sizeof(bf16);
static constexpr int UNION_X_DISPATCH_HIDDEN_SLICES =
    (UNION_X_DISPATCH_HIDDEN + UNION_X_DISPATCH_SLICE_COLS - 1)
    / UNION_X_DISPATCH_SLICE_COLS;

static constexpr uint32_t UNION_X_DISPATCH_EMPTY = 0;
static constexpr uint32_t UNION_X_DISPATCH_LOADING = 1;
static constexpr uint32_t UNION_X_DISPATCH_FULL = 2;

static_assert(UNION_X_DISPATCH_ROWS == config::DISPATCH_Mb);
static_assert(UNION_X_DISPATCH_SLICE_COLS == config::DISPATCH_Nb);
static_assert(UNION_X_DISPATCH_SLICE_BYTES == 1024);
static_assert(UNION_X_DISPATCH_HIDDEN_SLICES == 8);
static_assert(UNION_X_DISPATCH_SLICE_BYTES % 16 == 0);

static __device__ __forceinline__ uint32_t *union_x_dispatch_state_ptr(
    const index_gl &union_state,
    const int union_idx,
    const int hidden_slice
) {
    const size_t state_idx =
        static_cast<size_t>(union_idx) * UNION_X_DISPATCH_HIDDEN_SLICES
        + hidden_slice;
    return reinterpret_cast<uint32_t *>(union_state.raw_ptr + state_idx);
}

static __device__ __forceinline__ void union_x_dispatch_publish_full(
    uint32_t *state
) {
    const uint32_t full = UNION_X_DISPATCH_FULL;
    asm volatile(
        "{st.release.gpu.global.u32 [%0], %1;}"
        :: "l"(state), "r"(full) : "memory");
}

static __device__ __forceinline__ void union_x_dispatch_wait_full(
    const uint32_t *state
) {
    uint32_t observed;
    while (true) {
        asm volatile(
            "{ld.relaxed.gpu.global.u32 %0, [%1];}"
            : "=r"(observed) : "l"(state) : "memory");
        if (observed == UNION_X_DISPATCH_FULL)
            break;
        if (observed != UNION_X_DISPATCH_LOADING)
            asm volatile("{trap;}");
        __nanosleep(16);
    }
    asm volatile("{fence.acquire.gpu;}" ::: "memory");
}

// Resolve one legacy-shaped route-block x hidden-slice task.  Every valid row
// first claims its (union, slice) state.  Only EMPTY->LOADING winners contribute
// TMA bytes, store Union-X, and release FULL.  The first CTA sync guarantees all
// local winners have published before any loser can spin; remote winners never
// wait on this task and therefore guarantee progress.  Padding is resolved
// without touching Union-X or union_state.  Thread 0 publishes exactly one
// coarse arrival after all 128 rows are resolved.
static __device__ __forceinline__ void union_x_dispatch_task(
    const activation_bf16_pgl &peer_buf,
    const routed_bf16_gl &union_x,
    const index_gl &union_state,
    const index_gl &route_to_union,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl &transfer_done,
    semaphore &inputs_arrived,
    uint32_t &bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const int topk,
    const uint64_t smem_base_addr
) {
    static_assert(NUM_DEVICES == UNION_X_DISPATCH_EP_SIZE);
    static_assert(!USE_MXFP8);
    static_assert(config::NUM_THREADS >= UNION_X_DISPATCH_ROWS);

    auto &token_chunks = *reinterpret_cast<
        bf16 (*)[UNION_X_DISPATCH_ROWS][UNION_X_DISPATCH_SLICE_COLS]>(
            smem_base_addr);
    const int tid = threadIdx.x;
    const bool is_worker = tid < UNION_X_DISPATCH_ROWS;
    const int macro_offset = macrobatch_idx * macrobatch_size;
    const int macro_rows = min(macrobatch_size, num_tokens - macro_offset);
    const int num_tasks =
        (macro_rows / UNION_X_DISPATCH_ROWS)
        * UNION_X_DISPATCH_HIDDEN_SLICES;
    if (task_idx >= num_tasks)
        return;

    const int row_tile = task_idx / UNION_X_DISPATCH_HIDDEN_SLICES;
    const int hidden_slice =
        task_idx - row_tile * UNION_X_DISPATCH_HIDDEN_SLICES;
    const int row_start = row_tile * UNION_X_DISPATCH_ROWS;
    const int global_row = macro_offset + row_start + tid;
    const int col = hidden_slice * UNION_X_DISPATCH_SLICE_COLS;
    const int chunk_cols = min(
        UNION_X_DISPATCH_SLICE_COLS, UNION_X_DISPATCH_HIDDEN - col);
    const uint32_t chunk_bytes = chunk_cols * sizeof(bf16);

    int peer_rank = -1;
    int peer_token_idx = 0;
    int union_idx = -1;
    uint32_t observed = UNION_X_DISPATCH_FULL;
    if (is_worker) {
        peer_rank = schedule_peer_rank[{global_row}];
        union_idx = route_to_union[{global_row}];
        if (peer_rank >= 0) {
            if (peer_rank >= UNION_X_DISPATCH_EP_SIZE || union_idx < 0
                || union_idx >= union_x.rows()
                || union_x.cols() != UNION_X_DISPATCH_HIDDEN
                || topk <= 0)
                asm volatile("{trap;}");
            const size_t state_idx =
                static_cast<size_t>(union_idx)
                    * UNION_X_DISPATCH_HIDDEN_SLICES
                + hidden_slice;
            if (state_idx >= static_cast<size_t>(union_state.cols()))
                asm volatile("{trap;}");
            peer_token_idx = schedule_peer_token_idx[{global_row}];
            if (peer_token_idx < 0)
                asm volatile("{trap;}");
            observed = atomicCAS(
                union_x_dispatch_state_ptr(
                    union_state, union_idx, hidden_slice),
                UNION_X_DISPATCH_EMPTY,
                UNION_X_DISPATCH_LOADING);
            if (observed != UNION_X_DISPATCH_EMPTY
                && observed != UNION_X_DISPATCH_LOADING
                && observed != UNION_X_DISPATCH_FULL)
                asm volatile("{trap;}");
        } else if (union_idx != -1) {
            asm volatile("{trap;}");
        }
    }

    const bool is_winner =
        is_worker && peer_rank >= 0
        && observed == UNION_X_DISPATCH_EMPTY;
    const int num_winners = __syncthreads_count(is_winner);
    if (tid == 0 && num_winners > 0)
        tma::expect_bytes(inputs_arrived, num_winners * chunk_bytes);
    __syncthreads();

    if (is_winner) {
        const int source_token = peer_token_idx / topk;
        tma::load_async(
            token_chunks[tid],
            &peer_buf[peer_rank][
                static_cast<size_t>(source_token)
                    * UNION_X_DISPATCH_HIDDEN + col],
            chunk_bytes,
            inputs_arrived);
    }
    if (num_winners > 0) {
        wait(inputs_arrived, get_phasebit<0>(bitfield, 0));
        update_phasebit<0>(bitfield, 0);
    }

    if (is_winner) {
        tma::store_async(
            &union_x.raw_ptr[
                static_cast<size_t>(union_idx)
                    * UNION_X_DISPATCH_HIDDEN + col],
            token_chunks[tid],
            chunk_bytes);
        // Bulk groups are per-thread.  Every winning worker must wait for its
        // own store before publishing the corresponding state as FULL.
        tma::store_async_wait();
        // Bridge the async-proxy Union-X payload to the generic-proxy release
        // store which publishes FULL.
        asm volatile("{fence.proxy.async.global;}" ::: "memory");
        union_x_dispatch_publish_full(
            union_x_dispatch_state_ptr(
                union_state, union_idx, hidden_slice));
    }

    // All local winners publish before any local loser can spin.  This avoids
    // a divergent warp waiting on a winner lane from the same CTA.
    __syncthreads();
    if (is_worker && peer_rank >= 0 && !is_winner) {
        union_x_dispatch_wait_full(
            union_x_dispatch_state_ptr(
                union_state, union_idx, hidden_slice));
    }
    __syncthreads();

    if (tid == 0) {
        const int global_minibatch =
            (macro_offset + row_start) / minibatch_size;
        barrier_arrive(transfer_done, global_minibatch);
    }
}
