// Private EP8 BF16 Union-X FC1 K64 compile/runtime slice.
//
// This is deliberately not a Forward implementation.  One fixed cluster of
// two 256-thread CTAs computes a single M256 x [Gate128 | Up128] x K64 partial
// accumulator.  Producer warpgroup local warp 2 issues A TMA gather4, local
// warp 1 waits for that CTA-local transaction and publishes cluster readiness,
// local warp 3 loads only B with cluster TMA, and CTA0 local warp 0 issues one
// tcgen05 mm2_ABt.  Consumer warpgroup 0 writes the raw FP32 accumulator for a
// small numerical reference.  There is no CLC, SwiGLU, Down, or fallback.

static constexpr int UNION_X_FC1_K64_ROWS = 256;
static constexpr int UNION_X_FC1_K64_CTA_ROWS = 128;
static constexpr int UNION_X_FC1_K64_LOGICAL_N = 128;
static constexpr int UNION_X_FC1_K64_PACKED_N = 256;
static constexpr int UNION_X_FC1_K64_K = 64;
static constexpr int UNION_X_FC1_K64_A_WAIT_WARP = 5;
static constexpr int UNION_X_FC1_K64_A_TMA_WARP = 6;
static constexpr int UNION_X_FC1_K64_B_TMA_WARP = 7;
static constexpr int UNION_X_FC1_K64_MMA_WARP = 4;

static_assert(UNION_X_FC1_K64_CTA_ROWS == mlp_bf16_tile::rows);
static_assert(UNION_X_FC1_K64_K == mlp_bf16_tile::cols);
static_assert(UNION_X_FC1_K64_ROWS == config::MLP_Mb);
static_assert(UNION_X_FC1_K64_PACKED_N == config::MLP_Nb);
static_assert(UNION_X_FC1_K64_LOGICAL_N == config::SWIGLU_Nb);
static_assert(config::NUM_THREADS == 256);
static_assert(config::CLUSTER_SIZE == 2);

using union_x_fc1_k64_output_gl = gl<float, 1, 1, -1, -1>;

struct union_x_fc1_k64_globals {
    routed_bf16_gl union_x;             // (U_capacity, 4096)
    CUtensorMap union_x_gather4_tma;     // 2D [H, U], K64 x one-row box
    index_gl route_to_union;             // at least 256 rows; -1 is padding
    weight_bf16_gl gate_weight;          // (128, 4096)
    weight_bf16_gl up_weight;            // (128, 4096)
    union_x_fc1_k64_output_gl output;    // (256, 256), FP32 partial
};

struct union_x_fc1_k64_config {
    static constexpr int NUM_BLOCKS = 2;
    static constexpr int NUM_THREADS = 256;
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int MIN_BLOCKS_PER_SM = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        2 * sizeof(mlp_bf16_tile) + 1023;
};

static __device__ __forceinline__ void union_x_fc1_k64_kernel(
    const union_x_fc1_k64_globals &g
) {
    static_assert(NUM_DEVICES == 8);
    static_assert(!USE_MXFP8);

    const int global_warp = threadIdx.x / WARP_THREADS;
    const int cta_rank = cluster_ctarank();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr =
        (reinterpret_cast<uint64_t>(&__shm[0]) + 1023)
        & ~uint64_t(1023);
    auto &a_smem = *reinterpret_cast<mlp_bf16_tile *>(smem_base_addr);
    auto &b_smem = *reinterpret_cast<mlp_bf16_tile *>(
        smem_base_addr + sizeof(mlp_bf16_tile));

    __shared__ semaphore union_a_tma_arrived;
    __shared__ semaphore union_a_cluster_arrived;
    __shared__ semaphore b_tma_arrived;
    __shared__ semaphore ab_consumed;
    __shared__ semaphore acc_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(union_a_tma_arrived, 0, 1);
        init_semaphore(
            union_a_cluster_arrived, 0, config::CLUSTER_SIZE);
        init_semaphore(b_tma_arrived, 0, 1);
        init_semaphore(ab_consumed, 0, 1);
        init_semaphore(acc_arrived, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    auto d_tt = tm_alloc.template allocate<
        tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    everyone::tma::cluster::sync();

    if (global_warp == UNION_X_FC1_K64_A_TMA_WARP)
        union_x_prefetch_gather4_tma(g.union_x_gather4_tma);

    // Arm CTA0's B transaction barrier before either CTA can issue its TMA.
    if (cta_rank == 0
        && global_warp == UNION_X_FC1_K64_MMA_WARP
        && warp::elect_leader()) {
        tma::expect_bytes(
            b_tma_arrived,
            config::CLUSTER_SIZE * sizeof(mlp_bf16_tile));
    }
    everyone::tma::cluster::sync();

    if (global_warp == UNION_X_FC1_K64_A_WAIT_WARP
        && warp::elect_leader()) {
        wait(union_a_tma_arrived, 0);
        warp::tma::cluster::arrive(union_a_cluster_arrived, 0);
    } else if (global_warp == UNION_X_FC1_K64_A_TMA_WARP) {
        const int route_row_start =
            cta_rank * UNION_X_FC1_K64_CTA_ROWS;
        const int4 union_ids = union_x_load_route_ids4(
            g.route_to_union, route_row_start, g.union_x.rows());
        // Production revisits one route tile for every FC1 N tile; keep the
        // probe on the same cache policy as that reuse-sensitive path.
        const uint64_t cache_hint =
            make_cache_policy<cache_policy::EVICT_LAST>();
        union_x_issue_gather4_a_tile(
            g.union_x_gather4_tma,
            union_a_tma_arrived,
            a_smem,
            0,
            union_ids,
            cache_hint);
    } else if (
        global_warp == UNION_X_FC1_K64_B_TMA_WARP
        && warp::elect_leader()) {
        const weight_bf16_gl &weight =
            cta_rank == 0 ? g.gate_weight : g.up_weight;
        tma::cluster::load_async(
            b_smem,
            weight,
            {0, 0},
            b_tma_arrived,
            static_cast<uint16_t>(1 << cta_rank),
            0);
    } else if (
        cta_rank == 0
        && global_warp == UNION_X_FC1_K64_MMA_WARP
        && warp::elect_leader()) {
        tma::cluster::wait(union_a_cluster_arrived, 0);
        wait(b_tma_arrived, 0);
        tensor_after_thread_sync();
        mm2_ABt(d_tt, a_smem, b_smem, ab_consumed);
        detail::tcgen05::commit<config::CLUSTER_SIZE>(acc_arrived);
    } else if (warpgroup::groupid() == 0) {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(acc_arrived, 0);
        const int tile_row = epilogue_group::laneid();
        const int output_row =
            cta_rank * UNION_X_FC1_K64_CTA_ROWS + tile_row;

        #pragma unroll 1
        for (int block = 0; block < UNION_X_FC1_K64_PACKED_N / 32;
             ++block) {
            float2 values[16];
            asm volatile(R"(
                tcgen05.ld.sync.aligned.32x32b.x32.b32
                {%0, %1, %2, %3, %4, %5, %6, %7,
                 %8, %9, %10, %11, %12, %13, %14, %15,
                 %16, %17, %18, %19, %20, %21, %22, %23,
                 %24, %25, %26, %27, %28, %29, %30, %31}, [%32];
                )"
                : "=f"(values[0].x), "=f"(values[0].y),
                  "=f"(values[1].x), "=f"(values[1].y),
                  "=f"(values[2].x), "=f"(values[2].y),
                  "=f"(values[3].x), "=f"(values[3].y),
                  "=f"(values[4].x), "=f"(values[4].y),
                  "=f"(values[5].x), "=f"(values[5].y),
                  "=f"(values[6].x), "=f"(values[6].y),
                  "=f"(values[7].x), "=f"(values[7].y),
                  "=f"(values[8].x), "=f"(values[8].y),
                  "=f"(values[9].x), "=f"(values[9].y),
                  "=f"(values[10].x), "=f"(values[10].y),
                  "=f"(values[11].x), "=f"(values[11].y),
                  "=f"(values[12].x), "=f"(values[12].y),
                  "=f"(values[13].x), "=f"(values[13].y),
                  "=f"(values[14].x), "=f"(values[14].y),
                  "=f"(values[15].x), "=f"(values[15].y)
                : "r"(d_tt.addr
                      + ((warpgroup::warpid() * 32) << 16)
                      + block * 32));
            tensor_load_wait();

            #pragma unroll
            for (int pair = 0; pair < 16; ++pair) {
                const size_t output_offset =
                    static_cast<size_t>(output_row)
                        * UNION_X_FC1_K64_PACKED_N
                    + block * 32 + pair * 2;
                *reinterpret_cast<float2 *>(
                    &g.output.raw_ptr[output_offset]) = values[pair];
            }
        }
        tensor_before_thread_sync();
        epilogue_group::sync(1);
    }

    // Keep the managed TMEM allocator alive until both CTA epilogues have
    // completed all raw output stores; its destructor performs deallocation.
    everyone::tma::cluster::sync();
}

static __host__ __forceinline__ at::Tensor union_x_fc1_k64(
    const at::Tensor &union_x,
    const at::Tensor &route_to_union,
    const at::Tensor &gate_weight,
    const at::Tensor &up_weight
) {
    TORCH_CHECK(NUM_DEVICES == 8, "Union-X FC1 K64 requires EP8");
    TORCH_CHECK(!USE_MXFP8, "Union-X FC1 K64 requires BF16");
    CHECK_INPUT(union_x);
    CHECK_INPUT(route_to_union);
    CHECK_INPUT(gate_weight);
    CHECK_INPUT(up_weight);
    TORCH_CHECK(union_x.scalar_type() == at::kBFloat16
                && union_x.dim() == 2
                && union_x.size(0) > 0
                && union_x.size(0) % UNION_X_FC1_K64_CTA_ROWS == 0
                && union_x.size(1) == UNION_X_DISPATCH_HIDDEN,
                "union_x must be BF16 [U_capacity, 4096] with U_capacity divisible by 128");
    TORCH_CHECK(route_to_union.scalar_type() == at::kInt
                && route_to_union.dim() == 1
                && route_to_union.size(0) == UNION_X_FC1_K64_ROWS,
                "route_to_union must be int32 [256]");
    TORCH_CHECK(gate_weight.scalar_type() == at::kBFloat16
                && gate_weight.dim() == 2
                && gate_weight.size(0) == UNION_X_FC1_K64_LOGICAL_N
                && gate_weight.size(1) == UNION_X_DISPATCH_HIDDEN,
                "gate_weight must be BF16 [128, 4096]");
    TORCH_CHECK(up_weight.sizes() == gate_weight.sizes()
                && up_weight.scalar_type() == at::kBFloat16,
                "up_weight must match gate_weight");
    TORCH_CHECK(route_to_union.device() == union_x.device()
                && gate_weight.device() == union_x.device()
                && up_weight.device() == union_x.device(),
                "all Union-X FC1 K64 tensors must share one CUDA device");

    at::Tensor output = at::empty(
        {UNION_X_FC1_K64_ROWS, UNION_X_FC1_K64_PACKED_N},
        union_x.options().dtype(at::kFloat));
    union_x_fc1_k64_globals g {
        .union_x = kittens::py::tensor_to_gl<routed_bf16_gl>(union_x),
        .union_x_gather4_tma = union_x_make_gather4_tma(union_x),
        .route_to_union =
            kittens::py::tensor_to_gl<index_gl>(route_to_union),
        .gate_weight =
            kittens::py::tensor_to_gl<weight_bf16_gl>(gate_weight),
        .up_weight = kittens::py::tensor_to_gl<weight_bf16_gl>(up_weight),
        .output = kittens::py::tensor_to_gl<union_x_fc1_k64_output_gl>(output),
    };
    kittens::py::launch_kernel<
        union_x_fc1_k64_config,
        union_x_fc1_k64_globals,
        union_x_fc1_k64_kernel>(g);
    return output;
}
