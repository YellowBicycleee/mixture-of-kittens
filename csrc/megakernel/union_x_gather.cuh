// SM100/SM103 BF16 TMA gather4 support for the independent Union-X path.
//
// A compute CTA consumes 128 arbitrary Union-X rows at a time.  Each lane owns
// four consecutive route rows and keeps their four union ids in registers for
// the complete FC1 K loop.  One lane-level gather4 transaction moves those
// four K64 rows directly into the existing 128B-swizzled mlp_bf16_tile.  All
// 32 lanes therefore fill one 128x64 A tile with one warp instruction and one
// 16 KiB transaction barrier phase.  A union id of -1 is intentionally passed
// as an out-of-bounds tensor coordinate; the descriptor's normal OOB behavior
// zero-fills that destination row.

static constexpr int UNION_X_GATHER_ROWS = 128;
static constexpr int UNION_X_GATHER_K_COLS = 64;
static constexpr int UNION_X_GATHER_ROWS_PER_TMA = 4;
static constexpr int UNION_X_GATHER_TMA_ISSUERS = WARP_THREADS;
static constexpr int UNION_X_GATHER_ROW_BYTES =
    UNION_X_GATHER_K_COLS * sizeof(bf16);
static constexpr int UNION_X_GATHER_BYTES =
    UNION_X_GATHER_ROWS * UNION_X_GATHER_ROW_BYTES;

static_assert(UNION_X_GATHER_ROWS == config::MLP_Mb / config::CLUSTER_SIZE);
static_assert(UNION_X_GATHER_K_COLS == config::MLP_BF16_Kb);
static_assert(UNION_X_GATHER_ROWS_PER_TMA * UNION_X_GATHER_TMA_ISSUERS
              == UNION_X_GATHER_ROWS);
static_assert(UNION_X_GATHER_ROW_BYTES == 128);
static_assert(UNION_X_GATHER_BYTES == 16 * 1024);
static_assert(UNION_X_GATHER_BYTES == sizeof(mlp_bf16_tile));
static_assert(mlp_bf16_tile::rows == UNION_X_GATHER_ROWS);
static_assert(mlp_bf16_tile::cols == UNION_X_GATHER_K_COLS);
static_assert(mlp_bf16_tile::swizzle);
static_assert(mlp_bf16_tile::swizzle_bytes == 128);

// TK's normal swizzled-tile descriptor is five-dimensional.  TMA gather4
// requires a true two-dimensional tensor map whose bounding-box second
// dimension is exactly one.  Keep this descriptor private to the Union-X
// operator instead of changing routed_bf16_gl or ThunderKittens.
static __host__ __forceinline__ CUtensorMap union_x_make_gather4_tma(
    const at::Tensor &union_x
) {
    TORCH_CHECK(
        union_x.is_cuda() && union_x.is_contiguous()
            && union_x.scalar_type() == at::kBFloat16
            && union_x.dim() == 2 && union_x.size(0) > 0
            && union_x.size(1) >= UNION_X_GATHER_K_COLS,
        "Union-X gather4 requires contiguous CUDA BF16 [U, H]");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(union_x.data_ptr()) % 16 == 0,
        "Union-X gather4 base pointer must be 16-byte aligned");
    TORCH_CHECK(
        (union_x.size(1) * static_cast<int64_t>(sizeof(bf16))) % 16 == 0,
        "Union-X gather4 row stride must be 16-byte aligned");

    CUtensorMap tensor_map{};
    const cuuint64_t gmem_dims[2] = {
        static_cast<cuuint64_t>(union_x.size(1)),
        static_cast<cuuint64_t>(union_x.size(0)),
    };
    const cuuint64_t gmem_strides[1] = {
        static_cast<cuuint64_t>(
            union_x.size(1) * static_cast<int64_t>(sizeof(bf16))),
    };
    const cuuint32_t box_dims[2] = {
        static_cast<cuuint32_t>(UNION_X_GATHER_K_COLS),
        1,
    };
    const cuuint32_t elem_strides[2] = {1, 1};

    const CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        union_x.data_ptr(),
        gmem_dims,
        gmem_strides,
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    const char *error_string = nullptr;
    if (result != CUDA_SUCCESS)
        cuGetErrorString(result, &error_string);
    TORCH_CHECK(
        result == CUDA_SUCCESS,
        "cuTensorMapEncodeTiled failed for Union-X gather4: ",
        error_string == nullptr ? "unknown CUDA driver error" : error_string);
    return tensor_map;
}

static __device__ __forceinline__ void union_x_prefetch_gather4_tma(
    const CUtensorMap &tensor_map
) {
    if (warp::elect_leader()) {
        asm volatile(
            "{prefetch.tensormap [%0];}"
            :: "l"(reinterpret_cast<uint64_t>(&tensor_map))
            : "memory");
    }
}

// One aligned 128-bit load per lane fetches all route metadata that lane needs
// for every K64 stage of the current raw FC1 task.
static __device__ __forceinline__ int4 union_x_load_route_ids4(
    const index_gl &route_to_union,
    const int route_row_start,
    const int union_rows
) {
    const int route_row =
        route_row_start + warp::laneid() * UNION_X_GATHER_ROWS_PER_TMA;
    if (route_row_start < 0
        || route_row + UNION_X_GATHER_ROWS_PER_TMA > route_to_union.cols())
        asm volatile("{trap;}");

    const int *source = &route_to_union.raw_ptr[route_row];
    if ((reinterpret_cast<uintptr_t>(source) & 15u) != 0)
        asm volatile("{trap;}");

    int4 union_ids;
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(union_ids.x), "=r"(union_ids.y),
          "=r"(union_ids.z), "=r"(union_ids.w)
        : "l"(source)
        : "memory");
    if (union_ids.x < -1 || union_ids.x >= union_rows
        || union_ids.y < -1 || union_ids.y >= union_rows
        || union_ids.z < -1 || union_ids.z >= union_rows
        || union_ids.w < -1 || union_ids.w >= union_rows)
        asm volatile("{trap;}");
    return union_ids;
}

// All 32 lanes must call this function convergently.  Lane 0 first arms the
// local transaction barrier for exactly one 16 KiB A tile; after the warp sync,
// every lane issues one independent four-row gather into a disjoint 512-byte
// region.  The 128B tensor-map swizzle and the destination's 512-byte lane
// offset reproduce mlp_bf16_tile's absolute-address swizzle pattern.
static __device__ __forceinline__ void union_x_issue_gather4_a_tile(
    const CUtensorMap &tensor_map,
    semaphore &tma_arrived,
    mlp_bf16_tile &a_smem,
    const int k_block,
    const int4 &union_ids,
    const uint64_t cache_hint
) {
    if (k_block < 0)
        asm volatile("{trap;}");
    if (warp::elect_leader())
        tma::expect_bytes(tma_arrived, UNION_X_GATHER_BYTES);
    __syncwarp();

    const uint32_t a_smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&a_smem));
    const uint32_t lane_dst = a_smem_addr
        + warp::laneid() * UNION_X_GATHER_ROWS_PER_TMA
            * UNION_X_GATHER_ROW_BYTES;
    const uint32_t mbarrier_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&tma_arrived));
    const int source_col = k_block * UNION_X_GATHER_K_COLS;

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
        "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(lane_dst),
          "l"(reinterpret_cast<uint64_t>(&tensor_map)),
          "r"(source_col),
          "r"(union_ids.x), "r"(union_ids.y),
          "r"(union_ids.z), "r"(union_ids.w),
          "r"(mbarrier_addr), "l"(cache_hint)
        : "memory");
}
