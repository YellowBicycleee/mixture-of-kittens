#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <tuple>

using namespace kittens;

// An independent extension of the legacy routed-token scheduler.  It preserves
// the four legacy outputs and their ordering, and additionally assigns every
// valid route a compact union id for (peer_rank, original_token_idx).
namespace union_scheduler {

struct config {
    static constexpr int EXPERT_PADDING = 256;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1024;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals {
    using topk_gl = gl<int, 1, -1, -1, -1>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    topk_gl topk;                        // (world_size, num_local_tokens, topk)
    index_gl schedule_peer_rank;         // (schedule_capacity,), initialized -1
    index_gl schedule_peer_token_idx;    // (schedule_capacity,), token * topk + slot
    index_gl num_tokens;                 // (1,), initialized zero
    index_gl tokens_per_expert;          // (num_local_experts,)
    index_gl tokens_per_expert_and_peer; // (num_local_experts * world_size,), initialized zero

    index_gl dense_present;              // (world_size * num_local_tokens,), initialized zero
    index_gl dense_to_union;             // (world_size * num_local_tokens,)
    index_gl route_to_union;             // (schedule_capacity,), initialized -1
    index_gl num_union;                  // (1,)

    int rank;
};

// Count routed rows exactly as the legacy scheduler does.  For every valid
// local route, also mark its dense (peer, original token) key as present.
static __device__ __forceinline__ void count_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_global_routes = world_size * rank_stride;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;
    const int last_expert = first_expert + num_local_experts;

    extern __shared__ int tokens_per_expert_and_peer[];
    for (int i = threadIdx.x; i < G.tokens_per_expert_and_peer.cols(); i += blockDim.x)
        tokens_per_expert_and_peer[i] = 0;
    __syncthreads();

    const int grid_stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_global_routes;
         idx += grid_stride) {
        const int peer_rank = idx / rank_stride;
        const int peer_token_idx = idx - peer_rank * rank_stride;
        const int original_token_idx = peer_token_idx / topk;
        const int expert_idx = G.topk[{peer_rank, original_token_idx, peer_token_idx % topk}];
        if (expert_idx >= first_expert && expert_idx < last_expert) {
            atomicAdd(
                &tokens_per_expert_and_peer[
                    (expert_idx - first_expert) * world_size + peer_rank],
                1);
            atomicExch(
                &G.dense_present[{peer_rank * num_local_tokens + original_token_idx}],
                1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < G.tokens_per_expert_and_peer.cols(); i += blockDim.x)
        if (tokens_per_expert_and_peer[i] != 0)
            atomicAdd(&G.tokens_per_expert_and_peer[{i}], tokens_per_expert_and_peer[i]);
}

static __device__ __forceinline__ void pad_kernel(const globals &G) {
    const int local_expert = blockIdx.x;
    const int world_size = G.topk.depth();
    int num_tokens = 0;
    for (int peer_rank = 0; peer_rank < world_size; ++peer_rank)
        num_tokens += G.tokens_per_expert_and_peer[{local_expert * world_size + peer_rank}];
    const int padded_num_tokens =
        (num_tokens + config::EXPERT_PADDING - 1) / config::EXPERT_PADDING
        * config::EXPERT_PADDING;
    G.tokens_per_expert[{local_expert}] = padded_num_tokens;
    atomicAdd(&G.num_tokens[{0}], padded_num_tokens);
}

static __device__ __forceinline__ void finalize_num_union_kernel(const globals &G) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int last = G.dense_present.cols() - 1;
        G.num_union[{0}] = G.dense_to_union[{last}] + G.dense_present[{last}];
    }
}

// Preserve the legacy expert-major, peer-interleaved route ordering.  The only
// additional write is the compact union id of the route's dense key.
static __device__ __forceinline__ void schedule_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;

    if (G.num_tokens[{0}] > G.schedule_peer_rank.cols()) asm volatile("{trap;}");

    extern __shared__ int tokens_per_peer_rank[];
    __shared__ int cumulative_tokens_from_peer_rank[config::NUM_WARPS];

    for (int idx = blockIdx.x; idx < num_local_experts * world_size; idx += gridDim.x) {
        const int local_expert = idx / world_size;
        const int peer_rank = idx % world_size;

        int expert_base = 0;
        for (int expert_idx = 0; expert_idx < local_expert; ++expert_idx)
            expert_base += G.tokens_per_expert[{expert_idx}];

        for (int rank = threadIdx.x; rank < world_size; rank += blockDim.x)
            tokens_per_peer_rank[rank] =
                G.tokens_per_expert_and_peer[{local_expert * world_size + rank}];
        __syncthreads();

        int tokens_from_peer_rank = 0;
        for (int peer_token_idx = threadIdx.x;
             peer_token_idx < rank_stride;
             peer_token_idx += blockDim.x) {
            const int expert_idx = G.topk[
                {peer_rank, peer_token_idx / topk, peer_token_idx % topk}];
            tokens_from_peer_rank +=
                (expert_idx - first_expert == local_expert) ? 1 : 0;
        }

        int inclusive = tokens_from_peer_rank;
        for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
            const int n = __shfl_up_sync(0xffffffff, inclusive, offset);
            if (warp::laneid() >= offset) inclusive += n;
        }
        if (warp::laneid() == WARP_THREADS - 1)
            cumulative_tokens_from_peer_rank[warpid()] = inclusive;
        __syncthreads();

        if (warpid() == 0) {
            int warp_total = warp::laneid() < config::NUM_WARPS
                ? cumulative_tokens_from_peer_rank[warp::laneid()]
                : 0;
            for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
                const int n = __shfl_up_sync(0xffffffff, warp_total, offset);
                if (warp::laneid() >= offset) warp_total += n;
            }
            if (warp::laneid() < config::NUM_WARPS)
                cumulative_tokens_from_peer_rank[warp::laneid()] = warp_total;
        }
        __syncthreads();

        int j = (warpid() == 0 ? 0 : cumulative_tokens_from_peer_rank[warpid() - 1])
            + inclusive - tokens_from_peer_rank;

        for (int peer_token_idx = threadIdx.x;
             peer_token_idx < rank_stride;
             peer_token_idx += blockDim.x) {
            const int original_token_idx = peer_token_idx / topk;
            const int expert_idx =
                G.topk[{peer_rank, original_token_idx, peer_token_idx % topk}];
            if (expert_idx - first_expert == local_expert) {
                int dst_token_idx = expert_base;
                for (int rank = 0; rank < world_size; ++rank) {
                    const int num_tokens = tokens_per_peer_rank[rank];
                    dst_token_idx += min(num_tokens, j);
                    dst_token_idx += (rank < peer_rank && num_tokens > j) ? 1 : 0;
                }
                const int dense_key = peer_rank * num_local_tokens + original_token_idx;
                G.schedule_peer_rank[{dst_token_idx}] = peer_rank;
                G.schedule_peer_token_idx[{dst_token_idx}] = peer_token_idx;
                G.route_to_union[{dst_token_idx}] = G.dense_to_union[{dense_key}];
                ++j;
            }
        }
        __syncthreads();
    }
}

static __host__ std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
schedule_entrypoint(
    const at::Tensor &topk_all,
    const int num_local_experts,
    const int schedule_capacity,
    const int rank
) {
    const int world_size = static_cast<int>(topk_all.size(0));
    const int num_local_tokens = static_cast<int>(topk_all.size(1));
    const int num_dense_keys = world_size * num_local_tokens;

    at::Tensor schedule_peer_rank =
        at::empty({schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor schedule_peer_token_idx =
        at::empty({schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor num_tokens = at::zeros({1}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert =
        at::empty({num_local_experts}, topk_all.options().dtype(at::kInt));
    at::Tensor route_to_union =
        at::empty({schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor num_union = at::empty({1}, topk_all.options().dtype(at::kInt));

    at::Tensor tokens_per_expert_and_peer = at::zeros(
        {num_local_experts * world_size}, topk_all.options().dtype(at::kInt));
    at::Tensor dense_present =
        at::zeros({num_dense_keys}, topk_all.options().dtype(at::kInt));
    at::Tensor dense_to_union =
        at::empty({num_dense_keys}, topk_all.options().dtype(at::kInt));

    schedule_peer_rank.fill_(-1);
    route_to_union.fill_(-1);

    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_all),
        .schedule_peer_rank =
            kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx =
            kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<globals::index_gl>(num_tokens),
        .tokens_per_expert =
            kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .tokens_per_expert_and_peer =
            kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert_and_peer),
        .dense_present =
            kittens::py::tensor_to_gl<globals::index_gl>(dense_present),
        .dense_to_union =
            kittens::py::tensor_to_gl<globals::index_gl>(dense_to_union),
        .route_to_union =
            kittens::py::tensor_to_gl<globals::index_gl>(route_to_union),
        .num_union = kittens::py::tensor_to_gl<globals::index_gl>(num_union),
        .rank = rank,
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    kittens::py::global_kernel<config, globals, union_scheduler::count_kernel>
        <<<(G.topk.numel() + config::NUM_THREADS - 1) / config::NUM_THREADS,
           config::NUM_THREADS,
           num_local_experts * world_size * sizeof(int),
           stream>>>(G);
    kittens::py::global_kernel<config, globals, union_scheduler::pad_kernel>
        <<<num_local_experts, 1, 0, stream>>>(G);

    std::size_t scan_temp_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_temp_bytes,
        dense_present.data_ptr<int>(),
        dense_to_union.data_ptr<int>(),
        num_dense_keys,
        stream.stream()));
    at::Tensor scan_temp = at::empty(
        {static_cast<int64_t>(scan_temp_bytes)},
        topk_all.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_temp.data_ptr(),
        scan_temp_bytes,
        dense_present.data_ptr<int>(),
        dense_to_union.data_ptr<int>(),
        num_dense_keys,
        stream.stream()));

    kittens::py::global_kernel<config, globals, union_scheduler::finalize_num_union_kernel>
        <<<1, 1, 0, stream>>>(G);
    kittens::py::global_kernel<config, globals, union_scheduler::schedule_kernel>
        <<<num_local_experts * world_size,
           config::NUM_THREADS,
           world_size * sizeof(int),
           stream>>>(G);

    return {
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
        route_to_union,
        num_union,
    };
}

} // namespace union_scheduler
