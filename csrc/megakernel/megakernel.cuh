#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "../mxfp8.cuh"
#include "../utils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

enum class RoutedPrecision {
    BF16,
    MXFP8,
};

template <
    int NUM_DEVICES,
    RoutedPrecision ROUTED_PRECISION = RoutedPrecision::MXFP8,
    int FWD_CLC_PIPE_DEPTH = 1,
    int FWD_GATE_GROUP_SIZE = 1,
    int FWD_DOWN_GROUP_SIZE =
        (NUM_DEVICES == 8 && ROUTED_PRECISION == RoutedPrecision::MXFP8 ? 2 : 1)>
struct dispatch_mlp_swiglu_combiner {

static constexpr bool USE_MXFP8 = ROUTED_PRECISION == RoutedPrecision::MXFP8;

static_assert(
    (FWD_CLC_PIPE_DEPTH == 1 &&
     FWD_GATE_GROUP_SIZE == 1 &&
     FWD_DOWN_GROUP_SIZE == (NUM_DEVICES == 8 && USE_MXFP8 ? 2 : 1)) ||
    (NUM_DEVICES == 8 && !USE_MXFP8 &&
     FWD_CLC_PIPE_DEPTH == 2 &&
     (FWD_GATE_GROUP_SIZE == 1 || FWD_GATE_GROUP_SIZE == 2) &&
     (FWD_DOWN_GROUP_SIZE == 1 ||
      FWD_DOWN_GROUP_SIZE == 2 ||
      FWD_DOWN_GROUP_SIZE == 4)),
    "unsupported MoK forward CLC/group configuration");

#include "types.cuh"

#include "utils.cuh"
#include "dispatch_combine.cuh"
#include "swiglu.cuh"
#include "grouped_gemm.cuh"
#include "fused_gate_up.cuh"

#include "forward.cuh"
#include "backward.cuh"
#include "recompute_forward_context.cuh"

}; // struct dispatch_mlp_swiglu_combiner
