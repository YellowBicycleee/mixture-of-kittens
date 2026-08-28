from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "csrc" / "megakernel" / "union_x_routed_gate_up.cuh"
MEGA = ROOT / "csrc" / "megakernel" / "megakernel.cuh"
LEGACY = ROOT / "csrc" / "megakernel" / "fused_gate_up.cuh"
SOURCE = HEADER.read_text()
GATHER = (ROOT / "csrc" / "megakernel" / "union_x_gather.cuh").read_text()


def test_routed_bf16_only_signature_and_paired_n512_geometry() -> None:
    for token in (
        "template <bool IS_CLAMPED>",
        "expert_gate_up_swiglu_union_x_bf16_kernel",
        "const routed_bf16_gl &union_x",
        "const index_gl &route_to_union",
        "UNION_X_RGU_STAGES = 2",
        "UNION_X_RGU_A_WAIT_WARP = 5",
        "UNION_X_RGU_A_TMA_WARP = 6",
        "UNION_X_RGU_B_WARP = 7",
        "UNION_X_RGU_MMA_WARP = 4",
        "UNION_X_RGU_LOGICAL_N_TILES = 2",
        "UNION_X_RGU_PACKED_HIDDEN_N",
        "tt<float, config::MLP_Mb / 2, config::MLP_Nb> &paired_d_tt",
        "paired_b_smem",
        "static_assert(!USE_MXFP8)",
        "must not pass the legacy X-ring counter",
    ):
        assert token in SOURCE
    for forbidden in (
        "IS_SHARED",
        "routed_sc_gl",
        "mlp_fp8_tile",
        "quant_fp8_tile",
    ):
        assert forbidden not in SOURCE


def test_task_decode_matches_legacy_routed_ordering_seams() -> None:
    for token in (
        "global_minibatch_routed_first_row_block",
        "tokens_per_expert[{expert_idx}] / config::MLP_Mb",
        "gate_b_gmem.rows() / UNION_X_RGU_PACKED_HIDDEN_N",
        "get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>",
        "row_blocks, paired_col_blocks, task_idx",
        "global_first_row_block + swizzled.x",
        "task_group_idx * config::FUSED_GATE_UP_TASK_GROUP_SIZE",
        "macrobatch_idx * macrobatch_size",
        "+ tile_coord.x * config::MLP_Mb",
        "+ cta_rank * UNION_X_RGU_CTA_ROWS",
    ):
        assert token in SOURCE


def test_each_stage_gathers_a_once_and_consumes_two_adjacent_b_tiles() -> None:
    a_branch = SOURCE.index("global_warp == UNION_X_RGU_A_TMA_WARP")
    b_branch = SOURCE.index("global_warp == UNION_X_RGU_B_WARP", a_branch)
    a_section = SOURCE[a_branch:b_branch]
    assert a_section.count("union_x_issue_gather4_a_tile(") == 1

    mma_branch = SOURCE.index("global_warp == UNION_X_RGU_MMA_WARP", b_branch)
    b_section = SOURCE[b_branch:mma_branch]
    assert b_section.count("tma::cluster::load_async(") == 2
    assert "tile_coord.y * UNION_X_RGU_LOGICAL_N_TILES" in b_section
    assert "tile_coord.y * UNION_X_RGU_LOGICAL_N_TILES + 1" in b_section

    epilogue_branch = SOURCE.index("warpgroup::groupid() == 0", mma_branch)
    mma_section = SOURCE[mma_branch:epilogue_branch]
    assert mma_section.count("mm2_ABt(") == 2
    assert mma_section.count("mma2_ABt(") == 2
    assert "paired_d_tt" in mma_section
    # Only the second MMA releases A/B stage storage, after both consumers.
    first_mm = mma_section.index("mm2_ABt(")
    second_mm = mma_section.index("mm2_ABt(", first_mm + 1)
    assert "gemm_inputs_finished[input_stage]" not in mma_section[
        first_mm:second_mm
    ]
    assert "gemm_inputs_finished[input_stage]" in mma_section[second_mm:]


def test_epilogue_scratch_stays_above_next_generic_task_input_ring() -> None:
    for token in (
        "UNION_X_RGU_SCRATCH_OFFSET",
        "2 * FUSED_GATE_UP_LOAD_PIPE_DEPTH * sizeof(mlp_bf16_tile)",
        "<= UNION_X_RGU_SCRATCH_OFFSET",
        "smem_base_addr + UNION_X_RGU_SCRATCH_OFFSET",
        "Producer warps can accept the next CLC task",
    ):
        assert token in SOURCE
    assert "scratch_base_addr =\n        smem_base_addr\n        + sizeof(a_smem)" not in SOURCE


def test_k4096_pipeline_has_independent_a_and_b_readiness() -> None:
    for token in (
        "UNION_X_DISPATCH_HIDDEN / UNION_X_RGU_K",
        "union_x_load_route_ids4(",
        "union_x_issue_gather4_a_tile(",
        "a_tma_arrived[input_stage]",
        "warp::tma::cluster::arrive(",
        "tma::cluster::load_async(",
        "* UNION_X_RGU_LOGICAL_N_TILES",
        "* sizeof(mlp_bf16_tile)",
        "tma::cluster::wait(",
        "a_gather_bitfield",
        "mm2_ABt(",
        "mma2_ABt(",
        "detail::tcgen05::commit<config::CLUSTER_SIZE>",
        "ring_advance<UNION_X_RGU_STAGES>",
    ):
        assert token in SOURCE
    assert "sizeof(a_smem)" not in SOURCE[
        SOURCE.index("tma::expect_bytes("):
        SOURCE.index("tma::cluster::wait(")
    ]


def test_b_transaction_is_armed_before_both_ctas_can_load() -> None:
    mma_branch = SOURCE.index("global_warp == UNION_X_RGU_MMA_WARP")
    arm = SOURCE.index("tma::expect_bytes(", mma_branch)
    publish_local = SOURCE.index(
        "b_inputs_armed[input_stage], 0", arm
    )
    publish_remote = SOURCE.index(
        "b_inputs_armed[input_stage], 1", publish_local
    )
    producer_branch = SOURCE.index(
        "global_warp == UNION_X_RGU_B_WARP"
    )
    producer_wait = SOURCE.index(
        "wait(\n                b_inputs_armed[input_stage]",
        producer_branch,
    )
    b_load = SOURCE.index("tma::cluster::load_async(", producer_wait)
    assert arm < publish_local < publish_remote
    assert producer_wait < b_load
    assert "uint32_t &b_arm_bitfield" in SOURCE


def test_a_reuse_phase_is_handed_off_by_warp7() -> None:
    wait_branch = SOURCE.index("global_warp == UNION_X_RGU_A_WAIT_WARP")
    a_branch = SOURCE.index("global_warp == UNION_X_RGU_A_TMA_WARP", wait_branch)
    wait_section = SOURCE[wait_branch:a_branch]
    assert "a_tma_arrived[input_stage]" in wait_section
    assert "a_tma_bitfield" in wait_section
    assert "a_gather_arrived[input_stage], 0" in wait_section

    b_branch = SOURCE.index("global_warp == UNION_X_RGU_B_WARP", a_branch)
    a_section = SOURCE[a_branch:b_branch]
    assert "a_inputs_reusable[input_stage]" in a_section
    assert "a_reuse_bitfield" in a_section
    assert "union_x_issue_gather4_a_tile(" in a_section
    assert "gemm_inputs_finished[input_stage]" not in a_section

    mma_branch = SOURCE.index("global_warp == UNION_X_RGU_MMA_WARP", b_branch)
    b_section = SOURCE[b_branch:mma_branch]
    reuse_wait = b_section.index("gemm_inputs_finished[input_stage]")
    handoff = b_section.index("a_inputs_reusable[input_stage], cta_rank")
    b_load = b_section.index("tma::cluster::load_async(")
    assert reuse_wait < handoff < b_load


def test_route_ids_are_cached_once_and_padding_uses_tma_oob_fill() -> None:
    a_branch = SOURCE.index("global_warp == UNION_X_RGU_A_TMA_WARP")
    b_branch = SOURCE.index("global_warp == UNION_X_RGU_B_WARP", a_branch)
    section = SOURCE[a_branch:b_branch]
    route_ids = section.index("const int4 union_ids = union_x_load_route_ids4(")
    k_loop = section.index("for (int k_block = 0;", route_ids)
    gather = section.index("union_x_issue_gather4_a_tile(", k_loop)
    assert route_ids < k_loop < gather
    assert section.count("union_x_load_route_ids4(") == 1
    assert "make_cache_policy<cache_policy::EVICT_LAST>()" in section
    assert "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4" in GATHER
    assert "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE" in GATHER
    for forbidden in (
        "move<float4>::ldg(",
        "move<float4>::sts(",
        "fence.proxy.async.shared::cta",
    ):
        assert forbidden not in section


def test_dispatch_ready_acquire_is_bridged_to_every_async_gather_issuer() -> None:
    a_branch = SOURCE.index("global_warp == UNION_X_RGU_A_TMA_WARP")
    b_branch = SOURCE.index("global_warp == UNION_X_RGU_B_WARP", a_branch)
    section = SOURCE[a_branch:b_branch]
    ready_wait = section.index("barrier_wait(")
    warp_sync = section.index("__syncwarp();", ready_wait)
    proxy_fence = section.index("fence.proxy.async.global", warp_sync)
    k_loop = section.index("for (int k_block = 0;", proxy_fence)
    gather = section.index("union_x_issue_gather4_a_tile(", k_loop)
    assert ready_wait < warp_sync < proxy_fence < k_loop < gather
    assert section.count("fence.proxy.async.global") == 1
    # The fence is outside the elected-leader block because all 32 lanes issue
    # one gather4 transaction.
    assert section.rfind("if (warp::elect_leader())", 0, warp_sync) < warp_sync


def test_bf16_epilogue_preserves_context_swiglu_and_hidden_ready() -> None:
    for token in (
        "const bool save_context = macrobatch_idx == 0",
        "logical_n_offset < UNION_X_RGU_LOGICAL_N_TILES",
        "logical_n_offset == 0 ? d_tt : paired_d_tt",
        "tile_coord.y * UNION_X_RGU_LOGICAL_N_TILES + logical_n_offset",
        "__float22bfloat162_rn",
        "tma::store_async(\n                    gate_context_gmem",
        "tma::store_async(\n                    up_context_gmem",
        "base_ops::exp::op<float2>",
        "__floats2bfloat162_rn",
        "hidden_raw_smem",
        "tma::store_async_wait()",
        "barrier_arrive(\n                hidden_row_block_ready",
        "hidden_row_block_ready_base_index",
        "+ macrobatch_row_block_offset + tile_coord.x",
        "Preserve one publication for each old N128 hidden subtile",
    ):
        assert token in SOURCE


def test_new_header_is_parsed_without_calling_or_editing_legacy_function() -> None:
    assert '#include "union_x_routed_gate_up.cuh"' in MEGA.read_text()
    assert "expert_gate_up_swiglu_union_x_bf16_kernel" not in LEGACY.read_text()
    assert "dispatch_mlp_swiglu_combine_fwd_kernel" not in SOURCE
    assert "expert_grouped_gemm_kernel" not in SOURCE
