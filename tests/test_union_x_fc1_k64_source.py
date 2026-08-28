from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "csrc" / "megakernel" / "union_x_fused_gate_up.cuh"
MEGA = ROOT / "csrc" / "megakernel" / "megakernel.cuh"
ENTRYPOINTS = ROOT / "csrc" / "megakernel" / "entrypoints.cuh"
BINDINGS = ROOT / "csrc" / "bindings.cu"
OPS = ROOT / "mok" / "union_ops.py"
RUNTIME_TEST = ROOT / "tests" / "test_union_x_fc1_k64.py"
SOURCE = HEADER.read_text()
GATHER = (ROOT / "csrc" / "megakernel" / "union_x_gather.cuh").read_text()


def test_fixed_private_k64_geometry_and_roles() -> None:
    for token in (
        "UNION_X_FC1_K64_ROWS = 256",
        "UNION_X_FC1_K64_CTA_ROWS = 128",
        "UNION_X_FC1_K64_LOGICAL_N = 128",
        "UNION_X_FC1_K64_PACKED_N = 256",
        "UNION_X_FC1_K64_K = 64",
        "UNION_X_FC1_K64_A_WAIT_WARP = 5",
        "UNION_X_FC1_K64_A_TMA_WARP = 6",
        "UNION_X_FC1_K64_B_TMA_WARP = 7",
        "UNION_X_FC1_K64_MMA_WARP = 4",
        "NUM_BLOCKS = 2",
        "CLUSTER_SIZE = 2",
    ):
        assert token in SOURCE


def test_probe_uses_shared_gather4_descriptor_and_lane_mapping() -> None:
    for token in (
        "CUtensorMap union_x_gather4_tma",
        "union_x_make_gather4_tma(union_x)",
        "union_x_prefetch_gather4_tma(g.union_x_gather4_tma)",
        "union_x_load_route_ids4(",
        "make_cache_policy<cache_policy::EVICT_LAST>()",
        "union_x_issue_gather4_a_tile(",
    ):
        assert token in SOURCE
    assert "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4" in GATHER
    assert "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE" in GATHER
    for forbidden in ("move<float4>::ldg(", "move<float4>::sts("):
        assert forbidden not in SOURCE


def test_a_handoff_b_tma_and_mma_order() -> None:
    expect_b = SOURCE.index("tma::expect_bytes(")
    cluster_arm = SOURCE.index("everyone::tma::cluster::sync();", expect_b)
    a_wait_branch = SOURCE.index("UNION_X_FC1_K64_A_WAIT_WARP", cluster_arm)
    a_local_wait = SOURCE.index("wait(union_a_tma_arrived", a_wait_branch)
    a_arrive = SOURCE.index(
        "warp::tma::cluster::arrive(union_a_cluster_arrived", a_local_wait
    )
    a_tma_branch = SOURCE.index("UNION_X_FC1_K64_A_TMA_WARP", a_arrive)
    gather = SOURCE.index("union_x_issue_gather4_a_tile(", a_tma_branch)
    b_load = SOURCE.index("tma::cluster::load_async(", cluster_arm)
    wait_a = SOURCE.index(
        "tma::cluster::wait(union_a_cluster_arrived", cluster_arm
    )
    wait_b = SOURCE.index("wait(b_tma_arrived", wait_a)
    mma = SOURCE.index("mm2_ABt(", wait_b)
    commit = SOURCE.index("detail::tcgen05::commit", mma)
    assert expect_b < cluster_arm < a_wait_branch < a_local_wait < a_arrive
    assert a_arrive < a_tma_branch < gather
    assert cluster_arm < b_load
    assert wait_a < wait_b < mma < commit
    assert "config::CLUSTER_SIZE * sizeof(mlp_bf16_tile)" in SOURCE[
        expect_b:cluster_arm
    ]
    assert "init_semaphore(union_a_tma_arrived, 0, 1)" in SOURCE
    assert (
        "union_a_cluster_arrived, 0, config::CLUSTER_SIZE" in SOURCE
    )


def test_raw_fp32_output_is_complete_and_no_forward_features_leak() -> None:
    for token in (
        "union_x_fc1_k64_output_gl = gl<float",
        "tcgen05.ld.sync.aligned.32x32b.x32.b32",
        "UNION_X_FC1_K64_PACKED_N / 32",
        "g.output.raw_ptr[output_offset]",
        "dtype(at::kFloat)",
    ):
        assert token in SOURCE
    for forbidden in (
        "SwiGLU",
        "hidden_row_block_ready",
        "combine_buffer",
        "clc::",
        "MOK_FWD",
    ):
        assert forbidden not in SOURCE[HEADER.read_text().index("static __device__"):]


def test_private_binding_and_fake_are_isolated() -> None:
    assert '#include "union_x_fused_gate_up.cuh"' in MEGA.read_text()
    assert "union_x_fc1_k64_entrypoint" in ENTRYPOINTS.read_text()
    assert 'm.def("union_x_fc1_k64"' in BINDINGS.read_text()
    ops = OPS.read_text()
    assert '@torch.library.custom_op("mok::union_x_fc1_k64"' in ops
    assert '@torch.library.register_fake("mok::union_x_fc1_k64")' in ops
    assert "return union_x.new_empty((256, 256), dtype=torch.float32)" in ops
    assert "dispatch_mlp_swiglu_combine_fwd_bf16" not in SOURCE
    runtime = RUNTIME_TEST.read_text()
    assert "union_x_fc1_k64(" in runtime
    assert "gathered @ gate[:, :64].float().transpose(0, 1)" in runtime
    assert "gathered @ up[:, :64].float().transpose(0, 1)" in runtime
    assert "torch.testing.assert_close" in runtime


def test_kernel_has_no_fallback_or_production_loop() -> None:
    kernel = SOURCE[
        SOURCE.index("static __device__ __forceinline__ void union_x_fc1_k64_kernel"):
        SOURCE.index("static __host__ __forceinline__ at::Tensor")
    ]
    for forbidden in (
        "fallback",
        "while (",
        "macrobatch",
        "minibatch",
        "FUSED_GATE_UP_LOAD_PIPE_DEPTH",
    ):
        assert forbidden not in kernel
