from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "csrc" / "megakernel" / "union_x_gather.cuh").read_text()


def test_gather4_geometry_matches_one_cta_a_tile() -> None:
    for token in (
        "UNION_X_GATHER_ROWS = 128",
        "UNION_X_GATHER_K_COLS = 64",
        "UNION_X_GATHER_ROWS_PER_TMA = 4",
        "UNION_X_GATHER_TMA_ISSUERS = WARP_THREADS",
        "UNION_X_GATHER_ROW_BYTES == 128",
        "UNION_X_GATHER_BYTES == 16 * 1024",
        "UNION_X_GATHER_BYTES == sizeof(mlp_bf16_tile)",
        "mlp_bf16_tile::swizzle_bytes == 128",
    ):
        assert token in SOURCE

    covered = []
    for lane in range(32):
        covered.extend(range(lane * 4, lane * 4 + 4))
    assert covered == list(range(128))


def test_host_descriptor_is_true_2d_k64_by_one_row() -> None:
    for token in (
        "CUtensorMap union_x_make_gather4_tma(",
        "const cuuint64_t gmem_dims[2]",
        "union_x.size(1)",
        "union_x.size(0)",
        "const cuuint64_t gmem_strides[1]",
        "const cuuint32_t box_dims[2]",
        "static_cast<cuuint32_t>(UNION_X_GATHER_K_COLS)",
        "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
        "CU_TENSOR_MAP_INTERLEAVE_NONE",
        "CU_TENSOR_MAP_SWIZZLE_128B",
        "CU_TENSOR_MAP_L2_PROMOTION_L2_256B",
        "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE",
        "cuTensorMapEncodeTiled(",
    ):
        assert token in SOURCE
    assert "\n        2,\n        union_x.data_ptr()," in SOURCE
    assert "\n        1,\n    };" in SOURCE


def test_route_metadata_is_one_int4_load_per_lane() -> None:
    helper = SOURCE[
        SOURCE.index("int4 union_x_load_route_ids4(") :
        SOURCE.index("void union_x_issue_gather4_a_tile(")
    ]
    for token in (
        "warp::laneid() * UNION_X_GATHER_ROWS_PER_TMA",
        "ld.global.v4.u32",
        '"=r"(union_ids.x)',
        '"=r"(union_ids.w)',
        "union_ids.x < -1",
        "union_ids.w >= union_rows",
    ):
        assert token in helper
    assert "for (" not in helper


def test_all_lanes_issue_disjoint_gather4_with_exact_expected_bytes() -> None:
    issue = SOURCE[SOURCE.index("void union_x_issue_gather4_a_tile(") :]
    expect = issue.index("tma::expect_bytes(")
    sync = issue.index("__syncwarp();", expect)
    gather = issue.index("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4")
    assert expect < sync < gather
    for token in (
        "tma::expect_bytes(tma_arrived, UNION_X_GATHER_BYTES)",
        "warp::laneid() * UNION_X_GATHER_ROWS_PER_TMA",
        "* UNION_X_GATHER_ROW_BYTES",
        "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint",
        '"r"(union_ids.x)',
        '"r"(union_ids.w)',
    ):
        assert token in issue
    assert "if (union_ids" not in issue


def test_padding_uses_descriptor_oob_zero_fill_without_sentinel_storage() -> None:
    assert "union id of -1" in SOURCE
    assert "zero-fills that destination row" in SOURCE
    assert "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE" in SOURCE
    assert "float4 values" not in SOURCE
    assert "move<float4>::ldg" not in SOURCE
    assert "move<float4>::sts" not in SOURCE
    assert "mlp_bf16_tile::idx" not in SOURCE


def test_shared_helper_does_not_own_mma_or_cluster_handoff() -> None:
    for forbidden in (
        "mm2_",
        "mma2_",
        "barrier_arrive",
        "cluster::arrive",
        "hidden_row_block_ready",
    ):
        assert forbidden not in SOURCE
