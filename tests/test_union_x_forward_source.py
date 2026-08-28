from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FORWARD = (
    ROOT / "csrc" / "megakernel" / "forward_union_x.cuh"
).read_text()
LEGACY_FORWARD = (
    ROOT / "csrc" / "megakernel" / "forward.cuh"
).read_text()
MEGAKERNEL = (
    ROOT / "csrc" / "megakernel" / "megakernel.cuh"
).read_text()
ENTRYPOINTS = (
    ROOT / "csrc" / "megakernel" / "entrypoints.cuh"
).read_text()
BINDINGS = (ROOT / "csrc" / "bindings.cu").read_text()
OPS = (ROOT / "mok" / "union_ops.py").read_text()


class UnionXForwardSourceContractTest(unittest.TestCase):
    def test_private_fixed_ep8_bf16_specialization(self):
        self.assertIn("struct globals_union_x_fwd", FORWARD)
        self.assertIn("static_assert(NUM_DEVICES == 8);", FORWARD)
        self.assertIn("static_assert(!USE_MXFP8);", FORWARD)
        self.assertIn("static_assert(FWD_CLC_PIPE_DEPTH == 2);", FORWARD)
        self.assertIn("static_assert(FWD_GATE_GROUP_SIZE == 1);", FORWARD)
        self.assertIn("static_assert(FWD_DOWN_GROUP_SIZE == 1);", FORWARD)
        self.assertIn(
            "8, RoutedPrecision::BF16, 2, 1, 1", ENTRYPOINTS
        )
        self.assertNotIn("union_x", LEGACY_FORWARD)

    def test_union_storage_and_four_a_handoffs(self):
        self.assertIn("routed_bf16_gl union_x", FORWARD)
        self.assertIn("CUtensorMap union_x_gather4_tma", FORWARD)
        self.assertIn("index_gl union_state", FORWARD)
        self.assertIn("index_gl route_to_union", FORWARD)
        self.assertIn(
            "{union_capacity * UNION_X_DISPATCH_HIDDEN_SLICES}",
            FORWARD,
        )
        self.assertIn(
            "semaphore union_a_arrived[UNION_X_RGU_STAGES]", FORWARD
        )
        self.assertIn(
            "init_semaphore(union_a_arrived[i], 0, config::CLUSTER_SIZE)",
            FORWARD,
        )
        self.assertIn(
            "uint32_t union_a_gather_bitfield = 0xFFFF0000", FORWARD
        )
        self.assertIn(
            "semaphore union_a_inputs_reusable[UNION_X_RGU_STAGES]",
            FORWARD,
        )
        self.assertIn(
            "init_semaphore(union_a_inputs_reusable[i], 0, 1)",
            FORWARD,
        )
        self.assertIn(
            "uint32_t union_a_reuse_bitfield = 0xFFFF0000", FORWARD
        )
        self.assertIn(
            "semaphore union_a_tma_arrived[UNION_X_RGU_STAGES]",
            FORWARD,
        )
        self.assertIn(
            "init_semaphore(union_a_tma_arrived[i], 0, 1)", FORWARD
        )
        self.assertIn(
            "uint32_t union_a_tma_bitfield = 0xFFFF0000", FORWARD
        )
        self.assertIn(
            "semaphore union_b_inputs_armed[UNION_X_RGU_STAGES]",
            FORWARD,
        )
        self.assertIn(
            "init_semaphore(union_b_inputs_armed[i], 0, 1)",
            FORWARD,
        )
        self.assertIn(
            "uint32_t union_b_arm_bitfield = 0xFFFF0000", FORWARD
        )
        self.assertIn(
            ".union_x_gather4_tma = union_x_make_gather4_tma(union_x)",
            FORWARD,
        )
        self.assertIn(
            "union_x_prefetch_gather4_tma(g.union_x_gather4_tma)",
            FORWARD,
        )

    def test_comm_uses_union_dispatch_before_blocking_combine(self):
        self.assertIn("union_x_dispatch_task(", FORWARD)
        loop = FORWARD.index(
            "for (int macrobatch_idx = num_macrobatches - 1;"
        )
        routed = FORWARD.index(
            "expert_gate_up_swiglu_union_x_bf16_kernel", loop
        )
        section = FORWARD[loop:routed]
        dispatch = section.index(
            "dispatch(macrobatch_idx - 1, task_idx);"
        )
        combine = section.index("combine(macrobatch_idx, task_idx);")
        self.assertLess(dispatch, combine)
        self.assertIn("g.union_x_ready", FORWARD)
        self.assertNotIn("dispatch_kernel<false>", section)

    def test_compute_replaces_only_routed_a_path(self):
        self.assertIn(
            "expert_gate_up_swiglu_ep8_tuned_kernel<true, IS_CLAMPED>",
            FORWARD,
        )
        self.assertIn(
            "expert_gate_up_swiglu_union_x_bf16_kernel<IS_CLAMPED>",
            FORWARD,
        )
        self.assertIn(
            "g.hidden_routed, g.w_routed_down, nullptr, nullptr",
            FORWARD,
        )
        self.assertIn("combine_kernel<true>", FORWARD)
        self.assertNotIn("g.x_fp8_routed", FORWARD)
        self.assertNotIn("g.gate_up_tile_ready", FORWARD)

    def test_only_routed_gate_up_task_width_is_paired(self):
        self.assertIn(
            "w_routed_gate.rows() / UNION_X_RGU_PACKED_HIDDEN_N",
            FORWARD,
        )
        self.assertIn(
            "w_shared_gate.rows() / config::SWIGLU_Nb", FORWARD
        )
        self.assertIn(
            "g.w_routed_down.rows() / config::MLP_Nb", FORWARD
        )
        self.assertIn(
            "2 * config::MLP_Nb <= decltype(tm_alloc)::cols", FORWARD
        )
        self.assertIn("paired_d_tt", FORWARD)
        # Down still waits for the old per-N128 hidden publication count.
        self.assertIn(
            "g.hidden_shared.cols() / config::SWIGLU_Nb", FORWARD
        )

    def test_ring_and_readiness_extents_match_legacy(self):
        for declaration in (
            "at::empty({macrobatch_size, intermediate_dim}, x.options())",
            "at::empty({macrobatch_size, hidden_dim}, x.options())",
            "{shared_row_blocks + routed_row_blocks}",
            "{num_global_minibatches}",
            "{num_global_row_blocks}",
        ):
            self.assertIn(declaration, FORWARD)
        self.assertIn(
            "static_cast<int64_t>(NUM_DEVICES) * num_local_tokens",
            FORWARD,
        )

    def test_nine_output_abi_and_fake(self):
        name = "dispatch_mlp_swiglu_combine_fwd_bf16_union_x"
        self.assertIn('#include "forward_union_x.cuh"', MEGAKERNEL)
        self.assertIn(name + "_entrypoint", ENTRYPOINTS)
        self.assertIn(f'm.def("{name}"', BINDINGS)
        self.assertIn(f'"mok::{name}"', OPS)
        self.assertIn(
            "x.new_empty((8 * num_local_tokens, hidden_size))", OPS
        )
        returned = FORWARD[FORWARD.rindex("return {") :]
        for tensor in (
            "union_x",
            "gate_shared",
            "gate_routed",
            "up_shared",
            "up_routed",
            "hidden_shared",
            "hidden_routed",
            "y_shared",
            "y_routed",
        ):
            self.assertIn(tensor, returned)

    def test_low_level_comm_count_preserves_forward_progress(self):
        self.assertIn(
            "torch.cuda.get_device_properties(", OPS
        )
        self.assertIn(
            "num_comm_sms must leave at least one compute SM", OPS
        )
        self.assertIn(
            "at::cuda::getDeviceProperties(x.get_device())", FORWARD
        )
        self.assertIn(
            "num_comm_sms < device_properties->multiProcessorCount",
            FORWARD,
        )

    def test_retired_fc1_probe_is_not_in_production_abi(self):
        for source in (MEGAKERNEL, ENTRYPOINTS, BINDINGS, OPS):
            self.assertNotIn("union_x_fc1_k64", source)
        self.assertNotIn("union_x_fused_gate_up.cuh", MEGAKERNEL)

    def test_high_level_union_forward_is_distinct_and_forward_only(self):
        union_functional = (
            ROOT / "mok" / "union_functional.py"
        ).read_text()
        self.assertIn("def forward_union_x(", union_functional)
        self.assertIn(
            "dispatch_mlp_swiglu_combine_fwd_bf16_union_x(",
            union_functional,
        )
        self.assertIn("class MoKUnionXForwardContext", union_functional)
        self.assertNotIn("def backward", union_functional)


if __name__ == "__main__":
    unittest.main()
