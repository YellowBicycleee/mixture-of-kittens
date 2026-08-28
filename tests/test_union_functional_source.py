from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "mok" / "union_functional.py").read_text()


class UnionFunctionalSourceTest(unittest.TestCase):
    def test_context_is_distinct_and_keeps_schedule_alive(self):
        self.assertIn("class MoKUnionXForwardContext", SOURCE)
        self.assertIn("union_x: torch.Tensor", SOURCE)
        self.assertIn("schedule: MoKUnionSchedule", SOURCE)
        self.assertNotIn("MoKForwardContext(", SOURCE)

    def test_forward_is_bf16_only_and_uses_parallel_op(self):
        self.assertIn("def forward_union_x(", SOURCE)
        self.assertIn(
            "dispatch_mlp_swiglu_combine_fwd_bf16_union_x(", SOURCE
        )
        self.assertNotIn("dispatch_mlp_swiglu_combine_fwd_mxfp8", SOURCE)
        self.assertNotIn("backward(", SOURCE)

    def test_legacy_validation_view_preserves_schedule_fields(self):
        for token in (
            "legacy_schedule = MoKSchedule(",
            "peer_rank=schedule.peer_rank",
            "peer_token_idx=schedule.peer_token_idx",
            "num_tokens=schedule.num_tokens",
            "tokens_per_expert=schedule.tokens_per_expert",
            "validate_inputs(",
        ):
            self.assertIn(token, SOURCE)

    def test_forward_keeps_existing_communication_boundary(self):
        copy_x = SOURCE.index("workspace.x_buffer.copy_(x)")
        first_barrier = SOURCE.index("barrier_all(", copy_x)
        launch = SOURCE.index(
            "dispatch_mlp_swiglu_combine_fwd_bf16_union_x(",
            first_barrier,
        )
        second_barrier = SOURCE.index("barrier_all(", launch)
        epilogue = SOURCE.index("output = fwd_epilogue(", second_barrier)
        self.assertLess(copy_x, first_barrier)
        self.assertLess(first_barrier, launch)
        self.assertLess(launch, second_barrier)
        self.assertLess(second_barrier, epilogue)

    def test_config_preflight_runs_before_collective_or_forward_launch(self):
        self.assertIn("def _validate_union_config(", SOURCE)
        self.assertIn(
            "config.fwd_num_comm_sms >= "
            "device_properties.multi_processor_count",
            SOURCE,
        )
        build = SOURCE.index("def build_union_schedule(")
        build_preflight = SOURCE.index(
            "_validate_union_config(workspace, config)", build
        )
        all_gather = SOURCE.index("all_gather_top_experts(", build)
        forward = SOURCE.index("def forward_union_x(")
        forward_preflight = SOURCE.index(
            "_validate_union_config(workspace, config)", forward
        )
        launch = SOURCE.index(
            "dispatch_mlp_swiglu_combine_fwd_bf16_union_x(", forward
        )
        self.assertLess(build_preflight, all_gather)
        self.assertLess(forward_preflight, launch)

    def test_complete_low_level_preflight_runs_before_first_barrier(self):
        forward = SOURCE.index("def forward_union_x(")
        validate = SOURCE.index(
            "_validate_union_x_forward_args(", forward
        )
        copy_x = SOURCE.index("workspace.x_buffer.copy_(x)", forward)
        barrier = SOURCE.index("barrier_all(", copy_x)
        self.assertLess(validate, copy_x)
        self.assertLess(copy_x, barrier)
        self.assertIn("workspace.ep_size != 8", SOURCE)
        self.assertIn("workspace.hidden_size != 4096", SOURCE)


if __name__ == "__main__":
    unittest.main()
