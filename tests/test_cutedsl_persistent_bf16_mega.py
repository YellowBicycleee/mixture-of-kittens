from __future__ import annotations

import ast
from pathlib import Path
import unittest


_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mok"
    / "cutedsl"
    / "_persistent_bf16_mega.py"
)
_SOURCE = _SOURCE_PATH.read_text()
_TREE = ast.parse(_SOURCE)


class PersistentBf16MegaSourceTest(unittest.TestCase):
    def test_one_kernel_one_clustered_launch(self) -> None:
        kernels = []
        launches = []
        for node in ast.walk(_TREE):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Attribute)
                        and isinstance(decorator.value, ast.Name)
                        and decorator.value.id == "cute"
                        and decorator.attr == "kernel"
                    ):
                        kernels.append(node.name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "launch"
            ):
                launches.append(node)
        self.assertEqual(kernels, ["kernel"])
        self.assertEqual(len(launches), 1)
        keywords = {keyword.arg for keyword in launches[0].keywords}
        self.assertTrue({"grid", "block", "cluster", "smem", "stream"} <= keywords)

    def test_selector_and_fused_role_ladder_match_accepted_cuda(self) -> None:
        self.assertIn("CLC_PIPE_DEPTH = 1", _SOURCE)
        self.assertIn("FUSED_GATE_UP_TASK_GROUP_SIZE = 1", _SOURCE)
        self.assertIn("FUSED_DOWN_TASK_GROUP_SIZE = 1", _SOURCE)
        self.assertIn("if kind == Int32(0)", _SOURCE)
        self.assertIn("if kind == Int32(1)", _SOURCE)
        self.assertIn("if kind == Int32(2)", _SOURCE)
        self.assertIn("if kind == Int32(3)", _SOURCE)
        self.assertNotIn("shared_swiglu_tasks", _SOURCE)
        self.assertNotIn("minibatch_swiglu_tasks", _SOURCE)

    def test_weights_remain_separate_strided_views(self) -> None:
        self.assertIn("routed_gate_weights.permute(1, 2, 0)", _SOURCE)
        self.assertIn("routed_up_weights.permute(1, 2, 0)", _SOURCE)
        self.assertIn("routed_down_weights.permute(1, 2, 0)", _SOURCE)
        self.assertNotIn("torch.cat", _SOURCE)
        self.assertNotIn("packed_gate_up_weights", _SOURCE)

    def test_fc1_weight_tiles_advance_by_logical_n128(self) -> None:
        self.assertEqual(
            _SOURCE.count("tile_coord_n * Int32(_GATE_LOGICAL_N)"),
            4,
        )
        self.assertGreaterEqual(_SOURCE.count("gate_tile_base"), 4)
        self.assertGreaterEqual(_SOURCE.count("up_tile_base"), 4)
        self.assertIn("cute.domain_offset(", _SOURCE)

    def test_struct_extents_are_literal_for_supported_toolchain(self) -> None:
        self.assertNotIn("from __future__ import annotations", _SOURCE)
        self.assertIn(
            "clc_drain_mbarriers: cute.struct.MemRange[\n"
            "        cutlass.Int64,\n"
            "        16,\n"
            "    ]",
            _SOURCE,
        )
        self.assertIn("MemRange[cutlass.Int32, 32]", _SOURCE)
        self.assertIn(
            "gemm_ab_mbarriers: cute.struct.MemRange[cutlass.Int64, 12]",
            _SOURCE,
        )
        self.assertIn(
            "gemm_acc_mbarriers: cute.struct.MemRange[cutlass.Int64, 4]",
            _SOURCE,
        )
        self.assertIn("num_stages=_GATE_ACC_STAGES", _SOURCE)
        self.assertIn("MemRange[cutlass.Uint8, 229376]", _SOURCE)
        self.assertNotIn("4 * CLC_DRAIN_WARPS", _SOURCE)

    def test_block_load_shim_and_quack_release_election_are_explicit(self) -> None:
        self.assertEqual(_SOURCE.count("_make_tma_block_load_fn("), 10)
        self.assertNotIn("quack_copy_utils", _SOURCE)
        self.assertEqual(_SOURCE.count("elect_one_release=True"), 1)
        self.assertEqual(_SOURCE.count("syncwarp_before_release=False"), 1)

    def test_exact_dependency_versions_are_visible(self) -> None:
        self.assertIn('_REQUIRED_CUTLASS_DSL = "4.6.2"', _SOURCE)
        self.assertIn('_REQUIRED_QUACK_VERSION = "0.6.4"', _SOURCE)

    def test_supported_toolchain_converts_loaded_values_not_tensor_objects(self) -> None:
        self.assertEqual(
            _SOURCE.count(
                "cute.make_rmem_tensor_like(tRS_rD, BFloat16)"
            ),
            3,
        )
        self.assertNotIn("tRS_rD.to(BFloat16)", _SOURCE)
        bf16_to_calls = [
            node
            for node in ast.walk(_TREE)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "BFloat16"
        ]
        self.assertEqual(len(bf16_to_calls), 4)

        value_load_casts = []
        swiglu_scalar_casts = []
        for call in bf16_to_calls:
            receiver = call.func.value
            if (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "load"
                and isinstance(receiver.func.value, ast.Name)
                and receiver.func.value.id == "tRS_rD"
            ):
                value_load_casts.append(call)
            if (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == "swiglu"
            ):
                swiglu_scalar_casts.append(call)
        self.assertEqual(len(value_load_casts), 3)
        self.assertEqual(len(swiglu_scalar_casts), 1)

    def test_comm_dynamic_region_carries_only_shared_memory_leaf_pointers(self) -> None:
        mega_class = next(
            node
            for node in _TREE.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_PersistentBf16Mega"
        )
        comm_prefix = next(
            node
            for node in mega_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_comm_prefix"
        )
        parameter_names = [argument.arg for argument in comm_prefix.args.args]
        self.assertNotIn("storage", parameter_names)
        self.assertEqual(
            parameter_names[11:14],
            ["dispatch_mbarrier", "combine_mbarriers", "arena"],
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Name) and node.id == "storage"
                for node in ast.walk(comm_prefix)
            )
        )
        self.assertLess(
            _SOURCE.index("dispatch_mbarrier = storage.dispatch_mbarrier.ptr"),
            _SOURCE.index("if cluster_index < Int32(comm_clusters):"),
        )
        self.assertLess(
            _SOURCE.index(
                "combine_mbarriers = storage.combine_mbarriers.data_ptr()"
            ),
            _SOURCE.index("if cluster_index < Int32(comm_clusters):"),
        )
        self.assertLess(
            _SOURCE.index("comm_arena = storage.arena.data_ptr()"),
            _SOURCE.index("if cluster_index < Int32(comm_clusters):"),
        )
        self.assertEqual(
            _SOURCE.count("clc_response_ptr = storage.clc_response.data_ptr()"),
            1,
        )
        self.assertEqual(_SOURCE.count("clc_response_ptr,"), 5)

        kernel = next(
            node
            for node in mega_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "kernel"
        )
        for region in ast.walk(kernel):
            if isinstance(region, (ast.If, ast.While)):
                self.assertFalse(
                    any(
                        isinstance(node, ast.Name) and node.id == "storage"
                        for node in ast.walk(region)
                    ),
                    f"shared-storage object captured by staged region at "
                    f"line {region.lineno}",
                )

    def test_compile_and_run_share_the_exact_argument_builder(self) -> None:
        functions = {
            node.name: node
            for node in _TREE.body
            if isinstance(node, ast.FunctionDef)
        }
        prepare_calls = [
            node
            for node in ast.walk(functions["prepare_mega_bf16"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "make_mega_args"
        ]
        self.assertEqual(len(prepare_calls), 1)
        for name in ("compile_mega_bf16", "run_mega_bf16"):
            calls = [
                node
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "prepare_mega_bf16"
            ]
            self.assertEqual(len(calls), 1)
        run_executor_calls = [
            node
            for node in ast.walk(functions["run_mega_bf16"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "executor"
        ]
        self.assertEqual(len(run_executor_calls), 1)
        self.assertIn("_MEGA_RUNTIME_PREFIX_ARGS = 27", _SOURCE)
        self.assertIn(
            "runtime_args = cute_args[:_MEGA_RUNTIME_PREFIX_ARGS] + cute_args[-1:]",
            ast.get_source_segment(_SOURCE, functions["prepare_mega_bf16"]),
        )
        mega_class = next(
            node
            for node in _TREE.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_PersistentBf16Mega"
        )
        mega_call = next(
            node
            for node in mega_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        parameter_names = [argument.arg for argument in mega_call.args.args[1:]]
        self.assertEqual(len(parameter_names), 35)
        self.assertEqual(parameter_names[26], "y_routed_done")
        self.assertEqual(
            parameter_names[27:34],
            [
                "num_local_tokens",
                "schedule_capacity",
                "macrobatch_size",
                "minibatch_size",
                "num_comm_sms",
                "swiglu_limit",
                "is_clamped",
            ],
        )
        self.assertEqual(parameter_names[34], "stream")
        executor_call = run_executor_calls[0]
        self.assertEqual(len(executor_call.args), 1)
        self.assertIsInstance(executor_call.args[0], ast.Starred)
        self.assertEqual(executor_call.args[0].value.id, "runtime_args")

    def test_clamp_and_empty_routed_schedule_preserve_cuda_semantics(self) -> None:
        self.assertIn("swiglu_limit: cutlass.Constexpr", _SOURCE)
        self.assertIn("is_clamped: cutlass.Constexpr", _SOURCE)
        self.assertIn("if is_clamped:", _SOURCE)
        self.assertIn("gate_value = cutlass.min", _SOURCE)
        self.assertIn("cutlass.max(up_value, -limit)", _SOURCE)
        self.assertIn("if num_macrobatches > Int32(0):", _SOURCE)

    def test_down_ring_reuse_waits_only_for_overlapping_prior_rows(self) -> None:
        self.assertIn("prior_rows = cutlass.min(", _SOURCE)
        self.assertIn("local_cta_row = (", _SOURCE)
        self.assertIn("if local_cta_row < prior_rows:", _SOURCE)
        self.assertIn(
            "prior_offset + local_cta_row", _SOURCE
        )


if __name__ == "__main__":
    unittest.main()
