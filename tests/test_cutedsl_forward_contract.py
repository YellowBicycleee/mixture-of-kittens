from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "mok" / "cutedsl" / "forward_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("mok_cutedsl_forward_contract", _CONTRACT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
_FORWARD_SOURCE = (_CONTRACT_PATH.parent / "forward.py").read_text()
_FORWARD_TREE = ast.parse(_FORWARD_SOURCE)
_TMA_SOURCE = (_CONTRACT_PATH.parent / "_tma_1d.py").read_text()
_QUACK_SOURCE = (_CONTRACT_PATH.parent / "quack_gemm.py").read_text()
_QUACK_TREE = ast.parse(_QUACK_SOURCE)
_FUNCTIONAL_SOURCE = (_CONTRACT_PATH.parents[1] / "functional.py").read_text()
_FUNCTIONAL_TREE = ast.parse(_FUNCTIONAL_SOURCE)
_BENCHMARK_SOURCE = (
    _CONTRACT_PATH.parents[2] / "benchmarks" / "cutedsl" / "bench_qwen_fwd_ab.py"
).read_text()


class CuTeDSLForwardContractTest(unittest.TestCase):
    class _DeviceScalar:
        def __init__(self, value: int) -> None:
            self.value = value
            self.item_calls = 0

        def item(self) -> int:
            self.item_calls += 1
            return self.value

    def test_host_num_tokens_mirror_avoids_device_scalar_read(self) -> None:
        scalar = self._DeviceScalar(768)
        self.assertEqual(
            contract.resolve_routed_num_tokens(scalar, 512, 1024),
            512,
        )
        self.assertEqual(scalar.item_calls, 0)

    def test_backend_is_explicit_opt_in_and_import_stays_lazy(self) -> None:
        config_class = next(
            node
            for node in _FUNCTIONAL_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "MoKConfig"
        )
        fwd_backend = next(
            node
            for node in config_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "fwd_backend"
        )
        self.assertEqual(ast.literal_eval(fwd_backend.value), "cuda")
        self.assertFalse(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and "cutedsl" in node.module
                for node in _FUNCTIONAL_TREE.body
            )
        )
        self.assertIn(
            "from .cutedsl.persistent_bf16 import (",
            _FUNCTIONAL_SOURCE,
        )
        self.assertNotIn(
            "from .cutedsl.forward import forward_bf16 as cutedsl_forward_bf16",
            _FUNCTIONAL_SOURCE,
        )
        self.assertIn('if config.fwd_backend == "cuda":', _FUNCTIONAL_SOURCE)
        self.assertEqual(
            _FUNCTIONAL_SOURCE.count(
                "dispatch_mlp_swiglu_combine_fwd_bf16("
            ),
            1,
        )

    def test_schedule_build_keeps_num_tokens_on_device_for_both_backends(self) -> None:
        build_schedule = next(
            node
            for node in _FUNCTIONAL_TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_schedule"
        )
        source = ast.get_source_segment(_FUNCTIONAL_SOURCE, build_schedule)
        self.assertIsNotNone(source)
        self.assertNotIn("num_tokens.item()", source)
        self.assertNotIn("num_tokens_host=", source)

    def test_pipeline_v2_saves_preact_only_for_macro_zero(self) -> None:
        self.assertIs(contract.REPLAY_GATE_UP_STORE_ELISION, True)
        m = 32768
        self.assertTrue(contract.should_store_routed_preact(0, m))
        self.assertFalse(contract.should_store_routed_preact(m, m))
        self.assertFalse(contract.should_store_routed_preact(3 * m, m))
        # Every window belonging to macro 0 must be saved for macrobatches
        # wider than one wavefront, while every replay macro remains no-store.
        wide_m = 2 * contract.WAVEFRONT_WINDOW_ROWS
        decisions = tuple(
            contract.should_store_routed_preact(global_offset, wide_m)
            for global_offset, _, _ in contract.wavefront_windows(2 * wide_m, wide_m)
        )
        self.assertEqual(decisions, (False, False, True, True))

    def test_packed_weight_cache_key_tracks_tensor_identity_and_version(self) -> None:
        original = contract.packed_weight_cache_key(101, 4, 202, 7)
        self.assertEqual(original, (101, 4, 202, 7))
        self.assertNotEqual(
            original,
            contract.packed_weight_cache_key(101, 5, 202, 7),
        )
        self.assertNotEqual(
            original,
            contract.packed_weight_cache_key(303, 4, 202, 7),
        )

    def test_packed_gate_up_shapes_cover_shared_routed_and_bad_inputs(self) -> None:
        self.assertEqual(
            contract.packed_gate_up_shape("shared", (1024, 4096)),
            (2048, 4096),
        )
        self.assertEqual(
            contract.packed_gate_up_shape("routed", (64, 1024, 4096)),
            (64, 2048, 4096),
        )
        for slot, shape in (
            ("shared", (64, 1024, 4096)),
            ("routed", (1024, 4096)),
            ("unknown", (1024, 4096)),
        ):
            with self.subTest(slot=slot, shape=shape):
                with self.assertRaises(ValueError):
                    contract.packed_gate_up_shape(slot, shape)

    def test_quack_custom_epilogue_rounds_before_swiglu_and_elides_replay_stores(self) -> None:
        for snippet in (
            "from cutlass import BFloat16, Float32",
            "from quack.activation import swiglu",
            "from quack.epilogue.frontend import gemm_epilogue, unpack",
            '@gemm_epilogue(outputs=("gate", "up", "hidden"), mode="acc_pair")',
            '@gemm_epilogue(outputs=("hidden",), mode="acc_pair")',
            ".to(BFloat16).to(Float32)",
            "epilogue.gemm(",
            "epi_args=outputs",
            'concat_layout=("B",)',
            "is_dynamic_persistent=True",
            "GATED_CLUSTER_M = 1",
            'cat_dim = 0 if slot == "shared" else 1',
            "packed = torch.cat(",
            "int(gate_weights_enk._version)",
            "int(up_weights_enk._version)",
            "ready_event.record(torch.cuda.current_stream(gate_weights_enk.device))",
            "wait_event(entry.ready_event)",
            "entry.packed.record_stream(stream)",
        ):
            self.assertIn(snippet, _QUACK_SOURCE)
        self.assertEqual(_QUACK_SOURCE.count("torch.cat("), 1)
        self.assertNotIn("\n        b_kn=True,", _QUACK_SOURCE)
        self.assertNotIn("\n        packed_weights_e2ik.transpose", _QUACK_SOURCE)
        self.assertNotIn("torch.cat(", _FORWARD_SOURCE)
        self.assertEqual(
            _FORWARD_SOURCE.count("packed_routed_gate_up_weights("),
            1,
        )
        self.assertEqual(
            _FORWARD_SOURCE.count("packed_shared_gate_up_weights("),
            1,
        )
        self.assertIn("shared_gated_swiglu(", _FORWARD_SOURCE)
        self.assertIn("routed_gated_swiglu(", _FORWARD_SOURCE)
        self.assertIn("gate_output=gate_window", _FORWARD_SOURCE)
        self.assertIn("up_output=up_window", _FORWARD_SOURCE)
        self.assertIn('_packed_gate_up_weights("shared"', _QUACK_SOURCE)
        self.assertIn('_packed_gate_up_weights("routed"', _QUACK_SOURCE)
        self.assertIn("_PACKED_GATE_UP_BY_SLOT", _QUACK_SOURCE)
        self.assertNotIn("packed_preact_scratch", _FORWARD_SOURCE)
        self.assertNotIn("packed_preact_window", _FORWARD_SOURCE)
        self.assertNotIn("swiglu_routed_window", _FORWARD_SOURCE)
        self.assertNotIn("class _SwiGLUKernel", _FORWARD_SOURCE)

    def test_bf16_harness_keeps_fixed_seed_output_and_eight_grads(self) -> None:
        self.assertIn("MACROBATCH, MINIBATCH = 32768, 4096", _BENCHMARK_SOURCE)
        self.assertIn('bwd_schedule="macrobatch"', _BENCHMARK_SOURCE)
        self.assertIn('"fixed_seed": "1234 + EP rank', _BENCHMARK_SOURCE)
        self.assertIn("RESULT_NAMES = (", _BENCHMARK_SOURCE)
        self.assertIn('"output",', _BENCHMARK_SOURCE)
        self.assertEqual(_BENCHMARK_SOURCE.count('"d_w_'), 6)
        self.assertIn('"d_x",', _BENCHMARK_SOURCE)
        self.assertIn('"d_router_weights",', _BENCHMARK_SOURCE)
        self.assertIn("def full_scale_parity_case(", _BENCHMARK_SOURCE)
        self.assertIn('"expected_generations": 32', _BENCHMARK_SOURCE)
        self.assertIn(
            "T=102400 BF16 public FWD/CUDA-BWD parity failed",
            _BENCHMARK_SOURCE,
        )
        self.assertIn(
            'owner = importlib.import_module("mok.cutedsl.persistent_bf16")',
            _BENCHMARK_SOURCE,
        )
        self.assertNotIn(
            'owner = importlib.import_module("mok.cutedsl.forward")',
            _BENCHMARK_SOURCE,
        )
        self.assertIn("def checked_forward_backward(", _BENCHMARK_SOURCE)
        self.assertIn('"cuda1_vs_cuda2",', _BENCHMARK_SOURCE)
        self.assertIn('"cuda1_vs_cutedsl",', _BENCHMARK_SOURCE)
        self.assertIn('"cuda2_vs_cutedsl",', _BENCHMARK_SOURCE)
        self.assertIn(
            "gradients = backward(config, workspace, schedule, context, inputs)",
            _BENCHMARK_SOURCE,
        )
        self.assertIn(
            "*(getattr(context, name).clone() for name in FORWARD_CONTEXT_NAMES)",
            _BENCHMARK_SOURCE,
        )
        self.assertIn(
            "gradients = tuple(tensor.clone() for tensor in gradients)",
            _BENCHMARK_SOURCE,
        )
        self.assertIn("and global_maximum <= atol", _BENCHMARK_SOURCE)
        self.assertIn("and global_relative <= rtol", _BENCHMARK_SOURCE)

    def test_missing_host_num_tokens_mirror_keeps_legacy_fallback(self) -> None:
        scalar = self._DeviceScalar(768)
        self.assertEqual(
            contract.resolve_routed_num_tokens(scalar, None, 1024),
            768,
        )
        self.assertEqual(scalar.item_calls, 1)

    def test_host_num_tokens_mirror_has_strict_type_and_range(self) -> None:
        scalar = self._DeviceScalar(0)
        for invalid in (True, 512.0, "512"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "integer or None"):
                    contract.resolve_routed_num_tokens(scalar, invalid, 1024)
        for invalid in (-1, 1025):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RuntimeError, "schedule capacity"):
                    contract.resolve_routed_num_tokens(scalar, invalid, 1024)
        self.assertEqual(scalar.item_calls, 0)

    def test_reverse_macros_leave_macro_zero_in_ring(self) -> None:
        self.assertEqual(contract.macro_offsets(0, 4096), ())
        self.assertEqual(contract.macro_offsets(4096, 4096), (0,))
        self.assertEqual(
            contract.macro_offsets(16384, 4096),
            (12288, 8192, 4096, 0),
        )
        self.assertEqual(
            contract.macro_offsets(16640, 4096),
            (16384, 12288, 8192, 4096, 0),
        )

    def test_wavefront_uses_64k_forward_windows_and_reuses_ring_slots(self) -> None:
        w = contract.WAVEFRONT_WINDOW_ROWS
        self.assertEqual(w, 65536)
        self.assertEqual(
            contract.wavefront_windows(3 * w + 256, 2 * w),
            ((2 * w, 0, w), (3 * w, w, 256), (0, 0, w), (w, w, w)),
        )
        self.assertEqual(contract.wavefront_windows(0, 2 * w), ())

    def test_wavefront_source_keeps_event_dag_and_offset_free_cache_keys(self) -> None:
        self.assertEqual(_FORWARD_SOURCE.count("global_offset: Int32"), 4)
        for snippet in (
            '_aux_stream_for_device("shared", device_index)',
            '_aux_stream_for_device("dispatch", device_index)',
            '_aux_stream_for_device("combine", device_index)',
            "dispatch_torch_stream.wait_event(previous_down_done)",
            "caller_torch_stream.wait_event(dispatch_done)",
            "caller_torch_stream.wait_event(previous_combine_done)",
            "combine_torch_stream.wait_event(down_done)",
            "caller_torch_stream.wait_stream(dispatch_torch_stream)",
            "caller_torch_stream.wait_stream(combine_torch_stream)",
            "caller_torch_stream.wait_stream(shared_torch_stream)",
        ):
            self.assertIn(snippet, _FORWARD_SOURCE)
        cache_keys = {
            call.args[0].elts[0].value: call.args[0]
            for call in ast.walk(_FORWARD_TREE)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_compiled_executor"
            and isinstance(call.args[0], ast.Tuple)
        }
        for name in ("dispatch_window", "combine_window"):
            key_names = {
                node.id
                for node in ast.walk(cache_keys[name])
                if isinstance(node, ast.Name)
            }
            self.assertIn("window_rows", key_names)
            self.assertNotIn("global_offset", key_names)
        dispatch_key_names = {
            node.id
            for node in ast.walk(cache_keys["dispatch_window"])
            if isinstance(node, ast.Name)
        }
        self.assertIn("dispatch_worker_ctas", dispatch_key_names)

    def test_route_index_is_divided_only_for_dispatch(self) -> None:
        route_idx = 37 * contract.TOPK + 6
        self.assertEqual(contract.decode_schedule_entry(7, route_idx), (37, route_idx))
        self.assertIsNone(contract.decode_schedule_entry(-1, 0x5A5A5A5A))

    def test_dispatch_tasks_match_128_by_128_geometry(self) -> None:
        self.assertEqual(contract.DISPATCH_ROW_CHUNK_BYTES, 256)
        self.assertEqual(
            contract.DISPATCH_TILE_COLUMNS * 2,
            contract.DISPATCH_ROW_CHUNK_BYTES,
        )
        self.assertEqual(
            contract.DISPATCH_TILE_BYTES,
            32768,
        )
        self.assertEqual(contract.DISPATCH_STORAGE_BYTES, 33792)
        self.assertEqual(contract.DISPATCH_ROW_CHUNK_BYTES % 16, 0)
        barrier_prefix = (contract.DISPATCH_TILE_ROWS * 8 + 127) // 128 * 128
        self.assertEqual(
            contract.DISPATCH_STORAGE_BYTES,
            barrier_prefix + contract.DISPATCH_TILE_BYTES,
        )
        self.assertEqual(contract.dispatch_task_geometry(4096), (32, 32, 1024))
        self.assertEqual(
            contract.dispatch_task_geometry(contract.WAVEFRONT_WINDOW_ROWS),
            (512, 32, 16384),
        )
        self.assertEqual(contract.dispatch_task_coordinates(0, 4096), (0, 0))
        self.assertEqual(
            contract.dispatch_task_coordinates(31, 4096),
            (0, 3968),
        )
        self.assertEqual(contract.dispatch_task_coordinates(32, 4096), (128, 0))
        self.assertEqual(
            contract.dispatch_task_coordinates(1023, 4096),
            (3968, 3968),
        )

    def test_dispatch_publishes_expected_bytes_before_raw_tma_load(self) -> None:
        expect = _FORWARD_SOURCE.index(
            "cute.arch.mbarrier_arrive_and_expect_tx("
        )
        load = _FORWARD_SOURCE.index("tma_load_1d_raw(", expect)
        wait = _FORWARD_SOURCE.index("cute.arch.mbarrier_wait(", load)
        self.assertLess(expect, load)
        self.assertLess(load, wait)
        # These helpers emit direct PTX and bypass cute.copy's version-specific
        # elect_one lowering in CUTLASS DSL 4.6.x.
        self.assertIn("llvm.inline_asm(", _TMA_SOURCE)
        self.assertNotIn("cute.copy", _TMA_SOURCE)
        self.assertNotIn("elect_one", _TMA_SOURCE)

    def test_combine_tasks_match_cuda_16_by_1024_geometry(self) -> None:
        self.assertEqual(contract.COMBINE_ROW_CHUNK_BYTES, 2048)
        self.assertEqual(
            contract.COMBINE_TILE_COLUMNS * 2,
            contract.COMBINE_ROW_CHUNK_BYTES,
        )
        self.assertEqual(
            contract.COMBINE_TILE_ROWS * contract.COMBINE_ROW_CHUNK_BYTES,
            32768,
        )
        self.assertEqual(contract.combine_task_geometry(4096), (256, 4, 1024))
        self.assertEqual(contract.combine_task_coordinates(0, 4096), (0, 0))
        self.assertEqual(contract.combine_task_coordinates(3, 4096), (0, 3072))
        self.assertEqual(contract.combine_task_coordinates(4, 4096), (16, 0))
        self.assertEqual(
            contract.combine_task_coordinates(1023, 4096),
            (4080, 3072),
        )
        contract.validate_num_comm_sms(contract.DEFAULT_NUM_COMM_SMS)

    def test_combine_pipeline_matches_cuda_depth_and_storage(self) -> None:
        self.assertEqual(contract.COMBINE_PIPE_DEPTH, 7)
        self.assertEqual(contract.COMBINE_ARENA_BYTES, 229376)
        self.assertEqual(contract.COMBINE_STORAGE_BYTES, 229504)
        self.assertEqual(
            contract.combine_pipeline_geometry(4096),
            (1024, 147, 2),
        )

    def test_combine_pipeline_tail_contains_only_existing_tiles(self) -> None:
        self.assertEqual(
            contract.combine_pipeline_stage_coordinates(146, 4096),
            ((4080, 2048), (4080, 3072)),
        )
        self.assertEqual(
            contract.combine_pipeline_geometry(131072),
            (32768, 4682, 1),
        )
        self.assertEqual(
            contract.combine_pipeline_stage_coordinates(4681, 131072),
            ((131056, 3072),),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            contract.combine_pipeline_stage_coordinates(147, 4096)

    def test_combine_pipeline_requires_opt_in_smem_capacity(self) -> None:
        contract.validate_combine_smem_capacity(contract.COMBINE_STORAGE_BYTES)
        with self.assertRaisesRegex(NotImplementedError, "229504"):
            contract.validate_combine_smem_capacity(
                contract.COMBINE_STORAGE_BYTES - 1
            )

    def test_comm_geometry_rejects_partial_tiles_and_odd_pool(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            contract.dispatch_task_geometry(129)
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            contract.combine_task_geometry(17)
        with self.assertRaisesRegex(ValueError, "positive even"):
            contract.validate_num_comm_sms(39)

    def test_standalone_comm_grid_covers_physical_sms(self) -> None:
        self.assertEqual(contract.DISPATCH_CTAS_PER_SM, 4)
        self.assertEqual(contract.standalone_comm_worker_grids(148), (592, 148))
        tasks = contract.dispatch_task_geometry(
            contract.WAVEFRONT_WINDOW_ROWS
        )[2]
        visited = [
            task
            for block in range(592)
            for task in range(block, tasks, 592)
        ]
        self.assertEqual(len(visited), tasks)
        self.assertEqual(set(visited), set(range(tasks)))
        self.assertEqual(
            contract.standalone_comm_worker_grids(148, combine_ctas_per_sm=2),
            (592, 296),
        )
        for invalid in (True, 0, -1):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                contract.standalone_comm_worker_grids(invalid)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                contract.standalone_comm_worker_grids(
                    148, combine_ctas_per_sm=invalid
                )

    def test_macro_local_cu_seqlens_skew_and_zero_experts(self) -> None:
        # The macro starts inside expert 0, crosses a zero-row expert, consumes
        # all of expert 2, and ends inside expert 3.
        self.assertEqual(
            contract.macro_local_cu_seqlens([10, 0, 4, 8], 8, 8),
            (0, 2, 2, 6, 8),
        )

    def test_macro_local_cu_seqlens_single_expert_spans_macro(self) -> None:
        self.assertEqual(
            contract.macro_local_cu_seqlens([20, 0, 5], 8, 8),
            (0, 8, 8, 8),
        )

    def test_macro_local_cu_seqlens_partial_tail(self) -> None:
        self.assertEqual(
            contract.macro_local_cu_seqlens([3, 0, 7, 2], 8, 4),
            (0, 0, 0, 2, 4),
        )

    def test_macro_local_cu_seqlens_rejects_out_of_range_macro(self) -> None:
        with self.assertRaisesRegex(ValueError, "extends past"):
            contract.macro_local_cu_seqlens([3, 5], 4, 8)

    def test_fixed_qwen_contract(self) -> None:
        contract.validate_fixed_forward_contract(
            ep_size=8,
            hidden_size=4096,
            intermediate_size=1024,
            num_local_experts=64,
            topk=10,
            num_local_tokens=512,
            schedule_capacity=40960,
            macrobatch_size=4096,
            minibatch_size=4096,
            x_ptrs=[0x1000 + i * 0x1000 for i in range(8)],
            combine_ptrs=[0x10000 + i * 0x1000 for i in range(8)],
        )

    def test_rejects_a_nearby_but_unsupported_shape(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "topk=10"):
            contract.validate_fixed_forward_contract(
                ep_size=8,
                hidden_size=4096,
                intermediate_size=1024,
                num_local_experts=64,
                topk=6,
                num_local_tokens=512,
                schedule_capacity=40960,
                macrobatch_size=4096,
                minibatch_size=4096,
                x_ptrs=[0x1000] * 8,
                combine_ptrs=[0x2000] * 8,
            )


if __name__ == "__main__":
    unittest.main()
