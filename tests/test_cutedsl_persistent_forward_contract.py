from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "mok" / "cutedsl"

# Load the dependency-free modules without importing mok.cutedsl.__init__ or
# the CUDA extension.  The relative import in persistent_forward_contract is
# preserved by installing a minimal package shell in sys.modules.
_PACKAGE_NAME = "mok_cutedsl_contract_tests"
_PACKAGE = types.ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_PACKAGE_ROOT)]
sys.modules.setdefault(_PACKAGE_NAME, _PACKAGE)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{name}",
        _PACKAGE_ROOT / filename,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load("forward_contract", "forward_contract.py")
contract = _load("persistent_forward_contract", "persistent_forward_contract.py")


class CuTeDSLPersistentForwardContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = contract.make_forward_geometry(
            num_local_tokens=2048,
            schedule_capacity=16384,
            macrobatch_size=4096,
            minibatch_size=1024,
            num_comm_sms=40,
        )

    def test_qwen_task_and_counter_geometry_matches_cuda(self) -> None:
        geometry = self.geometry
        self.assertEqual(geometry.comm_clusters, 20)
        self.assertEqual(geometry.shared_gate_up_tasks, 32)
        self.assertEqual(geometry.shared_swiglu_tasks, 22)
        self.assertEqual(geometry.shared_down_tasks, 128)
        self.assertEqual(geometry.shared_tasks, 214)
        self.assertEqual(geometry.minibatch_routed_gate_up_tasks, 16)
        self.assertEqual(geometry.minibatch_routed_swiglu_tasks, 11)
        self.assertEqual(geometry.minibatch_routed_down_tasks, 64)
        self.assertEqual(geometry.minibatch_tasks, 107)
        self.assertEqual(
            geometry.counters,
            contract.CounterShapes(
                gate_up_tile_ready=288,
                hidden_row_block_ready=72,
                x_routed_ready=16,
                y_routed_ready=16,
                y_routed_done=128,
            ),
        )

    def test_repository_default_qwen_fixture_matches_cuda(self) -> None:
        geometry = contract.qwen_default_geometry()
        self.assertEqual(geometry.num_local_tokens, 7168)
        self.assertEqual(geometry.schedule_capacity, 286720)
        self.assertEqual(geometry.macrobatch_size, 131072)
        self.assertEqual(geometry.minibatch_size, 4096)
        self.assertEqual(geometry.shared_gate_up_tasks, 112)
        self.assertEqual(geometry.shared_swiglu_tasks, 75)
        self.assertEqual(geometry.shared_down_tasks, 448)
        self.assertEqual(geometry.shared_tasks, 747)
        self.assertEqual(geometry.minibatch_routed_gate_up_tasks, 64)
        self.assertEqual(geometry.minibatch_routed_swiglu_tasks, 43)
        self.assertEqual(geometry.minibatch_routed_down_tasks, 256)
        self.assertEqual(geometry.minibatch_tasks, 427)
        self.assertEqual(
            geometry.counters,
            contract.CounterShapes(4592, 1148, 70, 70, 2240),
        )

    def test_capacity_launch_envelope_is_not_runtime_true_grid(self) -> None:
        geometry = self.geometry
        self.assertEqual(geometry.capacity_num_minibatches, 16)
        self.assertEqual(geometry.capacity_compute_clusters, 1926)
        self.assertEqual(geometry.capacity_launch_clusters, 1946)
        self.assertEqual(geometry.capacity_launch_ctas, 3892)
        self.assertEqual(geometry.true_num_minibatches(9472), 10)
        self.assertEqual(geometry.true_runtime_compute_clusters(9472), 1284)
        self.assertEqual(geometry.true_runtime_clusters(9472), 1304)
        self.assertEqual(geometry.true_runtime_ctas(9472), 2608)
        self.assertTrue(geometry.is_runtime_active_cluster(1303, 9472))
        self.assertFalse(geometry.is_runtime_active_cluster(1304, 9472))

    def test_fixed_comm_prefix_never_decodes_as_clc_compute(self) -> None:
        num_tokens = 9472
        self.assertTrue(self.geometry.is_fixed_comm_cluster(0))
        self.assertTrue(self.geometry.is_fixed_comm_cluster(19))
        self.assertFalse(self.geometry.is_fixed_comm_cluster(20))
        self.assertEqual(self.geometry.comm_cta_index(19, 1), 39)
        self.assertEqual(
            self.geometry.physical_cta(19, 1),
            contract.PhysicalCTA(contract.PhysicalRole.FIXED_COMM, 19, 1, 39),
        )
        self.assertEqual(
            self.geometry.physical_cta(20, 0),
            contract.PhysicalCTA(contract.PhysicalRole.CLC_COMPUTE, 20, 0, None),
        )
        with self.assertRaisesRegex(ValueError, "do not enter"):
            self.geometry.decode_cluster_task(19, num_tokens)
        first_compute = self.geometry.decode_cluster_task(20, num_tokens)
        self.assertEqual(first_compute.kind, contract.ForwardTaskKind.SHARED_GATE)
        self.assertEqual(first_compute.task_index, 0)

    def test_physical_task_shapes_are_cuda_shaped(self) -> None:
        for kind in (
            contract.ForwardTaskKind.SHARED_GATE,
            contract.ForwardTaskKind.SHARED_UP,
            contract.ForwardTaskKind.SHARED_DOWN,
            contract.ForwardTaskKind.ROUTED_GATE,
            contract.ForwardTaskKind.ROUTED_UP,
            contract.ForwardTaskKind.ROUTED_DOWN,
        ):
            with self.subTest(kind=kind):
                self.assertEqual(
                    contract.task_physical_shape(kind),
                    contract.PhysicalTaskShape((256, 256), (128, 256), 1, True),
                )
        self.assertEqual(
            contract.task_physical_shape(contract.ForwardTaskKind.ROUTED_SWIGLU),
            contract.PhysicalTaskShape((128, 128), (128, 128), 3, False),
        )
        self.assertEqual(
            contract.task_physical_shape("dispatch"),
            contract.PhysicalTaskShape((128, 512), (128, 512), 1, False),
        )
        self.assertEqual(
            contract.task_physical_shape("combine"),
            contract.PhysicalTaskShape((16, 1024), (16, 1024), 7, False),
        )

    def test_shared_task_ladder_has_cuda_boundaries(self) -> None:
        geometry = self.geometry
        num_tokens = 9472
        boundaries = (
            (0, contract.ForwardTaskKind.SHARED_GATE, 0),
            (31, contract.ForwardTaskKind.SHARED_GATE, 31),
            (32, contract.ForwardTaskKind.SHARED_UP, 0),
            (63, contract.ForwardTaskKind.SHARED_UP, 31),
            (64, contract.ForwardTaskKind.SHARED_SWIGLU, 0),
            (85, contract.ForwardTaskKind.SHARED_SWIGLU, 21),
            (86, contract.ForwardTaskKind.SHARED_DOWN, 0),
            (213, contract.ForwardTaskKind.SHARED_DOWN, 127),
        )
        for index, kind, task_index in boundaries:
            with self.subTest(index=index):
                task = geometry.decode_compute_task(index, num_tokens)
                self.assertEqual(task.kind, kind)
                self.assertEqual(task.task_index, task_index)
                self.assertIsNone(task.macrobatch_index)
                self.assertIsNone(task.minibatch_index)

    def test_routed_minibatches_use_reverse_macro_order(self) -> None:
        geometry = self.geometry
        num_tokens = 9472  # macros contain 4096, 4096, and 1280 rows
        self.assertEqual(geometry.num_macrobatches(num_tokens), 3)
        self.assertEqual(geometry.true_num_minibatches(num_tokens), 10)

        decoded = []
        for ordered_minibatch in range(10):
            compute_index = geometry.shared_tasks + ordered_minibatch * geometry.minibatch_tasks
            task = geometry.decode_compute_task(compute_index, num_tokens)
            self.assertEqual(task.kind, contract.ForwardTaskKind.ROUTED_GATE)
            self.assertEqual(task.task_index, 0)
            decoded.append((task.macrobatch_index, task.minibatch_index))
        self.assertEqual(
            decoded,
            [
                (2, 0),
                (2, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
            ],
        )

    def test_routed_stage_ladder_preserves_two_cta_task_index(self) -> None:
        geometry = self.geometry
        num_tokens = 4096
        base = geometry.shared_tasks
        boundaries = (
            (0, contract.ForwardTaskKind.ROUTED_GATE, 0),
            (15, contract.ForwardTaskKind.ROUTED_GATE, 15),
            (16, contract.ForwardTaskKind.ROUTED_UP, 0),
            (31, contract.ForwardTaskKind.ROUTED_UP, 15),
            (32, contract.ForwardTaskKind.ROUTED_SWIGLU, 0),
            (42, contract.ForwardTaskKind.ROUTED_SWIGLU, 10),
            (43, contract.ForwardTaskKind.ROUTED_DOWN, 0),
            (106, contract.ForwardTaskKind.ROUTED_DOWN, 63),
        )
        for offset, kind, task_index in boundaries:
            with self.subTest(offset=offset):
                task = geometry.decode_compute_task(base + offset, num_tokens)
                self.assertEqual(task.kind, kind)
                self.assertEqual(task.task_index, task_index)
                self.assertEqual(task.macrobatch_index, 0)
                self.assertEqual(task.minibatch_index, 0)

    def test_five_monotonic_counter_thresholds_match_cuda(self) -> None:
        self.assertEqual(
            contract.counter_required_counts(1024),
            contract.CounterShapes(
                gate_up_tile_ready=4,
                hidden_row_block_ready=16,
                x_routed_ready=64,
                y_routed_ready=128,
                y_routed_done=32,
            ),
        )
        self.assertEqual(
            contract.counter_required_counts(256).x_routed_ready,
            16,
        )
        self.assertEqual(
            contract.counter_required_counts(256).y_routed_ready,
            32,
        )
        qwen_required = contract.counter_required_counts(4096)
        self.assertEqual(qwen_required, contract.CounterShapes(4, 16, 256, 512, 32))

    def test_counter_indices_bases_and_producer_consumer_edges(self) -> None:
        geometry = self.geometry
        self.assertEqual(
            geometry.gate_up_counter_index(
                is_shared=True,
                global_row_block=7,
                column_block=3,
            ),
            31,
        )
        self.assertEqual(
            geometry.gate_up_counter_index(
                is_shared=False,
                global_row_block=5,
                column_block=2,
            ),
            54,
        )
        self.assertEqual(
            geometry.hidden_counter_index(is_shared=True, global_row_block=7),
            7,
        )
        self.assertEqual(
            geometry.hidden_counter_index(is_shared=False, global_row_block=5),
            13,
        )
        self.assertEqual(geometry.minibatch_counter_index(4096), 4)
        self.assertEqual(geometry.y_done_counter_index(4096), 32)

        edges = contract.counter_contracts(geometry, 1024)
        self.assertEqual([edge.name for edge in edges], list(contract.CounterName))
        self.assertEqual([edge.required_arrivals for edge in edges], [4, 16, 64, 128, 32])
        self.assertIn("Gate and Up", edges[0].producer)
        self.assertIn("prior-macro Dispatch", edges[3].consumer)
        self.assertIn("Down epilogue", edges[4].consumer)

    def test_clc_depth1_phase_sync_and_drain_contract(self) -> None:
        self.assertEqual(contract.CLC_PIPE_DEPTH, 1)
        self.assertEqual(contract.CLC_RESPONSE_BYTES, 16)
        self.assertEqual(contract.CLC_SCHEDULER_WARP, 5)
        self.assertEqual(contract.CLC_COMPLETION_WARPS, 16)
        self.assertEqual(contract.CLC_DRAIN_WARPS, 8)
        self.assertEqual(contract.CLC_DRAIN_PIPE_DEPTH, 8)
        self.assertTrue(contract.CLC_TERMINAL_CLUSTER_SYNC_REQUIRED)
        self.assertEqual(contract.clc_logical_cluster_index(42), 21)
        self.assertEqual(len(contract.clc_response_lifecycle()), 6)
        self.assertIn("sixteen_warps", contract.clc_response_lifecycle()[4])
        self.assertTrue(
            contract.needs_cluster_sync_before(
                contract.ForwardTaskKind.ROUTED_SWIGLU,
                contract.ForwardTaskKind.ROUTED_DOWN,
            )
        )
        self.assertFalse(
            contract.needs_cluster_sync_before(
                contract.ForwardTaskKind.ROUTED_GATE,
                contract.ForwardTaskKind.ROUTED_UP,
            )
        )
        self.assertTrue(
            contract.needs_clc_drain(
                response_succeeded=True,
                response_cluster_index=1304,
                true_runtime_clusters=1304,
            )
        )
        self.assertFalse(
            contract.needs_clc_drain(
                response_succeeded=False,
                response_cluster_index=-1,
                true_runtime_clusters=1304,
            )
        )

    def test_multi_macro_ring_reuse_uses_distinct_global_generations(self) -> None:
        simulation = contract.simulate_ring_reuse(self.geometry, 9472)
        self.assertEqual(simulation.reverse_macrobatches, (2, 1, 0))
        self.assertTrue(simulation.arrivals_satisfy_waits)
        self.assertTrue(simulation.generations_are_distinct)
        self.assertEqual(len(simulation.dependencies), 84)

        x_slot_zero = [
            edge
            for edge in simulation.dependencies
            if edge.buffer == "x_routed" and edge.local_row_start == 0
        ]
        self.assertEqual([edge.counter_index for edge in x_slot_zero], [8, 4])
        self.assertTrue(
            all(edge.counter == contract.CounterName.Y_ROUTED_READY for edge in x_slot_zero)
        )
        y_slot_zero = [
            edge
            for edge in simulation.dependencies
            if edge.buffer == "y_routed" and edge.local_row_start == 0
        ]
        self.assertEqual([edge.counter_index for edge in y_slot_zero], [64, 32])
        self.assertTrue(
            all(edge.counter == contract.CounterName.Y_ROUTED_DONE for edge in y_slot_zero)
        )

    def test_quack_pipeline_constants_are_not_conflated(self) -> None:
        self.assertEqual(contract.MLP_BF16_K_TILE, 64)
        self.assertEqual(contract.MLP_LOAD_PIPE_DEPTH, 6)
        self.assertEqual(contract.MLP_EPILOGUE_SUBTILES, 8)
        self.assertEqual(contract.MLP_OUTPUT_SMEM_RING, 3)
        self.assertNotEqual(
            contract.MLP_EPILOGUE_SUBTILES,
            contract.MLP_OUTPUT_SMEM_RING,
        )

    def test_outer_nine_tensor_abi_shapes_match_cuda_bf16(self) -> None:
        self.assertEqual(
            contract.nine_output_shapes(2048, 4096),
            (
                (4096, 4096),
                (2048, 1024),
                (4096, 1024),
                (2048, 1024),
                (4096, 1024),
                (2048, 1024),
                (4096, 1024),
                (2048, 4096),
                (4096, 4096),
            ),
        )

    def test_rejects_non_cuda_aligned_runtime_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_tokens must be divisible"):
            self.geometry.true_runtime_compute_clusters(257)
        with self.assertRaisesRegex(ValueError, "minibatch_size must be divisible"):
            contract.make_forward_geometry(
                num_local_tokens=2048,
                schedule_capacity=16384,
                macrobatch_size=4096,
                minibatch_size=384,
                num_comm_sms=40,
            )


if __name__ == "__main__":
    unittest.main()
