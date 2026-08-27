from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_HOST_PATH = _ROOT / "mok" / "cutedsl" / "persistent_bf16.py"
_GEMM_PATH = _ROOT / "mok" / "cutedsl" / "_persistent_bf16_gemm.py"
_HOST_SOURCE = _HOST_PATH.read_text()
_GEMM_SOURCE = _GEMM_PATH.read_text()

_SPEC = importlib.util.spec_from_file_location("mok_fc1_slice_contract", _HOST_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HOST = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HOST
_SPEC.loader.exec_module(_HOST)


class TestPersistentBf16Fc1Slice(unittest.TestCase):
    def test_fixed_cuda_shaped_geometry(self) -> None:
        self.assertEqual(_HOST.FC1_TILE_M, 256)
        self.assertEqual(_HOST.FC1_LOGICAL_N, 128)
        self.assertEqual(_HOST.FC1_PACKED_N, 256)
        self.assertEqual(_HOST.FC1_TILE_K, 4096)
        self.assertEqual(_HOST.CLUSTER_SHAPE, (2, 1, 1))
        self.assertTrue(_HOST.FC1_SLICE_SOURCE_BODY_PRESENT)
        self.assertFalse(_HOST.FULL_PERSISTENT_FORWARD_COMPLETE)
        _HOST.Fc1SlicePlan().validate()

    def test_plan_rejects_generalization(self) -> None:
        with self.assertRaises(NotImplementedError):
            _HOST.Fc1SlicePlan(k=1024).validate()

    def test_host_rejects_unproven_alignment_before_assumed_align(self) -> None:
        self.assertIn("tensor.data_ptr() % 16", _HOST_SOURCE)
        self.assertIn("data pointer must be 16-byte aligned", _HOST_SOURCE)
        self.assertIn("from_dlpack(x, assumed_align=16)", _GEMM_SOURCE)

    def test_device_body_has_separate_rank_local_operands(self) -> None:
        self.assertIn("gA = thr_mma.partition_A(gA_cluster)", _GEMM_SOURCE)
        self.assertIn("b_rank0 = tiled_mma.get_slice(Int32(0))", _GEMM_SOURCE)
        self.assertIn("gB_gate = b_rank0.partition_B", _GEMM_SOURCE)
        self.assertIn("gB_up = b_rank0.partition_B", _GEMM_SOURCE)
        self.assertIn("[copy_A, copy_B_gate]", _GEMM_SOURCE)
        self.assertIn("[copy_A, copy_B_up]", _GEMM_SOURCE)
        self.assertNotIn("tma_multicast=", _GEMM_SOURCE)
        self.assertNotIn("packed_gate_up_weights", _GEMM_SOURCE)

    def test_mma_aware_tma_helpers_preserve_rest_k_mode(self) -> None:
        self.assertIn("cute.nvgpu.make_tiled_tma_atom_A(", _GEMM_SOURCE)
        self.assertEqual(
            _GEMM_SOURCE.count("cute.nvgpu.make_tiled_tma_atom_B("),
            2,
        )
        self.assertNotIn("cpasync.make_tiled_tma_atom(\n            tma_op", _GEMM_SOURCE)

    def test_device_body_reaches_bf16_smem_and_gmem_handoff(self) -> None:
        self.assertIn("collective.mma(", _GEMM_SOURCE)
        self.assertIn("epilog_tmem_copy_and_partition", _GEMM_SOURCE)
        self.assertIn("rD_bf16 = tRS_rD.to(BFloat16)", _GEMM_SOURCE)
        self.assertIn("copy_D(src_idx=epi_buffer, dst_idx=epi_coord)", _GEMM_SOURCE)
        self.assertNotIn("raise NotImplementedError", _GEMM_SOURCE)

    def test_compile_slice_is_private_and_has_no_fallback(self) -> None:
        tree = ast.parse(_GEMM_SOURCE)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("mok.functional", imports)
        self.assertNotIn("mok.cutedsl.forward", imports)
        self.assertNotIn("run_comm_cta", _GEMM_SOURCE)
        self.assertIn('cute.compile(_FC1_SLICE, *cute_args)', _GEMM_SOURCE)

    def test_run_compiles_then_calls_executor_with_identical_args(self) -> None:
        tree = ast.parse(_GEMM_SOURCE)
        run = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_fc1_slice"
        )
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        compile_call = next(
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "cute"
            and call.func.attr == "compile"
        )
        execute_call = next(
            call
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id == "executor"
        )
        for call in (compile_call, execute_call):
            starred = [arg for arg in call.args if isinstance(arg, ast.Starred)]
            self.assertEqual(len(starred), 1)
            self.assertEqual(starred[0].value.id, "cute_args")
        self.assertEqual(
            sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "executor"
                for node in ast.walk(run)
            ),
            1,
        )

    def test_compile_and_run_share_one_argument_builder(self) -> None:
        tree = ast.parse(_GEMM_SOURCE)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in ("compile_fc1_slice", "run_fc1_slice"):
            helper_calls = [
                node
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_make_fc1_slice_args"
            ]
            self.assertEqual(len(helper_calls), 1)

    def test_run_rejects_bad_host_tensors_before_cuda_gate(self) -> None:
        capability = mock.Mock(return_value=(10, 3))
        fake_torch = SimpleNamespace(
            Tensor=type("FakeTensor", (), {}),
            bfloat16=object(),
            cuda=SimpleNamespace(get_device_capability=capability),
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(TypeError, "x must be a torch.Tensor"):
                _HOST.run_fc1_slice(object(), object(), object(), object())
        capability.assert_not_called()

    def test_run_rejects_non_sm103_before_importing_device_body(self) -> None:
        x = SimpleNamespace(device="cuda:0")
        capability = mock.Mock(return_value=(9, 0))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=capability)
        )
        with (
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
            mock.patch.object(_HOST, "validate_fc1_slice_tensors") as validate,
        ):
            with self.assertRaisesRegex(NotImplementedError, "B300/SM103"):
                _HOST.run_fc1_slice(x, object(), object(), object())
        validate.assert_called_once()
        capability.assert_called_once_with("cuda:0")

    def test_exact_dependency_versions_are_visible(self) -> None:
        self.assertIn('_REQUIRED_CUTLASS_DSL = "4.6.2"', _GEMM_SOURCE)
        self.assertIn('_REQUIRED_QUACK = "0.6.4"', _GEMM_SOURCE)


if __name__ == "__main__":
    unittest.main()
