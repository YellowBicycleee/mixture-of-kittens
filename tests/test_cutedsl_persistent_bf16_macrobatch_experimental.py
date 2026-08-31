from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "mok/cutedsl/_persistent_bf16_macrobatch_experimental.py"
BENCH = ROOT / "benchmarks/cutedsl/bench_qwen_fwd_macrobatch_experimental.py"
FROZEN = {
    "mok/functional.py": "a1a1b5f392b8c42223d1e183a801c7c4f31941d806861de9c6af359b74714c2b",
    "mok/cutedsl/persistent_bf16.py": "fae66138682e1577e8a8760c4dedff2ba7d80244c21794f07651c86203e1084c",
    "mok/cutedsl/_persistent_bf16_mega.py": "9eadbf44eb775f470cda000072a233168a931011c57022e1ffe909b24d92b3ce",
    "csrc/megakernel/forward.cuh": "8809cff7fe2e4cac59e9cf8cde717f77b4ee0741bc0766994133a3dfd9745e81",
    "csrc/megakernel/entrypoints.cuh": "ba7cf15ba6c21b2787c522d73ee471fe54d48107649635653d5a2332acdece6b",
    "benchmarks/cutedsl/bench_qwen_fwd_ab.py": "51af8a77f0883074cd99827fb417d78ffad6eab48abb107fe1c851a9f7713b2b",
}
MODULE_SOURCE = MODULE.read_text()
BENCH_SOURCE = BENCH.read_text()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = tuple(shape)
        self.dtype = dtype

    def stride(self):
        result, value = [], 1
        for extent in reversed(self.shape):
            result.append(value)
            value *= extent
        return tuple(reversed(result))


class ExperimentalMacrobatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package_name = "mok_cutedsl_macro_test_package"
        package = types.ModuleType(package_name)
        package.__path__ = [str(MODULE.parent)]
        sys.modules[package_name] = package
        accepted_name = f"{package_name}.persistent_bf16"
        accepted_spec = importlib.util.spec_from_file_location(
            accepted_name, ROOT / "mok/cutedsl/persistent_bf16.py"
        )
        accepted = importlib.util.module_from_spec(accepted_spec)
        sys.modules[accepted_name] = accepted
        accepted_spec.loader.exec_module(accepted)
        module_name = f"{package_name}._persistent_bf16_macrobatch_experimental"
        module_spec = importlib.util.spec_from_file_location(module_name, MODULE)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        cls.package_names = (module_name, accepted_name, package_name)
        cls.module = module

    @classmethod
    def tearDownClass(cls) -> None:
        for name in cls.package_names:
            sys.modules.pop(name, None)

    def tearDown(self) -> None:
        self.module.clear_experimental_macrobatch_cache(synchronize=False)

    def test_frozen_sources_and_private_scope(self) -> None:
        for relative, expected in FROZEN.items():
            self.assertEqual(digest(ROOT / relative), expected)
        self.assertNotIn("_persistent_bf16_macrobatch_experimental", (ROOT / "mok/functional.py").read_text())
        self.assertNotIn("_persistent_bf16_macrobatch_experimental", (ROOT / "mok/cutedsl/__init__.py").read_text())
        tree = ast.parse(MODULE_SOURCE)
        decorators = [ast.unparse(item) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) for item in node.decorator_list]
        self.assertFalse(any("cute.kernel" in item or "cute.jit" in item for item in decorators))
        self.assertNotIn("torch.cat", MODULE_SOURCE)

    def test_bounded_plans_and_ring_shapes(self) -> None:
        self.assertEqual(self.module.MACROBATCH_CANDIDATES, (4096, 8192, 16384, 32768))
        for macro in self.module.MACROBATCH_CANDIDATES:
            self.module.ExperimentalMacrobatchPlan(macro).validate()
            shapes = self.module._output_shapes(102400, macro)
            self.assertEqual(shapes[0], (macro, 4096))
            self.assertEqual(shapes[2:7:2], ((macro, 1024),) * 3)
            self.assertEqual(shapes[-1], (macro, 4096))
        for macro in (0, 12288, 65536):
            with self.assertRaises(NotImplementedError):
                self.module.ExperimentalMacrobatchPlan(macro).validate()

    def test_cache_key_is_geometry_not_address_or_stream(self) -> None:
        tensors = [FakeTensor((2, 2)) for _ in range(6 + 6 + 9 + 5)]
        workspace = SimpleNamespace(
            x_buffer=tensors[0], combine_buffer=tensors[1],
            num_local_tokens=102400, schedule_capacity=4096000,
        )
        schedule = SimpleNamespace(
            peer_rank=tensors[2], peer_token_idx=tensors[3],
            num_tokens=tensors[4], tokens_per_expert=tensors[5],
        )
        weights = tuple(tensors[6:12])
        state = SimpleNamespace(
            abi_outputs=tuple(tensors[12:21]), counters=tuple(tensors[21:26])
        )
        kwargs = dict(runtime_environment=(("dsl", "4.6.2"),), device_index=0, context_key=7)
        key4 = self.module._executor_key(
            workspace, schedule, weights, state,
            self.module.ExperimentalMacrobatchPlan(4096), **kwargs,
        )
        key8 = self.module._executor_key(
            workspace, schedule, weights, state,
            self.module.ExperimentalMacrobatchPlan(8192), **kwargs,
        )
        self.assertNotEqual(key4, key8)
        self.assertNotIn("stream", repr(key4).lower())
        self.assertNotIn("address", repr(key4).lower())

    def test_cached_executor_rebinds_runtime_args(self) -> None:
        calls = {"prepare": 0, "make": 0}
        executor = object()

        def prepare(*args, **kwargs):
            calls["prepare"] += 1
            return executor, ("first", kwargs["stream"])

        def make(*args, **kwargs):
            calls["make"] += 1
            return ("dynamic", "constexpr", kwargs["stream"])

        with mock.patch.object(
            self.module._accepted,
            "_load_mega_runtime",
            return_value=(make, prepare, 1),
        ):
            first = self.module._cached_launch(("B4",), 0, (), {"stream": "s0"})
            second = self.module._cached_launch(("B4",), 0, (), {"stream": "s1"})
        self.assertIs(first[0], executor)
        self.assertIs(second[0], executor)
        self.assertEqual(first[1], ("first", "s0"))
        self.assertEqual(second[1], ("dynamic", "s1"))
        self.assertEqual(calls, {"prepare": 1, "make": 1})

    def test_fresh_state_and_bounded_abba_protocol(self) -> None:
        forward = ast.get_source_segment(
            MODULE_SOURCE,
            next(node for node in ast.parse(MODULE_SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == "forward_bf16"),
        )
        self.assertLess(forward.index("state = _prepare_state("), forward.index("_cached_launch("))
        self.assertIn("_record_public_launch_owners(", forward)
        self.assertIn("cuda_a =", BENCH_SOURCE)
        self.assertIn("cuda_b =", BENCH_SOURCE)
        self.assertLess(BENCH_SOURCE.index("cuda_b ="), BENCH_SOURCE.index("experimental ="))
        self.assertIn("public_b32 = accepted_bench.checked_forward_backward(", BENCH_SOURCE)
        self.assertIn("private_b32 = _candidate_leg(", BENCH_SOURCE)
        self.assertIn('"B32768/public_accepted_vs_private_wrapper"', BENCH_SOURCE)
        self.assertIn('len(b32_private_self_control["records"]) != 16', BENCH_SOURCE)
        self.assertIn('or not b32_private_self_control["pass"]', BENCH_SOURCE)
        self.assertLess(
            BENCH_SOURCE.index('or not b32_private_self_control["pass"]'),
            BENCH_SOURCE.index("accepted_run = lambda:"),
        )
        self.assertIn('"b32_private_self_control": b32_private_self_control', BENCH_SOURCE)
        self.assertIn('"accepted_A": "frozen public CuTe B32768"', BENCH_SOURCE)
        self.assertIn("WARMUP_PAIRS = 2", BENCH_SOURCE)
        self.assertIn("MEASURED_PAIRS = 10", BENCH_SOURCE)
        self.assertIn("torch.cuda.Event(enable_timing=True)", BENCH_SOURCE)
        self.assertIn('"screen_positive":', BENCH_SOURCE)
        self.assertIn('"screen_candidate": screen_candidate', BENCH_SOURCE)
        self.assertIn('"formal_selection": "N/A_requires_W10_N30_and_CV"', BENCH_SOURCE)
        for forbidden in (
            '"stable_positive"',
            '"selected_cutedsl_macrobatch"',
            '"advanced_from_accepted_32768"',
            "diagnostic CuTe macro winner",
        ):
            self.assertNotIn(forbidden, BENCH_SOURCE)
        self.assertNotIn("profiler", BENCH_SOURCE.lower())

    def test_default_cli_is_deterministic_static_only(self) -> None:
        command = [sys.executable, "-B", str(BENCH)]
        first = subprocess.check_output(command, cwd=ROOT, text=True)
        second = subprocess.check_output(command, cwd=ROOT, text=True)
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["timing"]["status"], "N/A")
        self.assertIn("later nodes", payload["decision_scope"])


if __name__ == "__main__":
    unittest.main()
