from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/cutedsl/bench_qwen_fwd_macrobatch_formal.py"
SPEC = importlib.util.spec_from_file_location("mok_fwd_macro_formal_test", BENCH)
FORMAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORMAL)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point(cuda_ms: float, cutedsl_ms: float):
    values = (cuda_ms, cutedsl_ms, cutedsl_ms, cuda_ms)
    blocks = []
    for index, (backend, value) in enumerate(zip(FORMAL.ORDER, values)):
        samples = [value] * FORMAL.SAMPLES
        blocks.append(
            {
                "order_index": index,
                "backend": backend,
                "rank_max_samples_ms": samples,
            }
            | FORMAL._sample_summary(samples)
        )
    return {"blocks": blocks, "summary": FORMAL._summarize_point(blocks)}


class FormalMacrobatchTest(unittest.TestCase):
    def test_static_cli_and_frozen_exact3(self) -> None:
        before = {
            relative: digest(ROOT / relative)
            for relative in FORMAL.EXACT3_SHA256
        }
        first = subprocess.check_output([sys.executable, "-B", str(BENCH)], cwd=ROOT, text=True)
        second = subprocess.check_output([sys.executable, "-B", str(BENCH)], cwd=ROOT, text=True)
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["timing"]["status"], "N/A_local_static_mode")
        self.assertEqual(payload["correctness"]["required_records_per_macrobatch"], 48)
        self.assertEqual(before, FORMAL.EXACT3_SHA256)
        self.assertEqual(
            before,
            {relative: digest(ROOT / relative) for relative in FORMAL.EXACT3_SHA256},
        )

    def test_cuda_self_control_precedes_private_leg(self) -> None:
        source = BENCH.read_text()
        correctness = source[source.index("def _correctness_point"):source.index("def _measure_block")]
        self_control = correctness.index('self_control = fixture._compare_legs(')
        fail_closed = correctness.index('raise AssertionError(f"B={macro} CUDA self-control failed")')
        private_leg = correctness.index('cutedsl = fixture._candidate_leg(')
        self.assertLess(self_control, fail_closed)
        self.assertLess(fail_closed, private_leg)

    def test_stability_and_direction_classification(self) -> None:
        self.assertEqual(point(10.0, 10.05)["summary"]["classification"], "EQUIVALENT")
        self.assertEqual(point(10.0, 9.5)["summary"]["classification"], "CUTEDSL_FASTER")
        self.assertEqual(point(10.0, 10.5)["summary"]["classification"], "CUDA_FASTER")

        conflicting = point(10.0, 9.5)
        conflicting["blocks"][2] = {
            "order_index": 2,
            "backend": "cutedsl",
            "rank_max_samples_ms": [10.5] * FORMAL.SAMPLES,
        } | FORMAL._sample_summary([10.5] * FORMAL.SAMPLES)
        summary = FORMAL._summarize_point(conflicting["blocks"])
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["classification"], "INCONCLUSIVE")

        noisy = point(10.0, 10.0)
        samples = [9.0, 11.0] * (FORMAL.SAMPLES // 2)
        noisy["blocks"][1] = {
            "order_index": 1,
            "backend": "cutedsl",
            "rank_max_samples_ms": samples,
        } | FORMAL._sample_summary(samples)
        summary = FORMAL._summarize_point(noisy["blocks"])
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["classification"], "INVALID_TIMING_STABILITY")

    def test_own_best_requires_complete_valid_sweep(self) -> None:
        cuda = (10.0, 10.05, 11.0, 12.0)
        cutedsl = (12.0, 11.0, 10.0, 10.09)
        points = {
            str(macro): point(cuda_ms, cutedsl_ms)
            for macro, cuda_ms, cutedsl_ms in zip(
                FORMAL.MACROBATCHES, cuda, cutedsl
            )
        }
        selection = FORMAL._select_own_best(points)
        self.assertEqual(selection["status"], "AVAILABLE")
        self.assertEqual(selection["cuda"]["best_macrobatch"], 4096)
        self.assertEqual(
            selection["cuda"]["co_best_macrobatches_within_1pct"], [4096, 8192]
        )
        self.assertEqual(selection["cutedsl"]["best_macrobatch"], 16384)
        self.assertEqual(
            selection["cutedsl"]["co_best_macrobatches_within_1pct"],
            [16384, 32768],
        )
        points.pop("4096")
        self.assertTrue(FORMAL._select_own_best(points)["status"].startswith("N/A"))


if __name__ == "__main__":
    unittest.main()
