from __future__ import annotations

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


class CuTeDSLForwardContractTest(unittest.TestCase):
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

    def test_route_index_is_divided_only_for_dispatch(self) -> None:
        route_idx = 37 * contract.TOPK + 6
        self.assertEqual(contract.decode_schedule_entry(7, route_idx), (37, route_idx))
        self.assertIsNone(contract.decode_schedule_entry(-1, 0x5A5A5A5A))

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
