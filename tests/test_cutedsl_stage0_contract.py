from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "mok"
    / "cutedsl"
    / "experimental"
    / "contract.py"
)
_SPEC = importlib.util.spec_from_file_location("mok_cutedsl_stage0_contract", _CONTRACT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


class Stage0ContractTest(unittest.TestCase):
    def test_dynamic_peer_reference(self) -> None:
        peers = [
            [[float(100 * rank + 10 * row + col) for col in range(4)] for row in range(3)]
            for rank in range(contract.EP_SIZE)
        ]
        ranks = [7, 0, 3, 4]
        rows = [2, 1, 0, 2]
        self.assertEqual(
            contract.reference_dispatch_rows(peers, ranks, rows),
            [peers[7][2], peers[0][1], peers[3][0], peers[4][2]],
        )

    def test_qwen_ep8_contract(self) -> None:
        contract.validate_stage0_contract(
            [0x1000 + rank * 0x1000 for rank in range(contract.EP_SIZE)],
            [7, 0, 3, 4],
            [2, 1, 0, 2],
            num_peer_tokens=3,
        )

    def test_contract_rejects_wrong_ep_and_odd_schedule(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            contract.validate_stage0_contract(
                [0x1000] * 4,
                [0, 1],
                [0, 0],
                num_peer_tokens=1,
            )
        with self.assertRaisesRegex(ValueError, "positive even"):
            contract.validate_stage0_contract(
                [0x1000] * contract.EP_SIZE,
                [0],
                [0],
                num_peer_tokens=1,
            )

    def test_mixed_clc_role_contract(self) -> None:
        self.assertEqual(
            contract.expected_role_log(num_comm_clusters=2, num_compute_clusters=3),
            [100, 101, 100, 101, 200, 201, 200, 201, 200, 201],
        )


if __name__ == "__main__":
    unittest.main()
