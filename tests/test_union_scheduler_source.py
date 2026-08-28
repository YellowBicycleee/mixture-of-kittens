import pathlib
import unittest

from tests.union_schedule_reference import build_union_schedule_reference


ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "csrc" / "union_scheduler.cuh"
OPS = ROOT / "mok" / "union_ops.py"
BINDINGS = ROOT / "csrc" / "bindings.cu"


class UnionScheduleReferenceTests(unittest.TestCase):
    def test_routes_with_the_same_rank_token_share_one_compact_id(self) -> None:
        # EP4, two local experts on rank0.  Peer0/token0 reaches both local
        # experts and therefore contributes two routes but one union row.
        topk_all = [
            [[0, 1, 6], [0, 5, 7], [2, 3, 4], [6, 7, 5]],
            [[1, 0, 6], [2, 4, 6], [3, 5, 7], [0, 4, 6]],
            [[4, 5, 6], [0, 1, 7], [2, 4, 6], [3, 5, 7]],
            [[6, 7, 5], [1, 3, 4], [0, 2, 6], [4, 5, 7]],
        ]
        result = build_union_schedule_reference(
            topk_all, num_local_experts=2, schedule_capacity=512, rank=0
        )

        valid_rows = [i for i, peer in enumerate(result.peer_rank) if peer >= 0]
        self.assertEqual(result.num_tokens, 512)
        self.assertEqual(result.tokens_per_expert, [256, 256])
        self.assertEqual(
            sorted({result.route_to_union[i] for i in valid_rows}),
            list(range(result.num_union)),
        )

        matching = [
            result.route_to_union[i]
            for i in valid_rows
            if result.peer_rank[i] == 0 and result.peer_token_idx[i] // 3 == 0
        ]
        self.assertEqual(len(matching), 2)
        self.assertEqual(len(set(matching)), 1)
        self.assertTrue(all(result.route_to_union[i] == -1 for i in range(512) if i not in valid_rows))

    def test_union_ids_are_deterministic_dense_key_order(self) -> None:
        topk_all = [
            [[0], [3], [0], [3]],
            [[0], [3], [0], [3]],
            [[3], [0], [3], [0]],
            [[3], [0], [3], [0]],
        ]
        result = build_union_schedule_reference(
            topk_all, num_local_experts=1, schedule_capacity=256, rank=0
        )
        valid = [i for i, peer in enumerate(result.peer_rank) if peer >= 0]
        keys_and_ids = sorted(
            (
                result.peer_rank[i] * 4 + result.peer_token_idx[i],
                result.route_to_union[i],
            )
            for i in valid
        )
        unique = []
        for pair in keys_and_ids:
            if not unique or pair != unique[-1]:
                unique.append(pair)
        self.assertEqual([union for _, union in unique], list(range(result.num_union)))

    def test_empty_destination_has_zero_union_rows(self) -> None:
        topk_all = [[[1] for _ in range(8)] for _ in range(4)]
        result = build_union_schedule_reference(
            topk_all, num_local_experts=1, schedule_capacity=256, rank=0
        )
        self.assertEqual(result.num_tokens, 0)
        self.assertEqual(result.tokens_per_expert, [0])
        self.assertEqual(result.num_union, 0)
        self.assertEqual(set(result.peer_rank), {-1})
        self.assertEqual(set(result.route_to_union), {-1})


class UnionScheduleSourceContractTests(unittest.TestCase):
    def test_cuda_uses_dense_presence_scan_without_owner_metadata(self) -> None:
        source = HEADER.read_text()
        self.assertIn("atomicExch(", source)
        self.assertIn("cub::DeviceScan::ExclusiveSum", source)
        self.assertIn("G.route_to_union[{dst_token_idx}]", source)
        self.assertIn("finalize_num_union_kernel", source)
        for forbidden in (
            "union_load_expert",
            "union_loader_slot",
            "union_key",
            "generation",
        ):
            self.assertNotIn(forbidden, source)

    def test_python_and_binding_expose_exactly_six_outputs(self) -> None:
        ops = OPS.read_text()
        bindings = BINDINGS.read_text()
        self.assertIn('@torch.library.custom_op("mok::union_schedule"', ops)
        self.assertIn("route_to_union", ops)
        self.assertIn("num_union", ops)
        self.assertIn('m.def("union_schedule"', bindings)
        self.assertNotIn("union_load_expert", ops)
        self.assertNotIn("union_loader_slot", ops)


if __name__ == "__main__":
    unittest.main()
