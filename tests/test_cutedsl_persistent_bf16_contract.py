from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "mok"
    / "cutedsl"
    / "persistent_bf16_contract.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "mok_cutedsl_persistent_bf16_contract", _CONTRACT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contract
_SPEC.loader.exec_module(contract)


class PersistentBF16ContractTest(unittest.TestCase):
    def test_fixed_qwen_ep8_target_and_task_geometry(self) -> None:
        self.assertEqual(contract.EP_SIZE, 8)
        self.assertEqual(contract.HIDDEN_SIZE, 4096)
        self.assertEqual(contract.INTERMEDIATE_SIZE, 1024)
        self.assertEqual(contract.MACROBATCH_ROWS, 32768)
        self.assertEqual(contract.MINIBATCH_ROWS, 4096)
        self.assertEqual(contract.CLUSTER_SIZE, 2)
        self.assertEqual(contract.MINIBATCHES_PER_MACROBATCH, 8)

        # FC1: 4096/256 M tiles x 1024/128 N tiles.
        self.assertEqual(contract.FC1_ROW_TILES, 16)
        self.assertEqual(contract.FC1_COLUMN_TILES, 8)
        self.assertEqual(contract.FC1_TASKS, 128)
        # Down: 4096/256 M tiles x 4096/256 N tiles.
        self.assertEqual(contract.DOWN_ROW_TILES, 16)
        self.assertEqual(contract.DOWN_COLUMN_TILES, 16)
        self.assertEqual(contract.DOWN_TASKS, 256)

    def test_reverse_macro_order_keeps_forward_order_inside_each_macro(self) -> None:
        # Three full macros plus one whole-minibatch tail macro.
        rows = 25 * contract.MINIBATCH_ROWS
        schedule = contract.reverse_minibatches(rows)
        self.assertEqual(
            [minibatch.generation for minibatch in schedule],
            [24, *range(16, 24), *range(8, 16), *range(0, 8)],
        )
        self.assertEqual(schedule[0].macro_index, 3)
        self.assertEqual(schedule[0].mini_index, 0)
        self.assertEqual(schedule[0].ring_slot, 0)
        self.assertEqual(schedule[-8].row_begin, 0)
        self.assertEqual(schedule[-1].generation, 7)

    def test_aligned_partial_minibatch_tail_is_supported(self) -> None:
        self.assertEqual(len(contract.reverse_minibatches(9 * 4096)), 9)
        self.assertEqual(contract.reverse_minibatches(0), ())
        partial = contract.reverse_minibatches(9 * 4096 + 256)
        self.assertEqual(len(partial), 10)
        self.assertEqual(partial[0], contract.MiniBatch(8, 4096))
        self.assertEqual(partial[1], contract.MiniBatch(9, 256))
        for rows in (1, 255, 257, 9 * 4096 + 1):
            with self.subTest(rows=rows):
                with self.assertRaises(contract.UnsupportedTailError):
                    contract.reverse_minibatches(rows)
        for invalid in (-1, True, 4096.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    contract.reverse_minibatches(invalid)

    def test_ready_counts_lock_cuda_v1_c2_derivations(self) -> None:
        self.assertEqual(
            contract.X_READY_REQUIRED,
            (4096 // 128) * (4096 // 512),
        )
        self.assertEqual(contract.X_READY_REQUIRED, 256)

        # One M256 hidden block gets two M128 CTA-row arrivals for each of
        # the eight N128 FC1 slices: (256/128) * (1024/128) = 16.
        self.assertEqual(
            contract.HIDDEN_READY_REQUIRED,
            contract.CLUSTER_SIZE * (1024 // 128),
        )
        self.assertEqual(contract.HIDDEN_READY_REQUIRED, 16)

        # Each of 256 Down cluster tasks contributes one arrival per CTA.
        self.assertEqual(
            contract.Y_READY_REQUIRED,
            contract.DOWN_TASKS * contract.CLUSTER_SIZE,
        )
        self.assertEqual(contract.Y_READY_REQUIRED, 512)

        # One M128 row block has eight M16 rows x four N1024 columns.
        self.assertEqual(
            contract.Y_DONE_REQUIRED,
            (128 // 16) * (4096 // 1024),
        )
        self.assertEqual(contract.Y_DONE_REQUIRED, 32)

    def test_readiness_specs_have_global_generation_keys(self) -> None:
        specs = contract.readiness_specs(contract.MiniBatch(8))
        by_kind = {}
        for spec in specs:
            by_kind.setdefault(spec.key.kind, []).append(spec)
        self.assertEqual(len(by_kind[contract.X_READY]), 1)
        self.assertEqual(len(by_kind[contract.HIDDEN_READY]), 16)
        self.assertEqual(len(by_kind[contract.Y_READY]), 1)
        self.assertEqual(len(by_kind[contract.Y_DONE]), 32)
        self.assertEqual(
            {spec.required_count for spec in by_kind[contract.HIDDEN_READY]},
            {16},
        )
        self.assertEqual(
            {spec.required_count for spec in by_kind[contract.Y_DONE]},
            {32},
        )

        # Generations 0 and 8 alias ring slot 0 but never alias a readiness ID.
        generation_zero = contract.ReadyKey(contract.X_READY, 0)
        generation_eight = contract.ReadyKey(contract.X_READY, 8)
        self.assertNotEqual(generation_zero, generation_eight)
        self.assertNotEqual(generation_zero.linear_id, generation_eight.linear_id)

        # The data buffers recycle slot 0 at generation 8; the counters do
        # not.  Their indices match CUDA's full schedule-capacity arrays.
        hidden_zero = contract.ReadyKey(contract.HIDDEN_READY, 0, 0)
        hidden_eight = contract.ReadyKey(contract.HIDDEN_READY, 8, 0)
        self.assertEqual(hidden_zero.counter_index, 0)
        self.assertEqual(hidden_eight.counter_index, 8 * 16)
        self.assertNotEqual(hidden_zero.counter_index, hidden_eight.counter_index)
        y_done_zero = contract.ReadyKey(contract.Y_DONE, 0, 0)
        y_done_eight = contract.ReadyKey(contract.Y_DONE, 8, 0)
        self.assertEqual(y_done_zero.counter_index, 0)
        self.assertEqual(y_done_eight.counter_index, 8 * 32)
        self.assertNotEqual(y_done_zero.counter_index, y_done_eight.counter_index)

    def test_every_fc1_claim_precedes_the_first_down_claim(self) -> None:
        schedule = contract.reverse_minibatches(16 * 4096)
        claims = contract.compute_claims(contract.MiniBatch(0), schedule)
        self.assertEqual(len(claims), 128 + 256)
        self.assertTrue(all(claim.role == contract.FC1_ROLE for claim in claims[:128]))
        self.assertTrue(all(claim.role == contract.DOWN_ROLE for claim in claims[128:]))
        self.assertEqual(claims[127].task_index, 127)
        self.assertEqual(claims[128].task_index, 0)

        first_fc1 = claims[0]
        self.assertEqual(first_fc1.waits, (contract.ReadyKey(contract.X_READY, 0),))
        self.assertEqual(
            first_fc1.arrival,
            contract.ReadyKey(contract.HIDDEN_READY, 0, 0),
        )
        self.assertEqual(first_fc1.arrival_count, 2)

    def test_down_waits_for_hidden_and_exact_prior_y_row_blocks(self) -> None:
        # Generation 0 reuses the physical slot previously owned by generation 8.
        schedule = contract.reverse_minibatches(16 * 4096)
        down = contract.compute_claims(contract.MiniBatch(0), schedule)[128:]
        self.assertEqual(
            down[0].waits,
            (
                contract.ReadyKey(contract.HIDDEN_READY, 0, 0),
                contract.ReadyKey(contract.Y_DONE, 8, 0),
                contract.ReadyKey(contract.Y_DONE, 8, 1),
            ),
        )
        self.assertEqual(
            down[16].waits,
            (
                contract.ReadyKey(contract.HIDDEN_READY, 0, 1),
                contract.ReadyKey(contract.Y_DONE, 8, 2),
                contract.ReadyKey(contract.Y_DONE, 8, 3),
            ),
        )
        self.assertEqual(down[0].arrival, contract.ReadyKey(contract.Y_READY, 0))
        self.assertEqual(down[0].arrival_count, 2)

    def test_dispatch_and_combine_are_distinct_comm_roles(self) -> None:
        minibatch = contract.MiniBatch(0)
        schedule = contract.reverse_minibatches(16 * 4096)
        dispatch = contract.comm_claims(
            contract.DISPATCH_ROLE, minibatch, schedule
        )
        combine = contract.comm_claims(
            contract.COMBINE_ROLE, minibatch, schedule
        )
        self.assertEqual(len(dispatch), 256)
        self.assertEqual(len(combine), 1024)
        self.assertEqual({claim.role for claim in dispatch}, {"dispatch"})
        self.assertEqual({claim.role for claim in combine}, {"combine"})
        self.assertEqual(
            dispatch[0].waits,
            (contract.ReadyKey(contract.Y_READY, 8),),
        )
        self.assertEqual(
            combine[0].waits,
            (contract.ReadyKey(contract.Y_READY, 0),),
        )
        with self.assertRaisesRegex(ValueError, "dispatch.*combine"):
            contract.comm_claims("comm", minibatch, schedule)

    def test_y_done_parts_receive_exactly_32_combine_arrivals(self) -> None:
        claims = contract.combine_claims(contract.MiniBatch(7))
        arrivals = {}
        for claim in claims:
            arrivals[claim.arrival] = arrivals.get(claim.arrival, 0) + 1
        self.assertEqual(len(arrivals), 32)
        self.assertEqual(set(arrivals.values()), {32})

    def test_partial_tail_uses_dynamic_counts_and_row_major_coordinates(self) -> None:
        tail = contract.MiniBatch(252, 768)
        self.assertEqual(tail.dispatch_tasks, (768 // 128) * 8)
        self.assertEqual(tail.fc1_tasks, (768 // 256) * 8)
        self.assertEqual(tail.down_tasks, (768 // 256) * 16)
        self.assertEqual(tail.combine_raw_tiles, (768 // 16) * 4)
        self.assertEqual(tail.y_done_parts, 768 // 128)
        specs = contract.readiness_specs(tail)
        by_kind = {}
        for spec in specs:
            by_kind.setdefault(spec.key.kind, []).append(spec)
        self.assertEqual(by_kind[contract.X_READY][0].required_count, 48)
        self.assertEqual(len(by_kind[contract.HIDDEN_READY]), 3)
        self.assertEqual(by_kind[contract.Y_READY][0].required_count, 96)
        self.assertEqual(len(by_kind[contract.Y_DONE]), 6)

        schedule = contract.reverse_minibatches(1032960)
        down = contract.compute_claims(tail, schedule)[tail.fc1_tasks :]
        self.assertEqual(contract.TASK_ORDER, "row_major_global_rows")
        self.assertEqual(
            [(claim.row_tile, claim.column_tile) for claim in down[:18]],
            [(0, column) for column in range(16)] + [(1, 0), (1, 1)],
        )

        # Generation 244 reuses slot 4 after generation 252's 768-row tail.
        # Only six M128 row tiles (48 Dispatch tasks) overlap that old slot;
        # CUDA lets every later row start without the prior Y-ready wait.
        by_generation = {mini.generation: mini for mini in schedule}
        current = by_generation[244]
        dispatch = contract.dispatch_claims(current, schedule)
        waiting = [claim for claim in dispatch if claim.waits]
        self.assertEqual(len(waiting), (768 // 128) * 8)
        self.assertEqual({claim.row_tile for claim in waiting}, set(range(6)))
        self.assertTrue(
            all(not claim.waits for claim in dispatch[(768 // 128) * 8 :])
        )

    def test_reuse_guard_rejects_ring_slot_only_aba_keys(self) -> None:
        schedule = contract.reverse_minibatches(16 * 4096)
        guards = contract.reuse_guards(contract.MiniBatch(0), schedule)
        self.assertEqual(len(guards), 1 + 32)
        self.assertEqual({guard.ring_slot for guard in guards}, {0})
        self.assertEqual(guards[0].wait, contract.ReadyKey(contract.Y_READY, 8))
        self.assertEqual(
            contract.reuse_guards(contract.MiniBatch(8), schedule), ()
        )

        # Naming generation 0 because it shares slot 0 is an ABA bug: the
        # release must carry the old owner's full generation, 8.
        with self.assertRaises(contract.ABAHazardError):
            contract.ReuseGuard(
                "x", 0, 8, contract.ReadyKey(contract.Y_READY, 0)
            )
        with self.assertRaises(contract.ABAHazardError):
            contract.ReuseGuard(
                "y", 0, 16, contract.ReadyKey(contract.Y_DONE, 16, 0)
            )

    def test_accepted_100k_rank_rows_validate_unique_exact_contracts(self) -> None:
        rows_by_rank = (
            1032192,
            1032960,
            1030400,
            1032192,
            1032704,
            1031424,
            1031936,
            1030912,
        )
        expected_tail_rows = (4096, 768, 2304, 4096, 512, 3328, 3840, 2816)
        self.assertEqual(len(rows_by_rank), len(expected_tail_rows))
        for rank, (rows, tail_rows) in enumerate(
            zip(rows_by_rank, expected_tail_rows)
        ):
            with self.subTest(rank=rank):
                schedule = contract.validate_contract(rows)
                self.assertEqual(schedule[0].macro_index, 31)
                by_generation = {mini.generation: mini for mini in schedule}
                last_generation = max(by_generation)
                self.assertEqual(
                    by_generation[last_generation].valid_rows,
                    tail_rows,
                )
                specs = [
                    spec
                    for minibatch in schedule
                    for spec in contract.readiness_specs(minibatch)
                ]
                self.assertEqual(len({spec.key for spec in specs}), len(specs))
                self.assertEqual(
                    len({spec.key.linear_id for spec in specs}), len(specs)
                )


if __name__ == "__main__":
    unittest.main()
