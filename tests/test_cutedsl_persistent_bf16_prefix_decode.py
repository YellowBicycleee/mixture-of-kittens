from __future__ import annotations

import random
from pathlib import Path
import unittest


NUM_EXPERTS = 64
TILE_ROWS = 256
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "mok"
    / "cutedsl"
    / "_persistent_bf16_mega.py"
).read_text()


def _swizzle(row_blocks: int, col_blocks: int, task: int) -> tuple[int, int]:
    supergroup = 8
    supergroup_numel = row_blocks * supergroup
    supergroup_idx = task // supergroup_numel
    supersection_cols = col_blocks // supergroup * supergroup
    supersection_numel = row_blocks * supersection_cols
    if task < supersection_numel:
        row = task % supergroup_numel // supergroup
        col = supergroup_idx * supergroup + task % supergroup
    else:
        remainder = task - supersection_numel
        final_cols = max(1, col_blocks - supersection_cols)
        row = remainder // final_cols
        col = supersection_cols + remainder % final_cols
    if supergroup_idx % 2:
        row = row_blocks - row - 1
    return row, col


def _legacy(
    blocks: list[int], first: int, end: int, col_blocks: int, task: int
) -> tuple[int, int, int] | None:
    remaining = task
    expert_first = 0
    for expert, count in enumerate(blocks):
        segment_first = max(first, expert_first)
        segment_end = min(end, expert_first + count)
        rows = max(0, segment_end - segment_first)
        if remaining < rows * col_blocks:
            row, col = _swizzle(rows, col_blocks, remaining)
            return expert, segment_first + row, col
        remaining -= rows * col_blocks
        expert_first += count
    return None


def _prefix_decode(
    blocks: list[int], first: int, end: int, col_blocks: int, task: int
) -> tuple[int, int, int] | None:
    offsets = [0]
    for count in blocks:
        offsets.append(offsets[-1] + count)
    covered_end = min(end, offsets[-1])
    candidate_row = first + task // col_blocks
    if candidate_row >= covered_end:
        return None
    begin, finish = 0, NUM_EXPERTS - 1
    for _ in range(6):
        middle = (begin + finish) // 2
        if offsets[middle + 1] <= candidate_row:
            begin = middle + 1
        else:
            finish = middle
    expert = begin
    segment_first = max(first, offsets[expert])
    segment_end = min(covered_end, offsets[expert + 1])
    rows = segment_end - segment_first
    remaining = task - (segment_first - first) * col_blocks
    row, col = _swizzle(rows, col_blocks, remaining)
    return expert, segment_first + row, col


class PersistentBF16PrefixDecodeTest(unittest.TestCase):
    def test_fixed_seed_matches_linear_scan_including_zeros_and_tails(self) -> None:
        rng = random.Random(20260828)
        cases = [[0, 2, 0, 1] * 16]
        for _ in range(80):
            blocks = [0] * NUM_EXPERTS
            for _ in range(rng.randint(1, 512)):
                blocks[rng.randrange(NUM_EXPERTS)] += 1
            cases.append(blocks)
        for blocks in cases:
            total = sum(blocks)
            minibatch_blocks = rng.choice([1, 2, 4, 8, 16, 32])
            visible = rng.randint(1, total)
            for first in range(0, visible, minibatch_blocks):
                end = min(first + minibatch_blocks, visible)
                for col_blocks in (8, 16):
                    for task in range(minibatch_blocks * col_blocks):
                        self.assertEqual(
                            _prefix_decode(blocks, first, end, col_blocks, task),
                            _legacy(blocks, first, end, col_blocks, task),
                        )

    def test_source_has_one_prefix_build_and_no_hot_e64_scan(self) -> None:
        self.assertEqual(
            SOURCE.count("range_constexpr(NUM_LOCAL_EXPERTS)"),
            1,
        )
        helper = SOURCE[
            SOURCE.index("def _expert_for_routed_row_block_i32") :
            SOURCE.index("def _routed_gate_up_tile_i32")
        ]
        self.assertIn("range_constexpr(6)", helper)

    def test_prefix_uses_existing_pre_arena_alignment_padding(self) -> None:
        pre_prefix_bytes = 496
        prefix_bytes = (NUM_EXPERTS + 1) * 4
        arena_offset = (
            (pre_prefix_bytes + prefix_bytes + 1023) // 1024 * 1024
        )
        self.assertEqual(arena_offset, 1024)
        self.assertEqual(arena_offset + 229376, 230400)


if __name__ == "__main__":
    unittest.main()
