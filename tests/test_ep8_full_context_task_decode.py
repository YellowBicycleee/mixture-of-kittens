"""CPU reference tests for the EP8 full-context Dgrad task decoder."""

from __future__ import annotations

import bisect
import random


MLP_MB = 256


def _expert_blocks(tokens_per_expert: list[int]) -> list[int]:
    assert all(rows >= 0 and rows % MLP_MB == 0 for rows in tokens_per_expert)
    return [rows // MLP_MB for rows in tokens_per_expert]


def _legacy_decode(
    tokens_per_expert: list[int],
    minibatch_blocks: int,
    global_minibatch: int,
    col_blocks: int,
    task: int,
) -> tuple[int, int, int, int] | None:
    """Mirror the original per-expert scan in grouped_gemm.cuh."""
    minibatch_begin = global_minibatch * minibatch_blocks
    minibatch_end = minibatch_begin + minibatch_blocks
    expert_begin = 0
    for expert, blocks in enumerate(_expert_blocks(tokens_per_expert)):
        first_block = max(minibatch_begin, expert_begin)
        row_blocks = max(
            0, min(minibatch_end, expert_begin + blocks) - first_block
        )
        expert_tasks = row_blocks * col_blocks
        if task < expert_tasks:
            return expert, first_block, row_blocks, task
        task -= expert_tasks
        expert_begin += blocks
    return None


def _prefix_decode(
    tokens_per_expert: list[int],
    minibatch_blocks: int,
    global_minibatch: int,
    col_blocks: int,
    task: int,
) -> tuple[tuple[int, int, int, int] | None, int]:
    """Mirror six unrolled upper-bound steps and the E>64 fallback loop."""
    offsets = [0]
    for blocks in _expert_blocks(tokens_per_expert):
        offsets.append(offsets[-1] + blocks)

    minibatch_begin = global_minibatch * minibatch_blocks
    global_task_block = minibatch_begin + task // col_blocks
    if global_task_block >= offsets[-1]:
        return None, 0

    expert_begin = 0
    expert_end = len(tokens_per_expert) - 1
    comparisons = 0
    for _ in range(6):
        if expert_begin < expert_end:
            expert_mid = (expert_begin + expert_end) // 2
            comparisons += 1
            if offsets[expert_mid + 1] <= global_task_block:
                expert_begin = expert_mid + 1
            else:
                expert_end = expert_mid
    while expert_begin < expert_end:
        expert_mid = (expert_begin + expert_end) // 2
        comparisons += 1
        if offsets[expert_mid + 1] <= global_task_block:
            expert_begin = expert_mid + 1
        else:
            expert_end = expert_mid

    expert = expert_begin
    first_block = max(minibatch_begin, offsets[expert])
    row_blocks = (
        min(minibatch_begin + minibatch_blocks, offsets[expert + 1])
        - first_block
    )
    expert_task = task - (first_block - minibatch_begin) * col_blocks
    return (expert, first_block, row_blocks, expert_task), comparisons


def _assert_matches_legacy(
    tokens_per_expert: list[int], minibatch_blocks: int, col_blocks: int
) -> list[int]:
    total_blocks = sum(_expert_blocks(tokens_per_expert))
    num_minibatches = (total_blocks + minibatch_blocks - 1) // minibatch_blocks
    comparisons = []
    for global_minibatch in range(num_minibatches):
        for task in range(minibatch_blocks * col_blocks):
            decoded, count = _prefix_decode(
                tokens_per_expert,
                minibatch_blocks,
                global_minibatch,
                col_blocks,
                task,
            )
            assert decoded == _legacy_decode(
                tokens_per_expert,
                minibatch_blocks,
                global_minibatch,
                col_blocks,
                task,
            )
            if decoded is not None:
                comparisons.append(count)
    return comparisons


def _random_partition(
    rng: random.Random, total_blocks: int, experts: int
) -> list[int]:
    counts = [0] * experts
    for _ in range(total_blocks):
        counts[rng.randrange(experts)] += MLP_MB
    return counts


def _task_widths(
    minibatch: int, hidden: int, intermediate: int
) -> tuple[int, int]:
    row_blocks = minibatch // MLP_MB
    intermediate_blocks = intermediate // MLP_MB
    hidden_blocks = hidden // MLP_MB
    swiglu_tiles = (minibatch // 128) * (intermediate // 128)
    swiglu_tasks = (swiglu_tiles + 4 - 1) // 4
    minibatch_bwd_tasks = (
        row_blocks * intermediate_blocks
        + swiglu_tasks
        + row_blocks * hidden_blocks
    )
    segment_wgrad_tasks = 3 * intermediate_blocks * hidden_blocks
    return minibatch_bwd_tasks, segment_wgrad_tasks


def _completed_segment_offsets(
    tokens_per_expert: list[int], macrobatch: int, minibatch: int
) -> list[int]:
    assert macrobatch > 0 and macrobatch % minibatch == 0
    routed_rows = sum(tokens_per_expert)
    num_minibatches = (routed_rows + minibatch - 1) // minibatch
    offsets = [0]
    completed = 0
    for global_minibatch in range(num_minibatches):
        minibatch_begin = global_minibatch * minibatch
        macrobatch_begin = (minibatch_begin // macrobatch) * macrobatch
        macrobatch_end = min(macrobatch_begin + macrobatch, routed_rows)
        expert_begin = 0
        for expert_rows in tokens_per_expert:
            expert_end = expert_begin + expert_rows
            segment_begin = max(expert_begin, macrobatch_begin)
            segment_end = min(expert_end, macrobatch_end)
            if (
                segment_begin < segment_end
                and (segment_end - 1) // minibatch == global_minibatch
            ):
                completed += 1
            expert_begin = expert_end
        offsets.append(completed)
    return offsets


def _task_prefixes(
    completed_segment_offsets: list[int],
    minibatch_bwd_tasks: int,
    segment_wgrad_tasks: int,
) -> list[int]:
    return [
        owner * minibatch_bwd_tasks
        + completed * segment_wgrad_tasks
        for owner, completed in enumerate(completed_segment_offsets)
    ]


def _build_task_owner_buckets(
    prefixes: list[int], bucket_width: int
) -> list[int]:
    num_minibatches = len(prefixes) - 1
    num_buckets = (prefixes[-1] + bucket_width - 1) // bucket_width
    buckets = []
    owner = 0
    for bucket in range(num_buckets):
        bucket_begin = bucket * bucket_width
        while (
            owner + 1 < num_minibatches
            and prefixes[owner + 1] <= bucket_begin
        ):
            owner += 1
        buckets.append(owner)
    return buckets


def _bucket_decode(
    prefixes: list[int], buckets: list[int], bucket_shift: int, idx: int
) -> tuple[int, int]:
    owner = buckets[idx >> bucket_shift]
    if idx >= prefixes[owner + 1]:
        owner += 1
    return owner, idx - prefixes[owner]


def _linear_task_owner(prefixes: list[int], idx: int) -> tuple[int, int]:
    for owner in range(len(prefixes) - 1):
        if idx < prefixes[owner + 1]:
            return owner, idx - prefixes[owner]
    raise AssertionError("valid task index has no owner")


def _binary_task_owner(prefixes: list[int], idx: int) -> tuple[int, int]:
    owner = bisect.bisect_right(prefixes, idx) - 1
    return owner, idx - prefixes[owner]


def _assert_task_owner_case(
    tokens_per_expert: list[int],
    macrobatch: int,
    minibatch: int,
    hidden: int,
    intermediate: int,
) -> list[int]:
    completed = _completed_segment_offsets(
        tokens_per_expert, macrobatch, minibatch
    )
    minibatch_tasks, segment_tasks = _task_widths(
        minibatch, hidden, intermediate
    )
    prefixes = _task_prefixes(completed, minibatch_tasks, segment_tasks)
    bucket_shift = minibatch_tasks.bit_length() - 1
    bucket_width = 1 << bucket_shift
    buckets = _build_task_owner_buckets(prefixes, bucket_width)
    assert bucket_width <= minibatch_tasks
    for idx in range(prefixes[-1]):
        decoded = _bucket_decode(prefixes, buckets, bucket_shift, idx)
        assert decoded == _linear_task_owner(prefixes, idx)
        assert decoded == _binary_task_owner(prefixes, idx)
    return completed


def test_seeded_decoder_matches_legacy_with_empty_experts() -> None:
    rng = random.Random(20260823)
    _assert_matches_legacy([0, 256, 0, 768, 0, 256], 2, 7)
    for _ in range(40):
        experts = rng.choice([1, 3, 16, 63, 64, 65, 128])
        tokens = _random_partition(rng, rng.randint(1, 80), experts)
        _assert_matches_legacy(
            tokens,
            rng.choice([1, 2, 4, 8, 16]),
            rng.choice([1, 4, 7]),
        )


def test_qwen_ep8_uses_exactly_six_unrolled_steps() -> None:
    tokens = [0, 512, 0, 256] * 16
    comparisons = _assert_matches_legacy(tokens, 8, 4)
    assert comparisons and set(comparisons) == {6}


def test_more_than_64_experts_uses_fallback() -> None:
    tokens = [0, 512, 0, 256] * 32
    comparisons = _assert_matches_legacy(tokens, 8, 4)
    assert comparisons and all(count > 6 for count in comparisons)


def test_scheduled_tail_tasks_remain_noops() -> None:
    tokens = [0, 256, 0, 512]
    minibatch_blocks = 8
    col_blocks = 4
    valid_tasks = sum(_expert_blocks(tokens)) * col_blocks
    for task in range(valid_tasks, minibatch_blocks * col_blocks):
        decoded, _ = _prefix_decode(
            tokens, minibatch_blocks, 0, col_blocks, task
        )
        assert decoded is None
        assert _legacy_decode(tokens, minibatch_blocks, 0, col_blocks, task) is None


def test_task_owner_buckets_cover_empty_split_and_partial_tail() -> None:
    empty = _assert_task_owner_case([0, 512, 0, 256], 512, 256, 1024, 512)
    assert empty == [0, 0, 1, 2]

    split = _assert_task_owner_case([768, 256], 512, 256, 1024, 512)
    assert split == [0, 0, 1, 2, 3]

    partial_tail = _assert_task_owner_case(
        [256, 512], 1024, 512, 2048, 1024
    )
    assert partial_tail == [0, 1, 2]


def test_seeded_task_owner_buckets_match_both_references() -> None:
    rng = random.Random(20260824)
    for _ in range(40):
        minibatch = rng.choice([256, 512, 1024])
        macrobatch = minibatch * rng.choice([1, 2, 4, 8])
        experts = rng.choice([1, 4, 16, 64])
        tokens = _random_partition(rng, rng.randint(1, 48), experts)
        _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            rng.choice([1024, 2048, 4096]),
            rng.choice([512, 1024, 2048]),
        )
