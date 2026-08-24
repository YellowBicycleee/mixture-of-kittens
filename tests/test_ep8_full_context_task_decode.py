"""CPU references for EP8 expert and routed-task decoders."""

from __future__ import annotations

import bisect
import random
from collections import Counter


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


def _replay_task_width(
    minibatch: int, intermediate: int, *, paired_gate_up: bool = False
) -> int:
    row_blocks = minibatch // MLP_MB
    intermediate_blocks = intermediate // MLP_MB
    swiglu_tiles = (minibatch // 128) * (intermediate // 128)
    swiglu_tasks = (swiglu_tiles + 6 - 1) // 6
    gate_up_tasks = row_blocks * intermediate_blocks
    return (1 if paired_gate_up else 2) * gate_up_tasks + swiglu_tasks


def _paired_replay_enabled(
    *,
    minibatch_release: bool,
    num_devices: int,
    use_mxfp8: bool,
    full_context: bool,
    num_macrobatches: int,
) -> bool:
    return (
        minibatch_release
        and num_devices == 8
        and not use_mxfp8
        and not full_context
        and num_macrobatches > 1
    )


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


def _completed_segment_experts_by_owner(
    tokens_per_expert: list[int], macrobatch: int, minibatch: int
) -> list[list[int]]:
    routed_rows = sum(tokens_per_expert)
    num_minibatches = (routed_rows + minibatch - 1) // minibatch
    experts_by_owner: list[list[int]] = [[] for _ in range(num_minibatches)]
    expert_begin = 0
    for expert, expert_rows in enumerate(tokens_per_expert):
        expert_end = expert_begin + expert_rows
        first_macrobatch = expert_begin // macrobatch
        last_macrobatch = (expert_end - 1) // macrobatch if expert_rows else -1
        for macrobatch_idx in range(first_macrobatch, last_macrobatch + 1):
            segment_begin = max(expert_begin, macrobatch_idx * macrobatch)
            segment_end = min(expert_end, (macrobatch_idx + 1) * macrobatch)
            if segment_begin < segment_end:
                owner = (segment_end - 1) // minibatch
                experts_by_owner[owner].append(expert)
        expert_begin = expert_end
    return experts_by_owner


def _task_prefixes(
    completed_segment_offsets: list[int],
    minibatch_bwd_tasks: int,
    segment_wgrad_tasks: int,
    saved_context_minibatches: int | None = None,
    minibatch_replay_tasks: int = 0,
) -> list[int]:
    num_minibatches = len(completed_segment_offsets) - 1
    if saved_context_minibatches is None:
        saved_context_minibatches = num_minibatches
    return [
        owner * minibatch_bwd_tasks
        + max(0, owner - saved_context_minibatches) * minibatch_replay_tasks
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
    replay: bool = False,
    paired_gate_up: bool = False,
) -> list[int]:
    completed = _completed_segment_offsets(
        tokens_per_expert, macrobatch, minibatch
    )
    minibatch_tasks, segment_tasks = _task_widths(
        minibatch, hidden, intermediate
    )
    saved_context_minibatches = macrobatch // minibatch
    replay_tasks = (
        _replay_task_width(
            minibatch, intermediate, paired_gate_up=paired_gate_up
        )
        if replay
        else 0
    )
    prefixes = _task_prefixes(
        completed,
        minibatch_tasks,
        segment_tasks,
        saved_context_minibatches,
        replay_tasks,
    )
    bucket_shift = minibatch_tasks.bit_length() - 1
    bucket_width = 1 << bucket_shift
    buckets = _build_task_owner_buckets(prefixes, bucket_width)
    assert bucket_width <= minibatch_tasks
    assert all(
        prefixes[owner + 1] - prefixes[owner] >= bucket_width
        for owner in range(len(prefixes) - 1)
    )
    for idx in range(prefixes[-1]):
        decoded = _bucket_decode(prefixes, buckets, bucket_shift, idx)
        assert decoded == _linear_task_owner(prefixes, idx)
        assert decoded == _binary_task_owner(prefixes, idx)
    experts_by_owner = _completed_segment_experts_by_owner(
        tokens_per_expert, macrobatch, minibatch
    )
    completed_experts = [
        expert for owner_experts in experts_by_owner for expert in owner_experts
    ]
    for owner, owner_experts in enumerate(experts_by_owner):
        begin = completed[owner]
        end = completed[owner + 1]
        assert completed_experts[begin:end] == owner_experts
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


def test_replay_aware_owner_buckets_cover_threshold_empty_split_and_tail() -> None:
    cases = [
        ([0, 512, 0, 768, 256], 512, 256, 1024, 512),
        ([768, 0, 256, 512], 512, 256, 2048, 1024),
        ([0, 768, 256, 0, 768], 1024, 512, 4096, 1024),
    ]
    for tokens, macrobatch, minibatch, hidden, intermediate in cases:
        completed = _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            hidden,
            intermediate,
            replay=True,
        )
        bwd_tasks, segment_tasks = _task_widths(
            minibatch, hidden, intermediate
        )
        replay_tasks = _replay_task_width(minibatch, intermediate)
        saved = macrobatch // minibatch
        prefixes = _task_prefixes(
            completed, bwd_tasks, segment_tasks, saved, replay_tasks
        )
        if len(prefixes) - 1 > saved:
            assert prefixes[saved + 1] - prefixes[saved] >= (
                bwd_tasks + replay_tasks
            )
            assert _binary_task_owner(prefixes, prefixes[saved]) == (saved, 0)


def test_seeded_replay_aware_owner_buckets_match_both_references() -> None:
    rng = random.Random(20260824 ^ 0xB300)
    for _ in range(40):
        minibatch = rng.choice([256, 512, 1024])
        macrobatch = minibatch * rng.choice([1, 2, 4, 8])
        experts = rng.choice([1, 4, 16, 64])
        tokens = _random_partition(rng, rng.randint(1, 96), experts)
        _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            rng.choice([1024, 2048, 4096]),
            rng.choice([512, 1024, 2048]),
            replay=True,
        )


def test_qwen_100k_b131072_replay_task_count_and_bucket_capacity() -> None:
    # 64 EP8-local expert spans use the three padded sizes observed in the
    # Qwen-shaped 100K benchmark.  This ordering avoids coincident expert and
    # macrobatch boundaries, giving the worst-case E + M - 1 segments.
    expert_blocks = [64] * 32 + [63] * 16 + [62] * 16
    random.Random(0).shuffle(expert_blocks)
    tokens = [blocks * MLP_MB for blocks in expert_blocks]
    macrobatch = 131072
    minibatch = 4096
    completed = _completed_segment_offsets(tokens, macrobatch, minibatch)
    bwd_tasks, segment_tasks = _task_widths(minibatch, 4096, 1024)
    replay_tasks = _replay_task_width(minibatch, 1024)
    saved = macrobatch // minibatch
    prefixes = _task_prefixes(
        completed, bwd_tasks, segment_tasks, saved, replay_tasks
    )
    bucket_shift = bwd_tasks.bit_length() - 1
    buckets = _build_task_owner_buckets(prefixes, 1 << bucket_shift)

    assert sum(tokens) == 1036288
    assert len(completed) - 1 == 253
    assert saved == 32
    assert bwd_tasks == 384
    assert replay_tasks == 171
    assert segment_tasks == 192
    assert completed[-1] == 71
    assert prefixes[-1] == 148575
    assert len(buckets) == 581

    boundary_indices = {
        index
        for prefix in prefixes
        for index in (prefix - 1, prefix, prefix + 1)
        if 0 <= index < prefixes[-1]
    }
    rng = random.Random(20260824)
    sampled_indices = boundary_indices | {
        rng.randrange(prefixes[-1]) for _ in range(10000)
    }
    for idx in sampled_indices:
        decoded = _bucket_decode(prefixes, buckets, bucket_shift, idx)
        assert decoded == _binary_task_owner(prefixes, idx)


def test_qwen_100k_paired_replay_counts_and_bucket_capacity() -> None:
    expert_blocks = [64] * 32 + [63] * 16 + [62] * 16
    random.Random(0).shuffle(expert_blocks)
    tokens = [blocks * MLP_MB for blocks in expert_blocks]
    macrobatch = 131072
    minibatch = 4096
    completed = _completed_segment_offsets(tokens, macrobatch, minibatch)
    bwd_tasks, segment_tasks = _task_widths(minibatch, 4096, 1024)
    old_replay_tasks = _replay_task_width(minibatch, 1024)
    paired_replay_tasks = _replay_task_width(
        minibatch, 1024, paired_gate_up=True
    )
    saved = macrobatch // minibatch
    old_prefixes = _task_prefixes(
        completed, bwd_tasks, segment_tasks, saved, old_replay_tasks
    )
    paired_prefixes = _task_prefixes(
        completed, bwd_tasks, segment_tasks, saved, paired_replay_tasks
    )
    bucket_shift = bwd_tasks.bit_length() - 1
    buckets = _build_task_owner_buckets(
        paired_prefixes, 1 << bucket_shift
    )

    assert bwd_tasks == 384
    assert segment_tasks == 192
    assert old_replay_tasks == 171
    assert paired_replay_tasks == 107
    assert len(completed) - 1 == 253
    assert saved == 32
    assert len(completed) - 1 - saved == 221
    assert completed[-1] == 71
    assert old_prefixes[-1] == 148575
    assert paired_prefixes[-1] == 134431
    assert old_prefixes[-1] - paired_prefixes[-1] == 221 * 64
    assert len(buckets) == 526
    assert 1 << bucket_shift == 256

    # The 100K harness reserves 4,096,000 routed rows even though this seeded
    # fixture executes 1,036,288.  Keep host launch capacity, device retirement,
    # and the compact Replay width on the same accounting contract.
    shared_tasks = 9792
    capacity_minibatches = 1000
    capacity_macrobatches = 32
    capacity_replay_minibatches = capacity_minibatches - saved
    max_completed_segments = capacity_macrobatches + len(tokens) - 1
    old_host_clusters = (
        shared_tasks
        + capacity_minibatches * bwd_tasks
        + capacity_replay_minibatches * old_replay_tasks
        + max_completed_segments * segment_tasks
    )
    paired_host_clusters = (
        shared_tasks
        + capacity_minibatches * bwd_tasks
        + capacity_replay_minibatches * paired_replay_tasks
        + max_completed_segments * segment_tasks
    )
    assert (old_host_clusters, paired_host_clusters) == (577560, 515608)
    assert (2 * old_host_clusters + 44, 2 * paired_host_clusters + 44) == (
        1155164,
        1031260,
    )

    old_device_clusters = shared_tasks + old_prefixes[-1]
    paired_device_clusters = shared_tasks + paired_prefixes[-1]
    assert (old_device_clusters, paired_device_clusters) == (158367, 144223)
    assert (2 * old_device_clusters + 44, 2 * paired_device_clusters + 44) == (
        316778,
        288490,
    )

    replay_dispatch_arrivals = (minibatch // 128) * (4096 // 512)
    legacy_gate_up_ready = 2 * 2
    paired_gate_up_ready = 1 * 2 * 2
    assert replay_dispatch_arrivals == 256
    assert legacy_gate_up_ready == paired_gate_up_ready == 4

    assert (old_prefixes[31], paired_prefixes[31]) == (13248, 13248)
    assert old_prefixes[32] == paired_prefixes[32] == 14016
    assert (old_prefixes[33], paired_prefixes[33]) == (14571, 14507)
    assert (old_prefixes[252], paired_prefixes[252]) == (147828, 133748)

    for idx in range(paired_prefixes[-1]):
        assert _bucket_decode(
            paired_prefixes, buckets, bucket_shift, idx
        ) == _binary_task_owner(paired_prefixes, idx)


def test_qwen_100k_paired_replay_work_identity() -> None:
    expert_blocks = [64] * 32 + [63] * 16 + [62] * 16
    random.Random(0).shuffle(expert_blocks)
    tokens = [blocks * MLP_MB for blocks in expert_blocks]
    macrobatch = 131072
    minibatch = 4096
    completed = _completed_segment_offsets(tokens, macrobatch, minibatch)
    bwd_tasks, segment_tasks = _task_widths(minibatch, 4096, 1024)
    old_replay_tasks = _replay_task_width(minibatch, 1024)
    paired_replay_tasks = _replay_task_width(
        minibatch, 1024, paired_gate_up=True
    )
    saved = macrobatch // minibatch
    old_prefixes = _task_prefixes(
        completed, bwd_tasks, segment_tasks, saved, old_replay_tasks
    )
    paired_prefixes = _task_prefixes(
        completed, bwd_tasks, segment_tasks, saved, paired_replay_tasks
    )
    bucket_shift = bwd_tasks.bit_length() - 1
    buckets = _build_task_owner_buckets(
        paired_prefixes, 1 << bucket_shift
    )
    gate_up_tasks = (minibatch // MLP_MB) * (1024 // MLP_MB)

    expanded = Counter()
    paired_tasks = 0
    for physical_idx in range(paired_prefixes[-1]):
        owner, local = _bucket_decode(
            paired_prefixes, buckets, bucket_shift, physical_idx
        )
        if owner < saved:
            expanded[(owner, local)] += 1
        elif local < gate_up_tasks:
            expanded[(owner, local)] += 1
            expanded[(owner, gate_up_tasks + local)] += 1
            paired_tasks += 1
        else:
            expanded[(owner, local + gate_up_tasks)] += 1

    canonical = Counter(
        (owner, local)
        for owner in range(len(old_prefixes) - 1)
        for local in range(old_prefixes[owner + 1] - old_prefixes[owner])
    )
    assert expanded == canonical
    assert paired_tasks == 221 * 64 == 14144
    assert sum(expanded.values()) == old_prefixes[-1] == 148575


def test_paired_replay_identity_guards_and_small_routing_cases() -> None:
    target = {
        "minibatch_release": True,
        "num_devices": 8,
        "use_mxfp8": False,
        "full_context": False,
        "num_macrobatches": 2,
    }
    assert _paired_replay_enabled(**target)
    for key, identity_value in (
        ("minibatch_release", False),
        ("num_devices", 4),
        ("use_mxfp8", True),
        ("full_context", True),
        ("num_macrobatches", 1),
    ):
        identity = dict(target)
        identity[key] = identity_value
        assert not _paired_replay_enabled(**identity)

    cases = [
        ([0, 512, 0, 768, 256], 512, 256, 1024, 512),
        ([768, 0, 256, 512], 512, 256, 2048, 1024),
        ([0, 768, 256, 0, 768], 1024, 512, 4096, 1024),
    ]
    for tokens, macrobatch, minibatch, hidden, intermediate in cases:
        assert _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            hidden,
            intermediate,
            replay=True,
            paired_gate_up=True,
        ) == _completed_segment_offsets(tokens, macrobatch, minibatch)
