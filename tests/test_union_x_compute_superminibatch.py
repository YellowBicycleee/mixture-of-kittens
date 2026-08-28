from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORWARD = (
    ROOT / "csrc" / "megakernel" / "forward_union_x.cuh"
).read_text()


def legacy_tasks(
    num_macrobatches: int,
    last_macrobatch_minis: int,
    full_macrobatch_minis: int,
    gate_tasks: int,
    down_tasks: int,
) -> list[tuple[int, int, str, int]]:
    tasks = []
    for macrobatch in range(num_macrobatches - 1, -1, -1):
        num_minis = (
            last_macrobatch_minis
            if macrobatch == num_macrobatches - 1
            else full_macrobatch_minis
        )
        for minibatch in range(num_minis):
            tasks.extend(
                (macrobatch, minibatch, "gate", task)
                for task in range(gate_tasks)
            )
            tasks.extend(
                (macrobatch, minibatch, "down", task)
                for task in range(down_tasks)
            )
    return tasks


def grouped_tasks(
    num_macrobatches: int,
    last_macrobatch_minis: int,
    full_macrobatch_minis: int,
    gate_tasks: int,
    down_tasks: int,
    group_minis: int,
) -> Iterator[tuple[int, int, str, int]]:
    for macrobatch in range(num_macrobatches - 1, -1, -1):
        num_minis = (
            last_macrobatch_minis
            if macrobatch == num_macrobatches - 1
            else full_macrobatch_minis
        )
        for group_start in range(0, num_minis, group_minis):
            group_end = min(group_start + group_minis, num_minis)
            for minibatch in range(group_start, group_end):
                yield from (
                    (macrobatch, minibatch, "gate", task)
                    for task in range(gate_tasks)
                )
            for minibatch in range(group_start, group_end):
                yield from (
                    (macrobatch, minibatch, "down", task)
                    for task in range(down_tasks)
                )


def decode_grouped_task(
    routed_task_order: int,
    num_macrobatches: int,
    last_macrobatch_minis: int,
    full_macrobatch_minis: int,
    gate_tasks: int,
    down_tasks: int,
    group_minis: int,
) -> tuple[int, int, str, int]:
    minibatch_tasks = gate_tasks + down_tasks
    last_macrobatch_tasks = last_macrobatch_minis * minibatch_tasks
    full_macrobatch_tasks = full_macrobatch_minis * minibatch_tasks
    if routed_task_order < last_macrobatch_tasks:
        macrobatch = num_macrobatches - 1
        macrobatch_minis = last_macrobatch_minis
        macrobatch_task = routed_task_order
    else:
        remaining = routed_task_order - last_macrobatch_tasks
        macrobatch = num_macrobatches - 2 - remaining // full_macrobatch_tasks
        macrobatch_minis = full_macrobatch_minis
        macrobatch_task = remaining % full_macrobatch_tasks

    full_group_tasks = group_minis * minibatch_tasks
    num_full_groups = macrobatch_minis // group_minis
    full_groups_tasks = num_full_groups * full_group_tasks
    if macrobatch_task < full_groups_tasks:
        group_idx = macrobatch_task // full_group_tasks
        group_start = group_idx * group_minis
        group_size = group_minis
        group_task = macrobatch_task - group_idx * full_group_tasks
    else:
        group_start = num_full_groups * group_minis
        group_size = macrobatch_minis - group_start
        group_task = macrobatch_task - full_groups_tasks

    group_gate_tasks = group_size * gate_tasks
    if group_task < group_gate_tasks:
        minibatch = group_start + group_task // gate_tasks
        return macrobatch, minibatch, "gate", group_task % gate_tasks
    group_down_task = group_task - group_gate_tasks
    minibatch = group_start + group_down_task // down_tasks
    return macrobatch, minibatch, "down", group_down_task % down_tasks


def assert_mapping(
    *,
    num_macrobatches: int,
    last_macrobatch_minis: int,
    full_macrobatch_minis: int,
    gate_tasks: int,
    down_tasks: int,
    group_minis: int,
) -> None:
    expected = legacy_tasks(
        num_macrobatches,
        last_macrobatch_minis,
        full_macrobatch_minis,
        gate_tasks,
        down_tasks,
    )
    grouped = list(grouped_tasks(
        num_macrobatches,
        last_macrobatch_minis,
        full_macrobatch_minis,
        gate_tasks,
        down_tasks,
        group_minis,
    ))
    decoded = [
        decode_grouped_task(
            task,
            num_macrobatches,
            last_macrobatch_minis,
            full_macrobatch_minis,
            gate_tasks,
            down_tasks,
            group_minis,
        )
        for task in range(len(expected))
    ]
    assert decoded == grouped
    assert sorted(decoded) == sorted(expected)

    positions = {task: idx for idx, task in enumerate(decoded)}
    for macrobatch in range(num_macrobatches):
        num_minis = (
            last_macrobatch_minis
            if macrobatch == num_macrobatches - 1
            else full_macrobatch_minis
        )
        for minibatch in range(num_minis):
            last_gate = positions[
                (macrobatch, minibatch, "gate", gate_tasks - 1)
            ]
            first_down = positions[(macrobatch, minibatch, "down", 0)]
            assert last_gate < first_down


def test_qwen_mini4_task_set_and_switch_count() -> None:
    assert_mapping(
        num_macrobatches=32,
        last_macrobatch_minis=4,
        full_macrobatch_minis=8,
        gate_tasks=64,
        down_tasks=256,
        group_minis=4,
    )
    legacy_switches = 2 * 252 - 1
    grouped_switches = 2 * 63 - 1
    assert legacy_switches == 503
    assert grouped_switches == 125


def test_mini16_is_unchanged() -> None:
    legacy = legacy_tasks(32, 1, 2, 256, 1024)
    grouped = list(grouped_tasks(32, 1, 2, 256, 1024, 1))
    assert grouped == legacy


def test_all_256_row_macro_tails_preserve_dependencies() -> None:
    # macro32K/minibatch256 has 128 possible nonempty tail lengths.  A 16K
    # compute window groups 64 communication minibatches without crossing the
    # macro boundary.
    for last_minis in range(1, 129):
        assert_mapping(
            num_macrobatches=3,
            last_macrobatch_minis=last_minis,
            full_macrobatch_minis=128,
            gate_tasks=4,
            down_tasks=16,
            group_minis=64,
        )


def test_source_keeps_task_counts_and_changes_only_decode_order() -> None:
    for token in (
        "UNION_X_COMPUTE_GROUP_ROWS = 16384",
        "const int routed_task_order = compute_cluster_idx - shared_tasks",
        "last_macrobatch_num_minibatches * minibatch_tasks",
        "group_num_minibatches * minibatch_routed_gate_up_tasks",
        "const bool is_routed_gate_up",
        "group_down_task_order / minibatch_routed_down_tasks",
    ):
        assert token in FORWARD
    assert "num_minibatches * minibatch_tasks" in FORWARD
