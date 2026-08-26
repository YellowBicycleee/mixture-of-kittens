"""Dependency-free contract for the first real CuTe DSL MoK forward.

The implementation deliberately supports one production target while the
kernel topology is being brought up: Qwen BF16, EP8, H=4096, I=1024,
E=512, top-k=10.  Keeping the shape checks here lets the contract be tested on
a CPU-only development machine without importing torch or CUTLASS.
"""

from __future__ import annotations

from collections.abc import Sequence


EP_SIZE = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
NUM_GLOBAL_EXPERTS = 512
NUM_LOCAL_EXPERTS = NUM_GLOBAL_EXPERTS // EP_SIZE
TOPK = 10
ROW_ALIGNMENT = 256
DISPATCH_TILE_ROWS = 128
DISPATCH_TILE_COLUMNS = 128
DISPATCH_ROW_CHUNK_BYTES = DISPATCH_TILE_COLUMNS * 2  # BF16
DISPATCH_TILE_BYTES = DISPATCH_TILE_ROWS * DISPATCH_ROW_CHUNK_BYTES
DISPATCH_STORAGE_BYTES = 33792  # 128 mbarriers, then aligned 32 KiB tile
DISPATCH_CTAS_PER_SM = 4
COMBINE_TILE_ROWS = 16
COMBINE_TILE_COLUMNS = 1024
COMBINE_ROW_CHUNK_BYTES = COMBINE_TILE_COLUMNS * 2  # BF16
COMBINE_PIPE_DEPTH = 7
COMBINE_STAGE_BYTES = COMBINE_TILE_ROWS * COMBINE_ROW_CHUNK_BYTES
COMBINE_ARENA_BYTES = COMBINE_PIPE_DEPTH * COMBINE_STAGE_BYTES
COMBINE_STORAGE_BYTES = 229504  # 7 mbarriers, padding, then aligned arena
DEFAULT_NUM_COMM_SMS = 40
WAVEFRONT_WINDOW_ROWS = 65536
# Pipeline-v2 fuses shared and routed Gate + Up + SwiGLU. Shared always writes
# separate contiguous Gate/Up tensors. Routed macro 0 does too; replay macros
# declare Hidden as their only epilogue output and issue no Gate/Up stores.
REPLAY_GATE_UP_STORE_ELISION = True


def packed_weight_cache_key(
    gate_identity: int,
    gate_version: int,
    up_identity: int,
    up_version: int,
) -> tuple[int, int, int, int]:
    """Return the minimal identity/version key for one packed Gate+Up mirror."""

    values = (gate_identity, gate_version, up_identity, up_version)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("packed weight cache keys must be non-negative integers")
    return values


def packed_gate_up_shape(
    slot: str,
    source_shape: Sequence[int],
) -> tuple[int, ...]:
    """Return the physical concat shape for one shared or routed weight pair."""

    expected_rank = 2 if slot == "shared" else 3 if slot == "routed" else None
    if expected_rank is None:
        raise ValueError("packed Gate+Up slot must be 'shared' or 'routed'")
    shape = tuple(source_shape)
    if len(shape) != expected_rank or any(
        type(dimension) is not int or dimension <= 0 for dimension in shape
    ):
        raise ValueError(f"{slot} Gate/Up weights must have rank {expected_rank}")
    concat_dim = 0 if slot == "shared" else 1
    return tuple(
        2 * dimension if index == concat_dim else dimension
        for index, dimension in enumerate(shape)
    )


def should_store_routed_preact(
    global_offset: int,
    macrobatch_size: int,
) -> bool:
    """Whether one reverse-wavefront window belongs to replay macro zero."""

    if type(global_offset) is not int or global_offset < 0:
        raise ValueError("global_offset must be a non-negative integer")
    if type(macrobatch_size) is not int or macrobatch_size <= 0:
        raise ValueError("macrobatch_size must be a positive integer")
    return global_offset < macrobatch_size


def resolve_routed_num_tokens(
    num_tokens_device: object,
    num_tokens_host: int | None,
    schedule_capacity: int,
) -> int:
    """Resolve the routed-row count without rereading a mirrored scalar.

    New CuTe schedules populate ``num_tokens_host`` once during schedule
    construction.  ``None`` preserves compatibility with schedules created
    by the CUDA config or before the mirror was introduced.
    """

    if num_tokens_host is None:
        # Keep the legacy schedule path exact: torch scalar -> Python int.
        routed_num_tokens = int(num_tokens_device.item())
    else:
        if type(num_tokens_host) is not int:
            raise TypeError("schedule.num_tokens_host must be an integer or None")
        routed_num_tokens = num_tokens_host
    if not 0 <= routed_num_tokens <= schedule_capacity:
        raise RuntimeError("schedule.num_tokens exceeds the schedule capacity")
    return routed_num_tokens


def _task_geometry(
    macro_rows: int,
    *,
    tile_rows: int,
    tile_columns: int,
) -> tuple[int, int, int]:
    """Return ``(row_tiles, column_tiles, tasks)`` for one comm geometry."""

    if type(macro_rows) is not int or macro_rows <= 0:
        raise ValueError("macro_rows must be a positive integer")
    if macro_rows % tile_rows:
        raise ValueError(f"macro_rows must be divisible by {tile_rows}")
    row_tiles = macro_rows // tile_rows
    column_tiles = HIDDEN_SIZE // tile_columns
    return row_tiles, column_tiles, row_tiles * column_tiles


def _task_coordinates(
    task_index: int,
    macro_rows: int,
    *,
    tile_rows: int,
    tile_columns: int,
) -> tuple[int, int]:
    """Decode one linear communication task into row and column offsets."""

    row_tiles, column_tiles, tasks = _task_geometry(
        macro_rows,
        tile_rows=tile_rows,
        tile_columns=tile_columns,
    )
    del row_tiles
    if type(task_index) is not int or not 0 <= task_index < tasks:
        raise ValueError("task_index is outside the communication task grid")
    return (
        task_index // column_tiles * tile_rows,
        task_index % column_tiles * tile_columns,
    )


def dispatch_task_geometry(macro_rows: int) -> tuple[int, int, int]:
    """Standalone dispatch geometry: 128 rows x 128 BF16 columns."""

    return _task_geometry(
        macro_rows,
        tile_rows=DISPATCH_TILE_ROWS,
        tile_columns=DISPATCH_TILE_COLUMNS,
    )


def dispatch_task_coordinates(task_index: int, macro_rows: int) -> tuple[int, int]:
    return _task_coordinates(
        task_index,
        macro_rows,
        tile_rows=DISPATCH_TILE_ROWS,
        tile_columns=DISPATCH_TILE_COLUMNS,
    )


def combine_task_geometry(macro_rows: int) -> tuple[int, int, int]:
    """CUDA-aligned single-stage combine geometry: 16 rows x 1024 columns."""

    return _task_geometry(
        macro_rows,
        tile_rows=COMBINE_TILE_ROWS,
        tile_columns=COMBINE_TILE_COLUMNS,
    )


def combine_task_coordinates(task_index: int, macro_rows: int) -> tuple[int, int]:
    return _task_coordinates(
        task_index,
        macro_rows,
        tile_rows=COMBINE_TILE_ROWS,
        tile_columns=COMBINE_TILE_COLUMNS,
    )


def combine_pipeline_geometry(macro_rows: int) -> tuple[int, int, int]:
    """Return ``(tiles, groups, tail_stages)`` for depth-7 Combine."""

    _, _, tiles = combine_task_geometry(macro_rows)
    groups = (tiles + COMBINE_PIPE_DEPTH - 1) // COMBINE_PIPE_DEPTH
    return tiles, groups, tiles - (groups - 1) * COMBINE_PIPE_DEPTH


def combine_pipeline_stage_coordinates(
    group_index: int,
    macro_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Return only the existing tile coordinates in one Combine group."""

    tiles, groups, _ = combine_pipeline_geometry(macro_rows)
    if type(group_index) is not int or not 0 <= group_index < groups:
        raise ValueError("group_index is outside the combine pipeline grid")
    first_tile = group_index * COMBINE_PIPE_DEPTH
    return tuple(
        combine_task_coordinates(tile, macro_rows)
        for tile in range(first_tile, min(first_tile + COMBINE_PIPE_DEPTH, tiles))
    )


def validate_combine_smem_capacity(available_bytes: int) -> None:
    """Reject devices that cannot opt in to the fixed Combine7 storage."""

    if type(available_bytes) is not int or available_bytes <= 0:
        raise ValueError("available_bytes must be a positive integer")
    if available_bytes < COMBINE_STORAGE_BYTES:
        raise NotImplementedError(
            f"Combine7 requires {COMBINE_STORAGE_BYTES} opt-in shared-memory "
            f"bytes; device supports {available_bytes}"
        )


def validate_num_comm_sms(num_comm_sms: int) -> None:
    """Keep the CuTe fixed CTA pool compatible with CUDA's paired SM count."""

    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2:
        raise ValueError("num_comm_sms must be a positive even integer")


def standalone_comm_worker_grids(
    sm_count: int,
    combine_ctas_per_sm: int = 1,
) -> tuple[int, int]:
    """Return standalone CuTe Dispatch and Combine worker-CTA counts.

    CUDA's public ``fwd_num_comm_sms`` reserves communication CTAs inside its
    fused megakernel.  CuTe launches communication as separate kernels, so
    Dispatch uses four workers per physical SM while Combine uses the requested
    workers per SM instead of inheriting that fused-kernel reservation.
    """

    for name, value in (
        ("sm_count", sm_count),
        ("combine_ctas_per_sm", combine_ctas_per_sm),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return sm_count * DISPATCH_CTAS_PER_SM, sm_count * combine_ctas_per_sm


def macro_offsets(num_tokens: int, macrobatch_size: int) -> tuple[int, ...]:
    """Return real padded-token offsets in reverse order, leaving macro 0 last."""

    if type(num_tokens) is not int or num_tokens < 0:
        raise ValueError("num_tokens must be a non-negative integer")
    if num_tokens % ROW_ALIGNMENT:
        raise ValueError("num_tokens must be divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0:
        raise ValueError("macrobatch_size must be a positive integer")
    if macrobatch_size % ROW_ALIGNMENT:
        raise ValueError("macrobatch_size must be divisible by 256")

    if num_tokens == 0:
        return ()
    last = ((num_tokens - 1) // macrobatch_size) * macrobatch_size
    return tuple(range(last, -1, -macrobatch_size))


def wavefront_windows(
    num_tokens: int,
    macrobatch_size: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return reverse-macro, forward-window work as (global, ring, rows)."""

    windows = []
    for macro_offset in macro_offsets(num_tokens, macrobatch_size):
        macro_rows = min(macrobatch_size, num_tokens - macro_offset)
        for ring_offset in range(0, macro_rows, WAVEFRONT_WINDOW_ROWS):
            rows = min(WAVEFRONT_WINDOW_ROWS, macro_rows - ring_offset)
            windows.append((macro_offset + ring_offset, ring_offset, rows))
    return tuple(windows)


def macro_local_cu_seqlens(
    tokens_per_expert: Sequence[int],
    macro_offset: int,
    macro_rows: int,
) -> tuple[int, ...]:
    """Return QuACK varlen-M boundaries for one slice of MoK's expert rows.

    The schedule concatenates padded expert rows globally.  If ``S[e]`` is
    the exclusive prefix of ``tokens_per_expert``, macro ``[o, o + R)`` owns
    the local boundary ``clamp(S[e] - o, 0, R)``.  Repeated boundaries are
    intentional: they encode experts with no rows in this macro.
    """

    if type(macro_offset) is not int or macro_offset < 0:
        raise ValueError("macro_offset must be a non-negative integer")
    if type(macro_rows) is not int or macro_rows <= 0:
        raise ValueError("macro_rows must be a positive integer")

    boundaries = [0]
    prefix = 0
    for count in tokens_per_expert:
        if type(count) is not int or count < 0:
            raise ValueError("tokens_per_expert must contain non-negative integers")
        prefix += count
        boundaries.append(min(max(prefix - macro_offset, 0), macro_rows))

    if prefix < macro_offset + macro_rows:
        raise ValueError("macro extends past the scheduled expert rows")
    return tuple(boundaries)


def decode_schedule_entry(peer_rank: int, route_idx: int) -> tuple[int, int] | None:
    """Decode one scheduler row into ``(source_token, return_route)``.

    The scheduler stores ``route_idx = source_token * topk + k``.  Padding is
    represented by ``peer_rank == -1`` and must neither read x nor write the
    remote combine buffer.
    """

    if int(peer_rank) == -1:
        return None
    if not 0 <= int(peer_rank) < EP_SIZE:
        raise ValueError(f"peer_rank must be -1 or in [0, {EP_SIZE})")
    if int(route_idx) < 0:
        raise ValueError("route_idx must be non-negative for a real route")
    return int(route_idx) // TOPK, int(route_idx)


def validate_fixed_forward_contract(
    *,
    ep_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    num_local_tokens: int,
    schedule_capacity: int,
    macrobatch_size: int,
    minibatch_size: int,
    x_ptrs: Sequence[int],
    combine_ptrs: Sequence[int],
) -> None:
    """Validate scalar metadata and raw symmetric pointers for the fixed path."""

    expected = {
        "ep_size": (ep_size, EP_SIZE),
        "hidden_size": (hidden_size, HIDDEN_SIZE),
        "intermediate_size": (intermediate_size, INTERMEDIATE_SIZE),
        "num_local_experts": (num_local_experts, NUM_LOCAL_EXPERTS),
        "topk": (topk, TOPK),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise NotImplementedError(
                f"CuTe DSL forward requires {name}={wanted}; got {actual}"
            )

    if type(num_local_tokens) is not int or num_local_tokens < 512:
        raise ValueError("num_local_tokens must be an integer at least 512")
    if num_local_tokens % ROW_ALIGNMENT:
        raise ValueError("num_local_tokens must be divisible by 256")
    if type(minibatch_size) is not int or minibatch_size <= 0:
        raise ValueError("minibatch_size must be a positive integer")
    if minibatch_size % ROW_ALIGNMENT:
        raise ValueError("minibatch_size must be divisible by 256")
    if macrobatch_size % minibatch_size:
        raise ValueError("macrobatch_size must be a multiple of minibatch_size")

    if type(schedule_capacity) is not int or schedule_capacity <= 0:
        raise ValueError("schedule_capacity must be a positive integer")
    if schedule_capacity % ROW_ALIGNMENT:
        raise ValueError("schedule_capacity must be divisible by 256")
    # Reuse the macrobatch validation from the iterator.
    macro_offsets(schedule_capacity, macrobatch_size)
    for name, pointers in (("x_ptrs", x_ptrs), ("combine_ptrs", combine_ptrs)):
        if len(pointers) != EP_SIZE:
            raise ValueError(f"{name} must contain exactly {EP_SIZE} pointers")
        if any(type(pointer) is not int or pointer <= 0 for pointer in pointers):
            raise ValueError(f"{name} must contain positive integer pointers")
