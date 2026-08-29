"""CPU-only schedule/readiness contract for the persistent BF16 forward.

This module intentionally models one target: Qwen EP8, H=4096, I=1024,
macrobatch=32768, minibatch=4096, and two-CTA GEMM clusters.  It is not a
runtime scheduler.  Its small immutable objects make the generation keys,
producer counts, reverse-macrobatch order, and ring-reuse dependencies
testable before they are translated to CuTe DSL device code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


EP_SIZE = 8
NUM_LOCAL_EXPERTS = 64
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
MACROBATCH_ROWS = 32768
MINIBATCH_ROWS = 4096
ROW_ALIGNMENT = 256
CLUSTER_SIZE = 2
MINIBATCHES_PER_MACROBATCH = MACROBATCH_ROWS // MINIBATCH_ROWS
MLP_SUPERGROUP_SIZE = 8
TASK_ORDER = "expert_segment_supergroup8"

# One persistent compute claim denotes one two-CTA cluster task.
FC1_TILE_M = 256
FC1_TILE_N = 128
DOWN_TILE_M = 256
DOWN_TILE_N = 256

# Communication claims are CTA-local, matching the BF16 V1-C2 tiling.
DISPATCH_TILE_M = 128
DISPATCH_TILE_N = 512
COMBINE_TILE_M = 16
COMBINE_TILE_N = 1024
Y_DONE_ROW_BLOCK = FC1_TILE_M // CLUSTER_SIZE  # 128 rows

FC1_ROW_TILES = MINIBATCH_ROWS // FC1_TILE_M
FC1_COLUMN_TILES = INTERMEDIATE_SIZE // FC1_TILE_N
FC1_TASKS = FC1_ROW_TILES * FC1_COLUMN_TILES
DOWN_ROW_TILES = MINIBATCH_ROWS // DOWN_TILE_M
DOWN_COLUMN_TILES = HIDDEN_SIZE // DOWN_TILE_N
DOWN_TASKS = DOWN_ROW_TILES * DOWN_COLUMN_TILES
DISPATCH_ROW_TILES = MINIBATCH_ROWS // DISPATCH_TILE_M
DISPATCH_COLUMN_TILES = HIDDEN_SIZE // DISPATCH_TILE_N
DISPATCH_TASKS = DISPATCH_ROW_TILES * DISPATCH_COLUMN_TILES
COMBINE_ROW_TILES = MINIBATCH_ROWS // COMBINE_TILE_M
COMBINE_COLUMN_TILES = HIDDEN_SIZE // COMBINE_TILE_N
COMBINE_RAW_TILES = COMBINE_ROW_TILES * COMBINE_COLUMN_TILES
Y_DONE_PARTS = MINIBATCH_ROWS // Y_DONE_ROW_BLOCK

# Required arrivals for one full minibatch.  FC1 and Down are cluster tasks,
# so both CTAs arrive after their disjoint M128 output stores are visible.
X_READY_REQUIRED = DISPATCH_TASKS
HIDDEN_READY_REQUIRED = FC1_COLUMN_TILES * CLUSTER_SIZE
Y_READY_REQUIRED = DOWN_TASKS * CLUSTER_SIZE
Y_DONE_REQUIRED = (
    Y_DONE_ROW_BLOCK // COMBINE_TILE_M
) * COMBINE_COLUMN_TILES

DISPATCH_ROLE = "dispatch"
FC1_ROLE = "fc1"
DOWN_ROLE = "down"
COMBINE_ROLE = "combine"
COMM_ROLES = (DISPATCH_ROLE, COMBINE_ROLE)

X_READY = "x_ready"
HIDDEN_READY = "hidden_ready"
Y_READY = "y_ready"
Y_DONE = "y_done"
_READY_KIND_CODE = {
    X_READY: 1,
    HIDDEN_READY: 2,
    Y_READY: 3,
    Y_DONE: 4,
}


class UnsupportedTailError(ValueError):
    """The fixed BF16 tiles cannot safely cover an unaligned routed tail."""


class ABAHazardError(ValueError):
    """A ring-reuse dependency names the slot but not its prior generation."""


@dataclass(frozen=True, order=True)
class ReadyKey:
    """Globally unique readiness key.

    ``generation`` is the global minibatch index, never the reused ring slot.
    ``part`` is a row-block index for Hidden/Y-done and zero otherwise.
    """

    kind: str
    generation: int
    part: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _READY_KIND_CODE:
            raise ValueError(f"unknown readiness kind: {self.kind}")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if type(self.part) is not int or not 0 <= self.part < 1 << 16:
            raise ValueError("part must fit in 16 unsigned bits")
        if self.generation >= 1 << 32:
            raise ValueError("generation must fit in 32 unsigned bits")
        if self.kind in (X_READY, Y_READY) and self.part != 0:
            raise ValueError(f"{self.kind} has one counter per global minibatch")
        if self.kind == HIDDEN_READY and self.part >= FC1_ROW_TILES:
            raise ValueError("hidden_ready part is outside the minibatch")
        if self.kind == Y_DONE and self.part >= Y_DONE_PARTS:
            raise ValueError("y_done part is outside the minibatch")

    @property
    def routed_counter_index(self) -> int:
        """Routed-relative index before CUDA's shared hidden prefix."""

        if self.kind == HIDDEN_READY:
            return self.generation * FC1_ROW_TILES + self.part
        if self.kind == Y_DONE:
            return self.generation * Y_DONE_PARTS + self.part
        return self.generation

    def physical_counter_index(self, *, shared_row_blocks: int) -> int:
        """Physical index in CUDA's allocated kind-specific counter array."""

        if type(shared_row_blocks) is not int or shared_row_blocks < 0:
            raise ValueError("shared_row_blocks must be a non-negative integer")
        if self.kind == HIDDEN_READY:
            return shared_row_blocks + self.routed_counter_index
        return self.routed_counter_index

    @property
    def linear_id(self) -> int:
        """Stable flat ID when the device implementation uses one counter slab."""

        return (
            _READY_KIND_CODE[self.kind] << 48
            | self.generation << 16
            | self.part
        )


@dataclass(frozen=True)
class ReadySpec:
    key: ReadyKey
    required_count: int

    def __post_init__(self) -> None:
        if type(self.required_count) is not int or self.required_count <= 0:
            raise ValueError("required_count must be a positive integer")


@dataclass(frozen=True)
class MiniBatch:
    """One aligned minibatch, identified by its non-recycled generation.

    ``valid_rows`` is normally 4096.  The final global minibatch may be a
    shorter 256-row-aligned tail, matching CUDA's dynamic readiness counts.
    """

    generation: int
    valid_rows: int = MINIBATCH_ROWS

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if (
            type(self.valid_rows) is not int
            or not 0 < self.valid_rows <= MINIBATCH_ROWS
            or self.valid_rows % ROW_ALIGNMENT
        ):
            raise ValueError(
                "valid_rows must be a positive 256-row-aligned value at most 4096"
            )

    @property
    def macro_index(self) -> int:
        return self.generation // MINIBATCHES_PER_MACROBATCH

    @property
    def mini_index(self) -> int:
        return self.generation % MINIBATCHES_PER_MACROBATCH

    @property
    def ring_slot(self) -> int:
        return self.mini_index

    @property
    def row_begin(self) -> int:
        return self.generation * MINIBATCH_ROWS

    @property
    def fc1_row_tiles(self) -> int:
        return self.valid_rows // FC1_TILE_M

    @property
    def down_row_tiles(self) -> int:
        return self.valid_rows // DOWN_TILE_M

    @property
    def dispatch_tasks(self) -> int:
        return self.valid_rows // DISPATCH_TILE_M * DISPATCH_COLUMN_TILES

    @property
    def fc1_tasks(self) -> int:
        return self.fc1_row_tiles * FC1_COLUMN_TILES

    @property
    def down_tasks(self) -> int:
        return self.down_row_tiles * DOWN_COLUMN_TILES

    @property
    def combine_raw_tiles(self) -> int:
        return self.valid_rows // COMBINE_TILE_M * COMBINE_COLUMN_TILES

    @property
    def y_done_parts(self) -> int:
        return self.valid_rows // Y_DONE_ROW_BLOCK


@dataclass(frozen=True)
class TaskClaim:
    """One CTA-local comm task or one two-CTA compute-cluster task.

    Compute tasks use CUDA's raw task order: expert segments are visited in
    ascending expert order, then each expert/minibatch intersection applies
    the ThunderKittens supergroup-8 row/column swizzle.  ``row_tile`` remains
    minibatch-local; ``expert_index`` records the owning padded expert.
    """

    role: str
    generation: int
    task_index: int
    row_tile: int
    column_tile: int
    waits: tuple[ReadyKey, ...]
    arrival: ReadyKey
    arrival_count: int = 1
    expert_index: int | None = None

    def __post_init__(self) -> None:
        if self.role not in (DISPATCH_ROLE, FC1_ROLE, DOWN_ROLE, COMBINE_ROLE):
            raise ValueError(f"unknown task role: {self.role}")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if type(self.task_index) is not int or self.task_index < 0:
            raise ValueError("task_index must be a non-negative integer")
        if type(self.row_tile) is not int or self.row_tile < 0:
            raise ValueError("row_tile must be a non-negative integer")
        if type(self.column_tile) is not int or self.column_tile < 0:
            raise ValueError("column_tile must be a non-negative integer")
        if type(self.arrival_count) is not int or self.arrival_count <= 0:
            raise ValueError("arrival_count must be a positive integer")
        if self.expert_index is not None and (
            type(self.expert_index) is not int
            or not 0 <= self.expert_index < NUM_LOCAL_EXPERTS
        ):
            raise ValueError("expert_index is outside the local expert range")
        if self.arrival.generation != self.generation:
            raise ABAHazardError("task arrival uses a different generation")


@dataclass(frozen=True)
class SharedTaskClaim:
    """One raw shared-expert FC1 or Down cluster task."""

    role: str
    task_index: int
    row_tile: int
    column_tile: int
    hidden_ready_index: int
    hidden_ready_required: int

    def __post_init__(self) -> None:
        if self.role not in (FC1_ROLE, DOWN_ROLE):
            raise ValueError("shared task role must be fc1 or down")
        for name, value in (
            ("task_index", self.task_index),
            ("row_tile", self.row_tile),
            ("column_tile", self.column_tile),
            ("hidden_ready_index", self.hidden_ready_index),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.hidden_ready_required) is not int or self.hidden_ready_required < 0:
            raise ValueError("hidden_ready_required must be a non-negative integer")


@dataclass(frozen=True)
class ReuseGuard:
    """Exact release needed before an earlier generation reuses one ring slot."""

    buffer: str
    next_generation: int
    prior_generation: int
    wait: ReadyKey

    def __post_init__(self) -> None:
        if self.buffer not in ("x", "y"):
            raise ValueError("buffer must be 'x' or 'y'")
        if type(self.next_generation) is not int or self.next_generation < 0:
            raise ValueError("next_generation must be a non-negative integer")
        if type(self.prior_generation) is not int:
            raise ValueError("prior_generation must be an integer")
        expected_prior = self.next_generation + MINIBATCHES_PER_MACROBATCH
        if self.prior_generation != expected_prior:
            raise ABAHazardError("ring reuse must name the immediately prior generation")
        if self.prior_generation % MINIBATCHES_PER_MACROBATCH != (
            self.next_generation % MINIBATCHES_PER_MACROBATCH
        ):
            raise ABAHazardError("ring reuse generations do not share a slot")
        expected_kind = Y_READY if self.buffer == "x" else Y_DONE
        if self.wait.kind != expected_kind or self.wait.generation != expected_prior:
            raise ABAHazardError("ring release key carries a stale generation")
        if self.buffer == "x" and self.wait.part != 0:
            raise ABAHazardError("x reuse is guarded by minibatch-wide y_ready")
        if self.buffer == "y" and not 0 <= self.wait.part < Y_DONE_PARTS:
            raise ABAHazardError("y reuse guard has an invalid row-block part")

    @property
    def ring_slot(self) -> int:
        return self.next_generation % MINIBATCHES_PER_MACROBATCH


def reverse_minibatches(total_rows: int) -> tuple[MiniBatch, ...]:
    """Return aligned minis in reverse-macro, forward-within-macro order."""

    if type(total_rows) is not int or total_rows < 0:
        raise ValueError("total_rows must be a non-negative integer")
    if total_rows % ROW_ALIGNMENT:
        raise UnsupportedTailError(
            f"total_rows must be divisible by {ROW_ALIGNMENT}"
        )
    num_minibatches = (
        total_rows + MINIBATCH_ROWS - 1
    ) // MINIBATCH_ROWS
    num_macrobatches = (
        num_minibatches + MINIBATCHES_PER_MACROBATCH - 1
    ) // MINIBATCHES_PER_MACROBATCH
    ordered: list[MiniBatch] = []
    for macro_index in range(num_macrobatches - 1, -1, -1):
        first = macro_index * MINIBATCHES_PER_MACROBATCH
        end = min(first + MINIBATCHES_PER_MACROBATCH, num_minibatches)
        ordered.extend(
            MiniBatch(
                generation,
                min(MINIBATCH_ROWS, total_rows - generation * MINIBATCH_ROWS),
            )
            for generation in range(first, end)
        )
    return tuple(ordered)


def readiness_specs(minibatch: MiniBatch) -> tuple[ReadySpec, ...]:
    """Return every counter and exact required arrival count for one mini."""

    generation = minibatch.generation
    return (
        ReadySpec(
            ReadyKey(X_READY, generation),
            minibatch.dispatch_tasks,
        ),
        *(
            ReadySpec(
                ReadyKey(HIDDEN_READY, generation, row_tile),
                HIDDEN_READY_REQUIRED,
            )
            for row_tile in range(minibatch.fc1_row_tiles)
        ),
        ReadySpec(
            ReadyKey(Y_READY, generation),
            minibatch.down_tasks * CLUSTER_SIZE,
        ),
        *(
            ReadySpec(ReadyKey(Y_DONE, generation, part), Y_DONE_REQUIRED)
            for part in range(minibatch.y_done_parts)
        ),
    )


def reuse_guards(
    minibatch: MiniBatch,
    schedule: tuple[MiniBatch, ...],
) -> tuple[ReuseGuard, ...]:
    """Return exact X/Y ring release keys, or no guards for first occupancy."""

    by_generation = {item.generation: item for item in schedule}
    if len(by_generation) != len(schedule):
        raise ValueError("schedule generations must be unique")
    generation = minibatch.generation
    if by_generation.get(generation) != minibatch:
        raise ValueError("minibatch generation is outside the schedule")
    prior = generation + MINIBATCHES_PER_MACROBATCH
    prior_minibatch = by_generation.get(prior)
    if prior_minibatch is None:
        return ()
    return (
        ReuseGuard("x", generation, prior, ReadyKey(Y_READY, prior)),
        *(
            ReuseGuard("y", generation, prior, ReadyKey(Y_DONE, prior, part))
            # Only rows written by both generations alias.  The last macro's
            # tail may own fewer row blocks than the later full macro that
            # reuses its ring slot.
            for part in range(
                min(minibatch.y_done_parts, prior_minibatch.y_done_parts)
            )
        ),
    )


def dispatch_claims(
    minibatch: MiniBatch,
    schedule: tuple[MiniBatch, ...],
) -> tuple[TaskClaim, ...]:
    guards = reuse_guards(minibatch, schedule)
    x_release = guards[0].wait if guards else None
    prior = {
        item.generation: item for item in schedule
    }.get(minibatch.generation + MINIBATCHES_PER_MACROBATCH)
    claims = []
    for task_index in range(minibatch.dispatch_tasks):
        row_tile = task_index // DISPATCH_COLUMN_TILES
        # A short prior tail only occupied its leading rows.  Match CUDA's
        # overlap predicate so non-aliasing rows of the next full mini can be
        # dispatched immediately instead of waiting for the whole prior mini.
        overlaps_prior = (
            prior is not None
            and row_tile * DISPATCH_TILE_M < prior.valid_rows
        )
        waits = (x_release,) if overlaps_prior and x_release is not None else ()
        claims.append(
            TaskClaim(
                DISPATCH_ROLE,
                minibatch.generation,
                task_index,
                row_tile,
                task_index % DISPATCH_COLUMN_TILES,
                waits,
                ReadyKey(X_READY, minibatch.generation),
            )
        )
    return tuple(claims)


def swizzled_mlp_tile_coord(
    row_blocks: int,
    column_blocks: int,
    task_index: int,
) -> tuple[int, int]:
    """Mirror ``get_swizzled_2d_idx<8>`` used by both CUDA GEMMs."""

    if type(row_blocks) is not int or row_blocks <= 0:
        raise ValueError("row_blocks must be a positive integer")
    if type(column_blocks) is not int or column_blocks <= 0:
        raise ValueError("column_blocks must be a positive integer")
    if (
        type(task_index) is not int
        or not 0 <= task_index < row_blocks * column_blocks
    ):
        raise ValueError("task_index is outside the MLP tile grid")

    supergroup_numel = row_blocks * MLP_SUPERGROUP_SIZE
    supergroup_index = task_index // supergroup_numel
    supersection_columns = (
        column_blocks // MLP_SUPERGROUP_SIZE
    ) * MLP_SUPERGROUP_SIZE
    supersection_numel = row_blocks * supersection_columns
    if task_index < supersection_numel:
        row = (task_index % supergroup_numel) // MLP_SUPERGROUP_SIZE
        column = (
            supergroup_index * MLP_SUPERGROUP_SIZE
            + task_index % MLP_SUPERGROUP_SIZE
        )
    else:
        remainder_columns = column_blocks - supersection_columns
        remainder_task = task_index - supersection_numel
        row = remainder_task // remainder_columns
        column = supersection_columns + remainder_task % remainder_columns
    if supergroup_index % 2:
        row = row_blocks - row - 1
    return row, column


def _validated_expert_rows(
    tokens_per_expert: tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(tokens_per_expert, tuple):
        raise TypeError("tokens_per_expert must be a tuple")
    if len(tokens_per_expert) != NUM_LOCAL_EXPERTS:
        raise ValueError(
            f"tokens_per_expert must contain {NUM_LOCAL_EXPERTS} local experts"
        )
    if any(
        type(rows) is not int or rows < 0 or rows % ROW_ALIGNMENT
        for rows in tokens_per_expert
    ):
        raise ValueError(
            "every tokens_per_expert entry must be a non-negative "
            "256-row-aligned integer"
        )
    return tokens_per_expert


def _default_expert_rows(
    schedule: tuple[MiniBatch, ...],
) -> tuple[int, ...]:
    """Use one populated expert when a test only exercises DAG geometry."""

    total_rows = sum(minibatch.valid_rows for minibatch in schedule)
    return (total_rows, *(0 for _ in range(NUM_LOCAL_EXPERTS - 1)))


def decoded_mlp_tasks(
    minibatch: MiniBatch,
    tokens_per_expert: tuple[int, ...],
    column_blocks: int,
) -> tuple[tuple[int, int, int], ...]:
    """Decode all raw tasks as ``(expert, minibatch_row, column)``.

    This is the CPU form of the expert loop in ``fused_gate_up.cuh`` and
    ``grouped_gemm.cuh``.  The swizzle restarts for every expert/minibatch
    intersection, exactly as in the device implementation.
    """

    expert_rows = _validated_expert_rows(tokens_per_expert)
    if type(column_blocks) is not int or column_blocks <= 0:
        raise ValueError("column_blocks must be a positive integer")
    total_rows = sum(expert_rows)
    minibatch_first_block = minibatch.row_begin // ROW_ALIGNMENT
    minibatch_end_block = min(
        minibatch_first_block + FC1_ROW_TILES,
        total_rows // ROW_ALIGNMENT,
    )
    if (
        minibatch_end_block - minibatch_first_block
        != minibatch.valid_rows // ROW_ALIGNMENT
    ):
        raise ValueError("minibatch extent does not match tokens_per_expert")

    decoded: list[tuple[int, int, int]] = []
    expert_first_block = 0
    for expert_index, rows in enumerate(expert_rows):
        expert_blocks = rows // ROW_ALIGNMENT
        first = max(minibatch_first_block, expert_first_block)
        end = min(minibatch_end_block, expert_first_block + expert_blocks)
        segment_blocks = max(0, end - first)
        for segment_task in range(segment_blocks * column_blocks):
            local_row, column = swizzled_mlp_tile_coord(
                segment_blocks,
                column_blocks,
                segment_task,
            )
            decoded.append(
                (
                    expert_index,
                    first + local_row - minibatch_first_block,
                    column,
                )
            )
        expert_first_block += expert_blocks

    expected = minibatch.valid_rows // ROW_ALIGNMENT * column_blocks
    if len(decoded) != expected:
        raise ValueError("tokens_per_expert does not cover the minibatch")
    return tuple(decoded)


def shared_compute_claims(num_local_tokens: int) -> tuple[SharedTaskClaim, ...]:
    """Mirror CUDA's shared fused-FC1 then shared-Down raw task order."""

    if (
        type(num_local_tokens) is not int
        or num_local_tokens < FC1_TILE_M
        or num_local_tokens % FC1_TILE_M
    ):
        raise ValueError("num_local_tokens must be a positive multiple of 256")
    row_blocks = num_local_tokens // FC1_TILE_M
    fc1 = tuple(
        SharedTaskClaim(
            FC1_ROLE,
            task_index,
            row_tile,
            column_tile,
            row_tile,
            0,
        )
        for task_index in range(row_blocks * FC1_COLUMN_TILES)
        for row_tile, column_tile in (
            swizzled_mlp_tile_coord(
                row_blocks,
                FC1_COLUMN_TILES,
                task_index,
            ),
        )
    )
    down = tuple(
        SharedTaskClaim(
            DOWN_ROLE,
            task_index,
            row_tile,
            column_tile,
            row_tile,
            HIDDEN_READY_REQUIRED,
        )
        for task_index in range(row_blocks * DOWN_COLUMN_TILES)
        for row_tile, column_tile in (
            swizzled_mlp_tile_coord(
                row_blocks,
                DOWN_COLUMN_TILES,
                task_index,
            ),
        )
    )
    return fc1 + down


def compute_claims(
    minibatch: MiniBatch,
    schedule: tuple[MiniBatch, ...],
    tokens_per_expert: tuple[int, ...] | None = None,
) -> tuple[TaskClaim, ...]:
    """Claim CUDA-ordered fused-FC1 tiles before CUDA-ordered Down tiles."""

    generation = minibatch.generation
    expert_rows = (
        _default_expert_rows(schedule)
        if tokens_per_expert is None
        else _validated_expert_rows(tokens_per_expert)
    )
    schedule_rows = sum(item.valid_rows for item in schedule)
    if sum(expert_rows) != schedule_rows:
        raise ValueError("tokens_per_expert must sum to the schedule row count")
    fc1_tiles = decoded_mlp_tasks(
        minibatch,
        expert_rows,
        FC1_COLUMN_TILES,
    )
    fc1 = tuple(
        TaskClaim(
            FC1_ROLE,
            generation,
            task_index,
            row_tile,
            column_tile,
            (ReadyKey(X_READY, generation),),
            ReadyKey(HIDDEN_READY, generation, row_tile),
            CLUSTER_SIZE,
            expert_index,
        )
        for task_index, (expert_index, row_tile, column_tile) in enumerate(fc1_tiles)
    )
    guards = reuse_guards(minibatch, schedule)
    y_release = tuple(guard.wait for guard in guards if guard.buffer == "y")
    down = []
    down_tiles = decoded_mlp_tasks(
        minibatch,
        expert_rows,
        DOWN_COLUMN_TILES,
    )
    for task_index, (expert_index, row_tile, column_tile) in enumerate(down_tiles):
        waits = [ReadyKey(HIDDEN_READY, generation, row_tile)]
        if y_release:
            waits.extend(y_release[2 * row_tile : 2 * row_tile + 2])
        down.append(
            TaskClaim(
                DOWN_ROLE,
                generation,
                task_index,
                row_tile,
                column_tile,
                tuple(waits),
                ReadyKey(Y_READY, generation),
                CLUSTER_SIZE,
                expert_index,
            )
        )
    return fc1 + tuple(down)


def combine_claims(minibatch: MiniBatch) -> tuple[TaskClaim, ...]:
    generation = minibatch.generation
    claims = []
    for task_index in range(minibatch.combine_raw_tiles):
        row_tile = task_index // COMBINE_COLUMN_TILES
        part = row_tile * COMBINE_TILE_M // Y_DONE_ROW_BLOCK
        claims.append(
            TaskClaim(
                COMBINE_ROLE,
                generation,
                task_index,
                row_tile,
                task_index % COMBINE_COLUMN_TILES,
                (ReadyKey(Y_READY, generation),),
                ReadyKey(Y_DONE, generation, part),
            )
        )
    return tuple(claims)


def comm_claims(
    role: str,
    minibatch: MiniBatch,
    schedule: tuple[MiniBatch, ...],
) -> tuple[TaskClaim, ...]:
    """Build one explicit communication role; a mixed ``comm`` role is invalid."""

    if role == DISPATCH_ROLE:
        return dispatch_claims(minibatch, schedule)
    if role == COMBINE_ROLE:
        return combine_claims(minibatch)
    raise ValueError("communication role must be 'dispatch' or 'combine'")


def validate_contract(
    total_rows: int,
    tokens_per_expert: tuple[int, ...] | None = None,
) -> tuple[MiniBatch, ...]:
    """Validate unique keys and exact producer counts for the whole schedule."""

    schedule = reverse_minibatches(total_rows)
    if tokens_per_expert is not None:
        expert_rows = _validated_expert_rows(tokens_per_expert)
        if sum(expert_rows) != total_rows:
            raise ValueError("tokens_per_expert must sum to total_rows")
    specs = tuple(spec for minibatch in schedule for spec in readiness_specs(minibatch))
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError("readiness keys are not globally unique")
    if len(keys) != len({key.linear_id for key in keys}):
        raise AssertionError("flat readiness IDs are not globally unique")

    arrivals: Counter[ReadyKey] = Counter()
    required = {spec.key: spec.required_count for spec in specs}
    for minibatch in schedule:
        claims = (
            dispatch_claims(minibatch, schedule)
            + compute_claims(minibatch, schedule, tokens_per_expert)
            + combine_claims(minibatch)
        )
        for claim in claims:
            if any(wait not in required for wait in claim.waits):
                raise AssertionError("task wait is outside the full-context key space")
            arrivals[claim.arrival] += claim.arrival_count
    if dict(arrivals) != required:
        raise AssertionError("task arrivals do not match readiness requirements")
    return schedule
