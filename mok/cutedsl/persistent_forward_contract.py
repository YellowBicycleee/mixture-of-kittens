"""CPU-testable contract for the CUDA-shaped persistent CuTe forward core.

This module mirrors the role split and task numbering in
``csrc/mok_megakernel.cuh``.  It intentionally contains no torch or CUTLASS
imports so scheduler changes can be checked before spending a B300 lease.

The contract is not a second definition of the MoE math.  It only describes
the fixed Qwen BF16/EP8 launch supported by the first CuTe port:

* a fixed prefix of communication clusters (two CTAs per cluster);
* CLC work stealing only for the remaining compute clusters;
* shared Gate/Up/SwiGLU/Down followed by routed minibatches in reverse
  macrobatch order; and
* the five monotonic GPU-scope counters used by the CUDA forward DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .forward_contract import (
    COMBINE_TILE_COLUMNS,
    COMBINE_TILE_ROWS,
    DISPATCH_TILE_COLUMNS,
    DISPATCH_TILE_ROWS,
    EP_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    ROW_ALIGNMENT,
    TOPK,
    validate_num_comm_sms,
)


CLUSTER_SIZE = 2
MLP_TILE_ROWS = 256
MLP_TILE_COLUMNS = 256
MLP_BF16_K_TILE = 64
MLP_LOAD_PIPE_DEPTH = 6
MLP_EPILOGUE_SUBTILES = 8
MLP_OUTPUT_SMEM_RING = 3
SWIGLU_TILE_ROWS = 128
SWIGLU_TILE_COLUMNS = 128
SWIGLU_PIPE_DEPTH = 3
COMBINE_PIPE_DEPTH = 7
CLC_PIPE_DEPTH = 1
CLC_RESPONSE_BYTES = 16
CLC_SCHEDULER_WARP = 5
CLC_COMPLETION_WARPS = CLUSTER_SIZE * 8
CLC_DRAIN_WARPS = 8
CLC_DRAIN_PIPE_DEPTH = CLC_DRAIN_WARPS
CLC_TERMINAL_CLUSTER_SYNC_REQUIRED = True
QWEN_NUM_LOCAL_TOKENS = 7168
QWEN_MACROBATCH_SIZE = 131072
QWEN_MINIBATCH_SIZE = 4096
QWEN_NUM_COMM_SMS = 40
QWEN_SCHEDULE_CAPACITY_FACTOR = max(2, EP_SIZE // 2)
QWEN_SCHEDULE_CAPACITY = (
    QWEN_NUM_LOCAL_TOKENS * TOPK * QWEN_SCHEDULE_CAPACITY_FACTOR
)


class ForwardTaskKind(str, Enum):
    SHARED_GATE = "shared_gate"
    SHARED_UP = "shared_up"
    SHARED_SWIGLU = "shared_swiglu"
    SHARED_DOWN = "shared_down"
    ROUTED_GATE = "routed_gate"
    ROUTED_UP = "routed_up"
    ROUTED_SWIGLU = "routed_swiglu"
    ROUTED_DOWN = "routed_down"


class PhysicalRole(str, Enum):
    FIXED_COMM = "fixed_comm"
    CLC_COMPUTE = "clc_compute"


class CounterName(str, Enum):
    GATE_UP_TILE_READY = "gate_up_tile_ready"
    HIDDEN_ROW_BLOCK_READY = "hidden_row_block_ready"
    X_ROUTED_READY = "x_routed_ready"
    Y_ROUTED_READY = "y_routed_ready"
    Y_ROUTED_DONE = "y_routed_done"


@dataclass(frozen=True)
class PhysicalTaskShape:
    """Logical cluster tile and the maximum work owned by one physical CTA."""

    logical_tile: tuple[int, int]
    cta_tile: tuple[int, int]
    tiles_per_cta: int
    cooperative_cluster: bool


@dataclass(frozen=True)
class PhysicalCTA:
    role: PhysicalRole
    cluster_index: int
    cta_rank: int
    comm_cta_index: int | None


@dataclass(frozen=True)
class CounterContract:
    name: CounterName
    length: int
    required_arrivals: int
    producer: str
    consumer: str


@dataclass(frozen=True)
class RingReuseDependency:
    """One cross-macrobatch edge protecting a repeated local ring slot."""

    buffer: str
    local_row_start: int
    protected_macrobatch: int
    overwriting_macrobatch: int
    counter: CounterName
    counter_index: int
    required_arrivals: int
    observed_arrivals: int
    producer: str
    consumer: str


@dataclass(frozen=True)
class RingReuseSimulation:
    reverse_macrobatches: tuple[int, ...]
    dependencies: tuple[RingReuseDependency, ...]

    @property
    def arrivals_satisfy_waits(self) -> bool:
        return all(
            edge.observed_arrivals >= edge.required_arrivals
            for edge in self.dependencies
        )

    @property
    def generations_are_distinct(self) -> bool:
        generations: dict[tuple[str, int], list[int]] = {}
        for edge in self.dependencies:
            generations.setdefault(
                (edge.buffer, edge.local_row_start), []
            ).append(edge.counter_index)
        return all(
            len(indices) == len(set(indices))
            for indices in generations.values()
        )


@dataclass(frozen=True)
class CounterShapes:
    """Lengths of CUDA's five zero-initialized forward counter arrays."""

    gate_up_tile_ready: int
    hidden_row_block_ready: int
    x_routed_ready: int
    y_routed_ready: int
    y_routed_done: int


@dataclass(frozen=True)
class ComputeTask:
    """One logical two-CTA compute task after removing the comm prefix."""

    kind: ForwardTaskKind
    task_index: int
    macrobatch_index: int | None = None
    minibatch_index: int | None = None


@dataclass(frozen=True)
class ForwardGeometry:
    """Static task geometry shared by the host seam and device scheduler."""

    num_local_tokens: int
    schedule_capacity: int
    macrobatch_size: int
    minibatch_size: int
    num_comm_sms: int
    shared_gate_up_tasks: int
    shared_swiglu_tasks: int
    shared_down_tasks: int
    minibatch_routed_gate_up_tasks: int
    minibatch_routed_swiglu_tasks: int
    minibatch_routed_down_tasks: int
    counters: CounterShapes

    @property
    def comm_clusters(self) -> int:
        return self.num_comm_sms // CLUSTER_SIZE

    @property
    def shared_tasks(self) -> int:
        return (
            2 * self.shared_gate_up_tasks
            + self.shared_swiglu_tasks
            + self.shared_down_tasks
        )

    @property
    def minibatch_tasks(self) -> int:
        return (
            2 * self.minibatch_routed_gate_up_tasks
            + self.minibatch_routed_swiglu_tasks
            + self.minibatch_routed_down_tasks
        )

    @property
    def minibatches_per_macrobatch(self) -> int:
        return self.macrobatch_size // self.minibatch_size

    @property
    def capacity_num_minibatches(self) -> int:
        return _ceil_div(self.schedule_capacity, self.minibatch_size)

    @property
    def capacity_compute_clusters(self) -> int:
        """Compute tasks in CUDA's host launch envelope (capacity based)."""

        return self.shared_tasks + self.capacity_num_minibatches * self.minibatch_tasks

    @property
    def capacity_launch_clusters(self) -> int:
        return self.comm_clusters + self.capacity_compute_clusters

    @property
    def capacity_launch_ctas(self) -> int:
        """CTA grid passed by the host before ``num_tokens`` is read on device."""

        return self.capacity_launch_clusters * CLUSTER_SIZE

    def num_macrobatches(self, num_tokens: int) -> int:
        self._validate_num_tokens(num_tokens)
        return _ceil_div(num_tokens, self.macrobatch_size)

    def true_num_minibatches(self, num_tokens: int) -> int:
        self._validate_num_tokens(num_tokens)
        return _ceil_div(num_tokens, self.minibatch_size)

    def true_runtime_compute_clusters(self, num_tokens: int) -> int:
        return self.shared_tasks + self.true_num_minibatches(num_tokens) * self.minibatch_tasks

    def true_runtime_clusters(self, num_tokens: int) -> int:
        """Active prefix after the kernel reads the device ``num_tokens``."""

        return self.comm_clusters + self.true_runtime_compute_clusters(num_tokens)

    def true_runtime_ctas(self, num_tokens: int) -> int:
        return self.true_runtime_clusters(num_tokens) * CLUSTER_SIZE

    def is_fixed_comm_cluster(self, cluster_index: int) -> bool:
        self._validate_capacity_cluster_index(cluster_index)
        return cluster_index < self.comm_clusters

    def is_runtime_active_cluster(self, cluster_index: int, num_tokens: int) -> bool:
        self._validate_capacity_cluster_index(cluster_index)
        return cluster_index < self.true_runtime_clusters(num_tokens)

    def comm_cta_index(
        self,
        cluster_index: int,
        cta_rank: int,
    ) -> int:
        if not self.is_fixed_comm_cluster(cluster_index):
            raise ValueError("cluster_index is not in the fixed communication prefix")
        if type(cta_rank) is not int or not 0 <= cta_rank < CLUSTER_SIZE:
            raise ValueError(f"cta_rank must be in [0, {CLUSTER_SIZE})")
        return cluster_index * CLUSTER_SIZE + cta_rank

    def physical_cta(self, cluster_index: int, cta_rank: int) -> PhysicalCTA:
        self._validate_capacity_cluster_index(cluster_index)
        if type(cta_rank) is not int or not 0 <= cta_rank < CLUSTER_SIZE:
            raise ValueError(f"cta_rank must be in [0, {CLUSTER_SIZE})")
        if cluster_index < self.comm_clusters:
            return PhysicalCTA(
                PhysicalRole.FIXED_COMM,
                cluster_index,
                cta_rank,
                cluster_index * CLUSTER_SIZE + cta_rank,
            )
        return PhysicalCTA(
            PhysicalRole.CLC_COMPUTE,
            cluster_index,
            cta_rank,
            None,
        )

    def decode_compute_task(self, compute_cluster_index: int, num_tokens: int) -> ComputeTask:
        """Mirror the forward task ladder and reverse macro mapping in CUDA."""

        total = self.true_runtime_compute_clusters(num_tokens)
        if type(compute_cluster_index) is not int or not 0 <= compute_cluster_index < total:
            raise ValueError("compute_cluster_index is outside the true compute task grid")

        index = compute_cluster_index
        if index < self.shared_gate_up_tasks:
            return ComputeTask(ForwardTaskKind.SHARED_GATE, index)
        index -= self.shared_gate_up_tasks
        if index < self.shared_gate_up_tasks:
            return ComputeTask(ForwardTaskKind.SHARED_UP, index)
        index -= self.shared_gate_up_tasks
        if index < self.shared_swiglu_tasks:
            return ComputeTask(ForwardTaskKind.SHARED_SWIGLU, index)
        index -= self.shared_swiglu_tasks
        if index < self.shared_down_tasks:
            return ComputeTask(ForwardTaskKind.SHARED_DOWN, index)
        index -= self.shared_down_tasks

        task_ordered_global_minibatch = index // self.minibatch_tasks
        minibatch_task_index = index % self.minibatch_tasks
        macrobatch_index, minibatch_index = self._decode_reverse_minibatch(
            task_ordered_global_minibatch,
            num_tokens,
        )

        if minibatch_task_index < self.minibatch_routed_gate_up_tasks:
            kind = ForwardTaskKind.ROUTED_GATE
            task_index = minibatch_task_index
        elif minibatch_task_index < 2 * self.minibatch_routed_gate_up_tasks:
            kind = ForwardTaskKind.ROUTED_UP
            task_index = minibatch_task_index - self.minibatch_routed_gate_up_tasks
        elif minibatch_task_index < (
            2 * self.minibatch_routed_gate_up_tasks
            + self.minibatch_routed_swiglu_tasks
        ):
            kind = ForwardTaskKind.ROUTED_SWIGLU
            task_index = minibatch_task_index - 2 * self.minibatch_routed_gate_up_tasks
        else:
            kind = ForwardTaskKind.ROUTED_DOWN
            task_index = (
                minibatch_task_index
                - 2 * self.minibatch_routed_gate_up_tasks
                - self.minibatch_routed_swiglu_tasks
            )
        return ComputeTask(kind, task_index, macrobatch_index, minibatch_index)

    def decode_cluster_task(self, cluster_index: int, num_tokens: int) -> ComputeTask:
        """Decode a compute-suffix cluster; fixed comm clusters are rejected."""

        self._validate_runtime_cluster_index(cluster_index, num_tokens)
        if cluster_index < self.comm_clusters:
            raise ValueError("fixed communication clusters do not enter the CLC task decoder")
        return self.decode_compute_task(cluster_index - self.comm_clusters, num_tokens)

    def gate_up_counter_index(
        self,
        *,
        is_shared: bool,
        global_row_block: int,
        column_block: int,
    ) -> int:
        """Index one 256x256 Gate/Up output tile in the shared+routed array."""

        row_blocks = (
            self.num_local_tokens if is_shared else self.schedule_capacity
        ) // MLP_TILE_ROWS
        column_blocks = INTERMEDIATE_SIZE // MLP_TILE_COLUMNS
        if type(global_row_block) is not int or not 0 <= global_row_block < row_blocks:
            raise ValueError("global_row_block is outside the Gate/Up counter grid")
        if type(column_block) is not int or not 0 <= column_block < column_blocks:
            raise ValueError("column_block is outside the Gate/Up counter grid")
        base = 0 if is_shared else self.shared_gate_up_tasks
        return base + global_row_block * column_blocks + column_block

    def hidden_counter_index(
        self,
        *,
        is_shared: bool,
        global_row_block: int,
    ) -> int:
        """Index one 256-row block produced by SwiGLU and consumed by Down."""

        row_blocks = (
            self.num_local_tokens if is_shared else self.schedule_capacity
        ) // MLP_TILE_ROWS
        if type(global_row_block) is not int or not 0 <= global_row_block < row_blocks:
            raise ValueError("global_row_block is outside the hidden counter grid")
        shared_row_blocks = self.num_local_tokens // MLP_TILE_ROWS
        return (0 if is_shared else shared_row_blocks) + global_row_block

    def minibatch_counter_index(self, global_row: int) -> int:
        if type(global_row) is not int or not 0 <= global_row < self.schedule_capacity:
            raise ValueError("global_row is outside the routed schedule")
        return global_row // self.minibatch_size

    def y_done_counter_index(self, global_row: int) -> int:
        if type(global_row) is not int or not 0 <= global_row < self.schedule_capacity:
            raise ValueError("global_row is outside the routed schedule")
        return global_row // (MLP_TILE_ROWS // CLUSTER_SIZE)

    def _decode_reverse_minibatch(
        self,
        task_ordered_global_minibatch: int,
        num_tokens: int,
    ) -> tuple[int, int]:
        num_macrobatches = self.num_macrobatches(num_tokens)
        true_num_minibatches = self.true_num_minibatches(num_tokens)
        if num_macrobatches == 0:
            raise ValueError("a zero-token schedule has no routed minibatch task")
        if not 0 <= task_ordered_global_minibatch < true_num_minibatches:
            raise ValueError("task-ordered minibatch index is outside the true schedule")

        last_macro_minibatches = true_num_minibatches - (
            num_macrobatches - 1
        ) * self.minibatches_per_macrobatch
        if task_ordered_global_minibatch < last_macro_minibatches:
            return num_macrobatches - 1, task_ordered_global_minibatch

        index = task_ordered_global_minibatch - last_macro_minibatches
        return (
            num_macrobatches - 2 - index // self.minibatches_per_macrobatch,
            index % self.minibatches_per_macrobatch,
        )

    def _validate_num_tokens(self, num_tokens: int) -> None:
        if type(num_tokens) is not int or not 0 <= num_tokens <= self.schedule_capacity:
            raise ValueError("num_tokens must be in [0, schedule_capacity]")
        if num_tokens % ROW_ALIGNMENT:
            raise ValueError(f"num_tokens must be divisible by {ROW_ALIGNMENT}")

    def _validate_capacity_cluster_index(self, cluster_index: int) -> None:
        total = self.capacity_launch_clusters
        if type(cluster_index) is not int or not 0 <= cluster_index < total:
            raise ValueError("cluster_index is outside the capacity launch grid")

    def _validate_runtime_cluster_index(self, cluster_index: int, num_tokens: int) -> None:
        total = self.true_runtime_clusters(num_tokens)
        if type(cluster_index) is not int or not 0 <= cluster_index < total:
            raise ValueError("cluster_index is outside the runtime-active grid")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def make_forward_geometry(
    *,
    num_local_tokens: int,
    schedule_capacity: int,
    macrobatch_size: int,
    minibatch_size: int,
    num_comm_sms: int,
) -> ForwardGeometry:
    """Build the exact BF16/Qwen task and counter geometry used by CUDA."""

    validate_num_comm_sms(num_comm_sms)
    for name, value in (
        ("num_local_tokens", num_local_tokens),
        ("schedule_capacity", schedule_capacity),
        ("macrobatch_size", macrobatch_size),
        ("minibatch_size", minibatch_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value % ROW_ALIGNMENT:
            raise ValueError(f"{name} must be divisible by {ROW_ALIGNMENT}")
    if macrobatch_size % minibatch_size:
        raise ValueError("macrobatch_size must be a multiple of minibatch_size")

    shared_row_blocks = num_local_tokens // MLP_TILE_ROWS
    routed_row_blocks = schedule_capacity // MLP_TILE_ROWS
    minibatch_routed_row_blocks = minibatch_size // MLP_TILE_ROWS
    intermediate_column_blocks = INTERMEDIATE_SIZE // MLP_TILE_COLUMNS
    hidden_column_blocks = HIDDEN_SIZE // MLP_TILE_COLUMNS

    shared_gate_up_tasks = shared_row_blocks * intermediate_column_blocks
    minibatch_routed_gate_up_tasks = (
        minibatch_routed_row_blocks * intermediate_column_blocks
    )
    shared_swiglu_tiles = (
        num_local_tokens // SWIGLU_TILE_ROWS
    ) * (INTERMEDIATE_SIZE // SWIGLU_TILE_COLUMNS)
    minibatch_routed_swiglu_tiles = (
        minibatch_size // SWIGLU_TILE_ROWS
    ) * (INTERMEDIATE_SIZE // SWIGLU_TILE_COLUMNS)
    swiglu_tiles_per_cluster_task = CLUSTER_SIZE * SWIGLU_PIPE_DEPTH

    num_capacity_minibatches = _ceil_div(schedule_capacity, minibatch_size)
    counters = CounterShapes(
        gate_up_tile_ready=(
            shared_gate_up_tasks
            + routed_row_blocks * intermediate_column_blocks
        ),
        hidden_row_block_ready=shared_row_blocks + routed_row_blocks,
        x_routed_ready=num_capacity_minibatches,
        y_routed_ready=num_capacity_minibatches,
        y_routed_done=schedule_capacity // (MLP_TILE_ROWS // CLUSTER_SIZE),
    )
    return ForwardGeometry(
        num_local_tokens=num_local_tokens,
        schedule_capacity=schedule_capacity,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
        shared_gate_up_tasks=shared_gate_up_tasks,
        shared_swiglu_tasks=_ceil_div(
            shared_swiglu_tiles,
            swiglu_tiles_per_cluster_task,
        ),
        shared_down_tasks=shared_row_blocks * hidden_column_blocks,
        minibatch_routed_gate_up_tasks=minibatch_routed_gate_up_tasks,
        minibatch_routed_swiglu_tasks=_ceil_div(
            minibatch_routed_swiglu_tiles,
            swiglu_tiles_per_cluster_task,
        ),
        minibatch_routed_down_tasks=(
            minibatch_routed_row_blocks * hidden_column_blocks
        ),
        counters=counters,
    )


def counter_required_counts(minibatch_rows: int) -> CounterShapes:
    """Return the acquire thresholds for one full/partial CUDA dependency unit.

    The fields reuse :class:`CounterShapes` names, but the values here are
    required monotonic arrival counts rather than allocation lengths.
    """

    if type(minibatch_rows) is not int or minibatch_rows <= 0:
        raise ValueError("minibatch_rows must be a positive integer")
    if minibatch_rows % ROW_ALIGNMENT:
        raise ValueError(f"minibatch_rows must be divisible by {ROW_ALIGNMENT}")
    return CounterShapes(
        gate_up_tile_ready=2 * CLUSTER_SIZE,
        hidden_row_block_ready=(
            MLP_TILE_ROWS // SWIGLU_TILE_ROWS
        ) * (INTERMEDIATE_SIZE // SWIGLU_TILE_COLUMNS),
        x_routed_ready=(
            _ceil_div(minibatch_rows, DISPATCH_TILE_ROWS)
            * _ceil_div(HIDDEN_SIZE, DISPATCH_TILE_COLUMNS)
        ),
        y_routed_ready=(
            _ceil_div(minibatch_rows, MLP_TILE_ROWS)
            * (HIDDEN_SIZE // MLP_TILE_COLUMNS)
            * CLUSTER_SIZE
        ),
        y_routed_done=(
            (MLP_TILE_ROWS // CLUSTER_SIZE // COMBINE_TILE_ROWS)
            * _ceil_div(HIDDEN_SIZE, COMBINE_TILE_COLUMNS)
        ),
    )


def task_physical_shape(kind: ForwardTaskKind | str) -> PhysicalTaskShape:
    """Return the physical CUDA work grain for one forward role."""

    if isinstance(kind, str) and kind in ("dispatch", "combine"):
        if kind == "dispatch":
            return PhysicalTaskShape(
                logical_tile=(DISPATCH_TILE_ROWS, DISPATCH_TILE_COLUMNS),
                cta_tile=(DISPATCH_TILE_ROWS, DISPATCH_TILE_COLUMNS),
                tiles_per_cta=1,
                cooperative_cluster=False,
            )
        return PhysicalTaskShape(
            logical_tile=(COMBINE_TILE_ROWS, COMBINE_TILE_COLUMNS),
            cta_tile=(COMBINE_TILE_ROWS, COMBINE_TILE_COLUMNS),
            tiles_per_cta=COMBINE_PIPE_DEPTH,
            cooperative_cluster=False,
        )

    task_kind = ForwardTaskKind(kind)
    if task_kind in (
        ForwardTaskKind.SHARED_SWIGLU,
        ForwardTaskKind.ROUTED_SWIGLU,
    ):
        return PhysicalTaskShape(
            logical_tile=(SWIGLU_TILE_ROWS, SWIGLU_TILE_COLUMNS),
            cta_tile=(SWIGLU_TILE_ROWS, SWIGLU_TILE_COLUMNS),
            tiles_per_cta=SWIGLU_PIPE_DEPTH,
            cooperative_cluster=False,
        )
    return PhysicalTaskShape(
        logical_tile=(MLP_TILE_ROWS, MLP_TILE_COLUMNS),
        cta_tile=(MLP_TILE_ROWS // CLUSTER_SIZE, MLP_TILE_COLUMNS),
        tiles_per_cta=1,
        cooperative_cluster=True,
    )


def counter_contracts(
    geometry: ForwardGeometry,
    minibatch_rows: int,
) -> tuple[CounterContract, ...]:
    """Describe allocation, threshold, producer, and consumer for all counters."""

    required = counter_required_counts(minibatch_rows)
    return (
        CounterContract(
            CounterName.GATE_UP_TILE_READY,
            geometry.counters.gate_up_tile_ready,
            required.gate_up_tile_ready,
            "Gate and Up GEMM: two CTA arrivals each",
            "SwiGLU parent 256x256 tile wait",
        ),
        CounterContract(
            CounterName.HIDDEN_ROW_BLOCK_READY,
            geometry.counters.hidden_row_block_ready,
            required.hidden_row_block_ready,
            "SwiGLU: one arrival per 128x128 tile",
            "Down GEMM 256-row input wait",
        ),
        CounterContract(
            CounterName.X_ROUTED_READY,
            geometry.counters.x_routed_ready,
            required.x_routed_ready,
            "Dispatch: one arrival per 128x512 tile",
            "Routed Gate and Up minibatch wait",
        ),
        CounterContract(
            CounterName.Y_ROUTED_READY,
            geometry.counters.y_routed_ready,
            required.y_routed_ready,
            "Down GEMM: one arrival per physical CTA output tile",
            "Combine wait and prior-macro Dispatch ring reuse",
        ),
        CounterContract(
            CounterName.Y_ROUTED_DONE,
            geometry.counters.y_routed_done,
            required.y_routed_done,
            "Combine: one arrival per 16x1024 tile",
            "Prior-macro Down epilogue 128-row ring reuse",
        ),
    )


def clc_logical_cluster_index(response_x: int) -> int:
    """Decode the cluster-aligned CLC CTA x coordinate used by CUDA."""

    if type(response_x) is not int or response_x < 0:
        raise ValueError("response_x must be a non-negative integer")
    if response_x % CLUSTER_SIZE:
        raise ValueError("response_x must identify the first CTA of a cluster")
    return response_x // CLUSTER_SIZE


def is_cta_local_task(kind: ForwardTaskKind) -> bool:
    return kind in (
        ForwardTaskKind.SHARED_SWIGLU,
        ForwardTaskKind.ROUTED_SWIGLU,
    )


def needs_cluster_sync_before(
    current: ForwardTaskKind,
    next_task: ForwardTaskKind | None,
) -> bool:
    """CUDA syncs only when leaving CTA-local SwiGLU for cluster GEMM."""

    return is_cta_local_task(current) and (
        next_task is not None and not is_cta_local_task(next_task)
    )


def needs_clc_drain(
    *,
    response_succeeded: bool,
    response_cluster_index: int,
    true_runtime_clusters: int,
) -> bool:
    """Successful but runtime-inactive CLC work needs the eight-warp drain."""

    if type(response_succeeded) is not bool:
        raise ValueError("response_succeeded must be bool")
    if type(response_cluster_index) is not int:
        raise ValueError("response_cluster_index must be an integer")
    if type(true_runtime_clusters) is not int or true_runtime_clusters < 0:
        raise ValueError("true_runtime_clusters must be non-negative")
    return (
        response_succeeded
        and response_cluster_index >= 0
        and response_cluster_index >= true_runtime_clusters
    )


def clc_response_lifecycle() -> tuple[str, ...]:
    """Depth-1 schedule phase shared by two CTAs and all sixteen warps."""

    return (
        "cta0_waits_for_reusable_depth1_stage",
        "cta0_issues_cluster_cancel_query",
        "both_ctas_expect_16_byte_response",
        "both_ctas_consume_same_logical_cluster",
        "sixteen_warps_arrive_schedule_finished",
        "advance_depth1_phase",
    )


def simulate_ring_reuse(
    geometry: ForwardGeometry,
    num_tokens: int,
) -> RingReuseSimulation:
    """Build cross-macro ring edges and check their monotonic generations.

    This is a dependency simulation, not a latency model.  It mirrors the two
    overwrite gates in CUDA: prior-macro Down completion protects ``x_routed``
    reuse, and prior-macro Combine completion protects ``y_routed`` reuse.
    """

    num_macrobatches = geometry.num_macrobatches(num_tokens)
    reverse_macrobatches = tuple(range(num_macrobatches - 1, -1, -1))
    required = counter_required_counts(geometry.minibatch_size)
    dependencies: list[RingReuseDependency] = []
    # Count concrete producer tiles independently from the wait-threshold
    # helper.  This keeps arrivals_satisfy_waits from comparing a value with
    # itself while remaining a small CPU-only dependency simulation.
    combine_arrivals_per_row_block = sum(
        1
        for _row in range(0, MLP_TILE_ROWS // CLUSTER_SIZE, COMBINE_TILE_ROWS)
        for _column in range(0, HIDDEN_SIZE, COMBINE_TILE_COLUMNS)
    )

    for overwriting_macro in range(num_macrobatches - 2, -1, -1):
        protected_macro = overwriting_macro + 1
        protected_offset = protected_macro * geometry.macrobatch_size
        overwriting_offset = overwriting_macro * geometry.macrobatch_size
        protected_rows = min(
            geometry.macrobatch_size,
            num_tokens - protected_offset,
        )
        overwriting_rows = min(
            geometry.macrobatch_size,
            num_tokens - overwriting_offset,
        )
        overlap_rows = min(protected_rows, overwriting_rows)

        for local_row in range(0, overlap_rows, DISPATCH_TILE_ROWS):
            global_row = protected_offset + local_row
            global_minibatch = geometry.minibatch_counter_index(global_row)
            minibatch_first_row = global_minibatch * geometry.minibatch_size
            minibatch_rows = min(
                geometry.minibatch_size,
                num_tokens - minibatch_first_row,
            )
            y_ready_required = counter_required_counts(
                minibatch_rows
            ).y_routed_ready
            down_cta_arrivals = sum(
                CLUSTER_SIZE
                for _row in range(0, minibatch_rows, MLP_TILE_ROWS)
                for _column in range(0, HIDDEN_SIZE, MLP_TILE_COLUMNS)
            )
            dependencies.append(
                RingReuseDependency(
                    buffer="x_routed",
                    local_row_start=local_row,
                    protected_macrobatch=protected_macro,
                    overwriting_macrobatch=overwriting_macro,
                    counter=CounterName.Y_ROUTED_READY,
                    counter_index=global_minibatch,
                    required_arrivals=y_ready_required,
                    observed_arrivals=down_cta_arrivals,
                    producer="protected-macro Down",
                    consumer="overwriting-macro Dispatch",
                )
            )

        for local_row in range(
            0,
            overlap_rows,
            MLP_TILE_ROWS // CLUSTER_SIZE,
        ):
            global_row = protected_offset + local_row
            dependencies.append(
                RingReuseDependency(
                    buffer="y_routed",
                    local_row_start=local_row,
                    protected_macrobatch=protected_macro,
                    overwriting_macrobatch=overwriting_macro,
                    counter=CounterName.Y_ROUTED_DONE,
                    counter_index=geometry.y_done_counter_index(global_row),
                    required_arrivals=required.y_routed_done,
                    observed_arrivals=combine_arrivals_per_row_block,
                    producer="protected-macro Combine",
                    consumer="overwriting-macro Down epilogue",
                )
            )

    return RingReuseSimulation(reverse_macrobatches, tuple(dependencies))


def qwen_default_geometry() -> ForwardGeometry:
    """Repository-default Qwen fixture used by the CUDA-parity contract."""

    return make_forward_geometry(
        num_local_tokens=QWEN_NUM_LOCAL_TOKENS,
        schedule_capacity=QWEN_SCHEDULE_CAPACITY,
        macrobatch_size=QWEN_MACROBATCH_SIZE,
        minibatch_size=QWEN_MINIBATCH_SIZE,
        num_comm_sms=QWEN_NUM_COMM_SMS,
    )


def nine_output_shapes(
    num_local_tokens: int,
    macrobatch_size: int,
) -> tuple[tuple[int, int], ...]:
    """Shapes of the CUDA BF16 forward's nine returned tensors, in ABI order."""

    for name, value in (
        ("num_local_tokens", num_local_tokens),
        ("macrobatch_size", macrobatch_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return (
        (macrobatch_size, HIDDEN_SIZE),
        (num_local_tokens, INTERMEDIATE_SIZE),
        (macrobatch_size, INTERMEDIATE_SIZE),
        (num_local_tokens, INTERMEDIATE_SIZE),
        (macrobatch_size, INTERMEDIATE_SIZE),
        (num_local_tokens, INTERMEDIATE_SIZE),
        (macrobatch_size, INTERMEDIATE_SIZE),
        (num_local_tokens, HIDDEN_SIZE),
        (macrobatch_size, HIDDEN_SIZE),
    )
