// Static resource contract for the BF16 EP8 warp-specialized forward path.
//
// Like the other megakernel implementation headers, this file is included
// inside dispatch_mlp_swiglu_combiner after types.cuh.  It deliberately does
// not contain a scheduler or a device main loop.  The future implementation
// must explicitly select this contract only for NUM_DEVICES == 8 and
// !USE_MXFP8.

// One warpgroup pulls remote activation chunks, one warpgroup owns the two
// TMA loaders plus the tcgen05 issuer and scheduler, and one warpgroup owns
// the SwiGLU/context-store epilogue.
static constexpr int EP8_BF16_WS_COMM_WARPS = 4;
static constexpr int EP8_BF16_WS_MAINLOOP_WARPS = 4;
static constexpr int EP8_BF16_WS_EPI_WARPS = 4;
static constexpr int EP8_BF16_WS_NUM_WARPGROUPS = 3;
static constexpr int EP8_BF16_WS_NUM_WARPS = 12;
static constexpr int EP8_BF16_WS_NUM_THREADS = 384;

static constexpr int EP8_BF16_WS_COMM_WARP_BEGIN = 0;
static constexpr int EP8_BF16_WS_COMM_WARP_END = 4;
static constexpr int EP8_BF16_WS_TMA_A_WARP = 4;
static constexpr int EP8_BF16_WS_TMA_B_WARP = 5;
static constexpr int EP8_BF16_WS_MMA_WARP = 6;
static constexpr int EP8_BF16_WS_SCHEDULER_WARP = 7;
static constexpr int EP8_BF16_WS_MAINLOOP_WARP_BEGIN = 4;
static constexpr int EP8_BF16_WS_MAINLOOP_WARP_END = 8;
static constexpr int EP8_BF16_WS_EPI_WARP_BEGIN = 8;
static constexpr int EP8_BF16_WS_EPI_WARP_END = 12;

static_assert(WARPGROUP_WARPS == 4);
static_assert(WARP_THREADS == 32);
static_assert(EP8_BF16_WS_COMM_WARPS == WARPGROUP_WARPS);
static_assert(EP8_BF16_WS_MAINLOOP_WARPS == WARPGROUP_WARPS);
static_assert(EP8_BF16_WS_EPI_WARPS == WARPGROUP_WARPS);
static_assert(
    EP8_BF16_WS_NUM_WARPGROUPS * WARPGROUP_WARPS
    == EP8_BF16_WS_NUM_WARPS);
static_assert(
    EP8_BF16_WS_NUM_WARPS * WARP_THREADS
    == EP8_BF16_WS_NUM_THREADS);
static_assert(
    EP8_BF16_WS_COMM_WARP_END - EP8_BF16_WS_COMM_WARP_BEGIN
    == EP8_BF16_WS_COMM_WARPS);
static_assert(EP8_BF16_WS_COMM_WARP_END == EP8_BF16_WS_TMA_A_WARP);
static_assert(EP8_BF16_WS_TMA_A_WARP + 1 == EP8_BF16_WS_TMA_B_WARP);
static_assert(EP8_BF16_WS_TMA_B_WARP + 1 == EP8_BF16_WS_MMA_WARP);
static_assert(
    EP8_BF16_WS_MMA_WARP + 1 == EP8_BF16_WS_SCHEDULER_WARP);
static_assert(
    EP8_BF16_WS_SCHEDULER_WARP + 1
    == EP8_BF16_WS_MAINLOOP_WARP_END);
static_assert(
    EP8_BF16_WS_MAINLOOP_WARP_END
        - EP8_BF16_WS_MAINLOOP_WARP_BEGIN
    == EP8_BF16_WS_MAINLOOP_WARPS);
static_assert(
    EP8_BF16_WS_MAINLOOP_WARP_END == EP8_BF16_WS_EPI_WARP_BEGIN);
static_assert(
    EP8_BF16_WS_EPI_WARP_END - EP8_BF16_WS_EPI_WARP_BEGIN
    == EP8_BF16_WS_EPI_WARPS);
static_assert(EP8_BF16_WS_EPI_WARP_END == EP8_BF16_WS_NUM_WARPS);

// The BF16 input tiles are 128 x 64 x 2 bytes per CTA.  Three A/B stages
// replace the legacy four-stage ring and release 32 KiB for communication.
static constexpr int EP8_BF16_WS_LOAD_STAGES = 3;
static constexpr int EP8_BF16_WS_A_STAGE_BYTES = 16384;
static constexpr int EP8_BF16_WS_B_STAGE_BYTES = 16384;
static constexpr int EP8_BF16_WS_A_RING_BYTES = 49152;
static constexpr int EP8_BF16_WS_B_RING_BYTES = 49152;

// Each communication warp exclusively owns one bounded pull buffer.  A
// buffer may be recycled only after comm_empty[warp] advances to the matching
// phase; the consumer observes it only after comm_full[warp] advances.
static constexpr int EP8_BF16_WS_COMM_BUFFER_BYTES = 4096;
static constexpr int EP8_BF16_WS_COMM_BYTES = 16384;

// Gate and Up stay live until their training-context TMA stores finish.
// Hidden is a third, disjoint tile: it must not alias either context source
// even after tcgen05 starts filling the next accumulator stage.
static constexpr int EP8_BF16_WS_GATE_BYTES = 32768;
static constexpr int EP8_BF16_WS_UP_BYTES = 32768;
static constexpr int EP8_BF16_WS_HIDDEN_BYTES = 32768;
static constexpr int EP8_BF16_WS_CONTEXT_SCRATCH_BYTES = 98304;

static constexpr int EP8_BF16_WS_SMEM_ALIGNMENT_BYTES = 1024;
static constexpr int EP8_BF16_WS_SMEM_ALIGNMENT_SLACK_BYTES = 1023;
static constexpr int EP8_BF16_WS_SMEM_CAPACITY_BYTES = 231424;
static constexpr int EP8_BF16_WS_LIVE_PAYLOAD_BYTES = 212992;
static constexpr int EP8_BF16_WS_SYNC_BARRIERS = 18;
static constexpr int EP8_BF16_WS_SYNC_BYTES = 144;
static constexpr int EP8_BF16_WS_STORAGE_ALIGN_BYTES = 128;
static constexpr int EP8_BF16_WS_SMEM_STORAGE_BYTES = 213248;

// tcgen05 and the epilogue alternate between two complete 256-column
// accumulators.  The epilogue must release a stage before the MMA warp reuses
// that exact range; no MXFP8 scale allocation is present in this BF16 layout.
static constexpr int EP8_BF16_WS_TMEM_STAGES = 2;
static constexpr int EP8_BF16_WS_TMEM_COLS_PER_STAGE = 256;
static constexpr int EP8_BF16_WS_TMEM_STAGE_0_OFFSET = 0;
static constexpr int EP8_BF16_WS_TMEM_STAGE_1_OFFSET = 256;
static constexpr int EP8_BF16_WS_TMEM_CAPACITY_COLS = 512;
static constexpr int EP8_BF16_WS_TMEM_STAGE_OFFSETS
    [EP8_BF16_WS_TMEM_STAGES] = {
        EP8_BF16_WS_TMEM_STAGE_0_OFFSET,
        EP8_BF16_WS_TMEM_STAGE_1_OFFSET,
    };

static_assert(EP8_BF16_WS_LOAD_STAGES == 3);
static_assert(sizeof(mlp_bf16_tile) == EP8_BF16_WS_A_STAGE_BYTES);
static_assert(sizeof(mlp_bf16_tile) == EP8_BF16_WS_B_STAGE_BYTES);
static_assert(
    EP8_BF16_WS_A_RING_BYTES
    == EP8_BF16_WS_LOAD_STAGES * EP8_BF16_WS_A_STAGE_BYTES);
static_assert(
    EP8_BF16_WS_B_RING_BYTES
    == EP8_BF16_WS_LOAD_STAGES * EP8_BF16_WS_B_STAGE_BYTES);
static_assert(EP8_BF16_WS_COMM_WARPS == 4);
static_assert(EP8_BF16_WS_COMM_BUFFER_BYTES <= 4 * 1024);
static_assert(
    EP8_BF16_WS_COMM_BYTES
    == EP8_BF16_WS_COMM_WARPS * EP8_BF16_WS_COMM_BUFFER_BYTES);
static_assert(sizeof(quant_bf16_tile) == EP8_BF16_WS_GATE_BYTES);
static_assert(sizeof(quant_bf16_tile) == EP8_BF16_WS_UP_BYTES);
static_assert(sizeof(quant_bf16_tile) == EP8_BF16_WS_HIDDEN_BYTES);
static_assert(
    EP8_BF16_WS_CONTEXT_SCRATCH_BYTES
    == EP8_BF16_WS_GATE_BYTES + EP8_BF16_WS_UP_BYTES
        + EP8_BF16_WS_HIDDEN_BYTES);
static_assert(
    EP8_BF16_WS_LIVE_PAYLOAD_BYTES
    == EP8_BF16_WS_A_RING_BYTES + EP8_BF16_WS_B_RING_BYTES
        + EP8_BF16_WS_COMM_BYTES + EP8_BF16_WS_CONTEXT_SCRATCH_BYTES);
static_assert(
    EP8_BF16_WS_SMEM_ALIGNMENT_SLACK_BYTES
    == EP8_BF16_WS_SMEM_ALIGNMENT_BYTES - 1);
static_assert(
    config::DYNAMIC_SHARED_MEMORY == EP8_BF16_WS_SMEM_CAPACITY_BYTES);
static_assert(EP8_BF16_WS_TMEM_STAGES == 2);
static_assert(EP8_BF16_WS_TMEM_COLS_PER_STAGE == config::MLP_Nb);
static_assert(EP8_BF16_WS_TMEM_STAGE_0_OFFSET == 0);
static_assert(
    EP8_BF16_WS_TMEM_STAGE_0_OFFSET + EP8_BF16_WS_TMEM_COLS_PER_STAGE
    <= EP8_BF16_WS_TMEM_STAGE_1_OFFSET);
static_assert(
    EP8_BF16_WS_TMEM_STAGE_1_OFFSET + EP8_BF16_WS_TMEM_COLS_PER_STAGE
    <= EP8_BF16_WS_TMEM_CAPACITY_COLS);
static_assert(
    EP8_BF16_WS_TMEM_STAGES * EP8_BF16_WS_TMEM_COLS_PER_STAGE
    == EP8_BF16_WS_TMEM_CAPACITY_COLS);

// Full/empty pairs make ownership explicit without prescribing the algorithm:
//   load: TMA-A/TMA-B -> MMA, one phase per three-stage A/B slot;
//   comm: COMM warp -> scheduler/mainloop, one queue slot per COMM warp;
//   tmem: MMA -> epilogue, one phase per accumulator stage.
// The phase bits themselves remain role-local monotonic state.
struct ep8_bf16_ws_smem_storage {
    mlp_bf16_tile a_smem[EP8_BF16_WS_LOAD_STAGES];
    mlp_bf16_tile b_smem[EP8_BF16_WS_LOAD_STAGES];
    uint8_t comm_smem[EP8_BF16_WS_COMM_WARPS]
                     [EP8_BF16_WS_COMM_BUFFER_BYTES];
    quant_bf16_tile gate_smem;
    quant_bf16_tile up_smem;
    quant_bf16_tile hidden_smem;

    semaphore load_full[EP8_BF16_WS_LOAD_STAGES];
    semaphore load_empty[EP8_BF16_WS_LOAD_STAGES];
    semaphore comm_full[EP8_BF16_WS_COMM_WARPS];
    semaphore comm_empty[EP8_BF16_WS_COMM_WARPS];
    semaphore tmem_full[EP8_BF16_WS_TMEM_STAGES];
    semaphore tmem_empty[EP8_BF16_WS_TMEM_STAGES];
};

static constexpr uint64_t EP8_BF16_WS_A_OFFSET =
    __builtin_offsetof(ep8_bf16_ws_smem_storage, a_smem);
static constexpr uint64_t EP8_BF16_WS_B_OFFSET =
    __builtin_offsetof(ep8_bf16_ws_smem_storage, b_smem);
static constexpr uint64_t EP8_BF16_WS_COMM_OFFSET =
    __builtin_offsetof(ep8_bf16_ws_smem_storage, comm_smem);
static constexpr uint64_t EP8_BF16_WS_GATE_OFFSET =
    __builtin_offsetof(ep8_bf16_ws_smem_storage, gate_smem);
static constexpr uint64_t EP8_BF16_WS_UP_OFFSET =
    __builtin_offsetof(ep8_bf16_ws_smem_storage, up_smem);
static constexpr uint64_t EP8_BF16_WS_HIDDEN_OFFSET =
    __builtin_offsetof(ep8_bf16_ws_smem_storage, hidden_smem);
static constexpr uint64_t EP8_BF16_WS_BARRIER_OFFSET =
    __builtin_offsetof(ep8_bf16_ws_smem_storage, load_full);

// These intervals are all simultaneously live.  Keep every comparison
// explicit so that adding a member or changing a tile cannot silently create
// the tempting but incorrect Gate/Up/Hidden overlay.
static_assert(EP8_BF16_WS_A_OFFSET == 0);
static_assert(
    EP8_BF16_WS_A_OFFSET + EP8_BF16_WS_A_RING_BYTES
    <= EP8_BF16_WS_B_OFFSET);
static_assert(
    EP8_BF16_WS_B_OFFSET + EP8_BF16_WS_B_RING_BYTES
    <= EP8_BF16_WS_COMM_OFFSET);
static_assert(
    EP8_BF16_WS_COMM_OFFSET + EP8_BF16_WS_COMM_BYTES
    <= EP8_BF16_WS_GATE_OFFSET);
static_assert(
    EP8_BF16_WS_GATE_OFFSET + EP8_BF16_WS_GATE_BYTES
    <= EP8_BF16_WS_UP_OFFSET);
static_assert(
    EP8_BF16_WS_UP_OFFSET + EP8_BF16_WS_UP_BYTES
    <= EP8_BF16_WS_HIDDEN_OFFSET);
static_assert(
    EP8_BF16_WS_HIDDEN_OFFSET + EP8_BF16_WS_HIDDEN_BYTES
    <= EP8_BF16_WS_BARRIER_OFFSET);
static_assert(EP8_BF16_WS_BARRIER_OFFSET == EP8_BF16_WS_LIVE_PAYLOAD_BYTES);
static_assert(sizeof(semaphore) == 8);
static_assert(
    EP8_BF16_WS_SYNC_BARRIERS
    == 2 * EP8_BF16_WS_LOAD_STAGES + 2 * EP8_BF16_WS_COMM_WARPS
        + 2 * EP8_BF16_WS_TMEM_STAGES);
static_assert(
    EP8_BF16_WS_SYNC_BYTES
    == EP8_BF16_WS_SYNC_BARRIERS * sizeof(semaphore));
static_assert(
    EP8_BF16_WS_SMEM_STORAGE_BYTES
    == ((EP8_BF16_WS_LIVE_PAYLOAD_BYTES + EP8_BF16_WS_SYNC_BYTES
         + EP8_BF16_WS_STORAGE_ALIGN_BYTES - 1)
        / EP8_BF16_WS_STORAGE_ALIGN_BYTES)
           * EP8_BF16_WS_STORAGE_ALIGN_BYTES);
static_assert(
    sizeof(ep8_bf16_ws_smem_storage) == EP8_BF16_WS_SMEM_STORAGE_BYTES);
static_assert(
    sizeof(ep8_bf16_ws_smem_storage)
    <= EP8_BF16_WS_SMEM_CAPACITY_BYTES);
static_assert(
    sizeof(ep8_bf16_ws_smem_storage)
        + EP8_BF16_WS_SMEM_ALIGNMENT_SLACK_BYTES
    <= config::DYNAMIC_SHARED_MEMORY);
