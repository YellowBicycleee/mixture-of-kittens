# BF16 EP8 B-sized Replay backward update

## Status and scope

This update adds an opt-in minibatch-grained backward schedule for BF16 EP8
while retaining the existing B-sized activation ring and routed-forward Replay.
Forward is unchanged, and the default deterministic
`bwd_schedule="macrobatch"` remains unchanged.

The measured path in this note does **not** retain all `C` routed rows and does
**not** use the MXFP8 full-context/no-Replay specialization. For a fixed
`macrobatch_size=B`, forward and backward use the same `minibatch_size` and
saved `x/gate/up/hidden` extent.

Enable the fine schedule with:

```python
config = MoKConfig(
    macrobatch_size=B,
    minibatch_size=minibatch_size,
    bwd_schedule="minibatch",
)
```

## What changed

1. **Minibatch-grained ring retirement.** A reused physical minibatch slot can
   advance after its non-Wgrad readers finish and Wgrad has consumed the source
   rows; it no longer waits for an entire macrobatch to retire.
2. **Compact routed Wgrad ownership.** The scheduler publishes Wgrad work only
   for real nonempty `(expert, macrobatch segment)` intersections instead of
   reserving a dense all-expert envelope for every macrobatch.
3. **Replay-aware O(1) task-owner decode.** A coarse owner table plus one prefix
   boundary correction replaces a per-task binary search over minibatches.
4. **Paired BF16 Replay Gate+Up.** Gate and Up share one routed `x` tile load,
   use independent accumulators, and keep the original two-arrival readiness
   contract. Replay SwiGLU remains present.

For the Qwen 100K fixture at `B=131072, minibatch_size=4096`, the active
compute-cluster namespace falls from 243,039 to 144,223. The difference is
84,672 removed Wgrad envelope tasks plus 14,144 removed Replay Gate/Up task
assignments; this is a structural task count, not a latency decomposition.

## Correctness and benchmark contract

- Qwen-shaped BF16, EP8, 512 experts / 64 local experts, top-k 10.
- 8 NVIDIA B300 GPUs; CUDA-Event timing; each sample uses the maximum latency
  across ranks; W2/N6 for final measurements.
- Every selected cell is checked against the exact same-cell OLD macrobatch
  schedule for output plus eight gradients.
- Final 100K/20K/8K matrices contain 24/24/23 legal macrobatch rows. In total,
  3,483 serialized output/gradient checks passed; maximum absolute error was
  0.381836 (`<=0.5`) and maximum relative error was 0.002716 (`<=0.01`).
- Every published BF16 row records `context_rows=B` and
  `full_context_or_no_replay=false`.

## Performance

The table reports independently tuned OLD/NEW BWD at the same macrobatch.
Positive gain means the minibatch schedule is faster. Configuration is
`minibatch/FWD-comm-SM/BWD-comm-SM`.

| Tokens/rank | macroB | OLD BWD | NEW BWD | NEW gain | NEW config |
|---:|---:|---:|---:|---:|---|
| 102,400 | 4,096 | 235.170731 ms | 118.650368 ms | +49.547% | 2048/32/44 |
| 102,400 | 131,072 | 98.480881 ms | 75.941711 ms | +22.887% | 16384/42/38 |
| 20,480 | 4,096 | 47.470592 ms | 24.425248 ms | +48.547% | 2048/32/44 |
| 20,480 | 131,072 | 16.593984 ms | 13.006784 ms | +21.617% | 16384/38/34 |
| 8,192 | 4,096 | 20.112208 ms | 10.233504 ms | +49.118% | 2048/28/32 |
| 8,192 | 32,768 | 8.671296 ms | 6.300432 ms | +27.342% | 8192/36/32 |

Strict same-cell measurements, which fix `B`, minibatch, forward SMs, and
backward SMs and change only `bwd_schedule`, show the same trend: NEW wins 64
of the 65 points where every rank performs Replay. All three zero-Replay
endpoints regress, because there is no later generation to overlap and the
fine-schedule overhead remains.

Measured incremental ablations at 100K and `B=131072`:

| Stage | Base BWD | Optimized BWD | Gain |
|---|---:|---:|---:|
| Initial macro-to-minibatch schedule (`mini=4096`, F36/B44) | 97.325043 ms | 83.149345 ms | +14.565% |
| Replay-aware O(1) owner decode (`mini=4096`, F36/B44) | 82.453087 ms | 80.748047 ms | +2.068% |
| Paired Replay Gate+Up (`mini=16384`, F42/B36) | 75.504684 ms | 74.547218 ms | +1.268% |

These ablations use different accepted builds/configurations and must not be
added together as one percentage.

## Limitations

- The measured claim is limited to BF16 EP8 Qwen-shaped workloads on B300.
- Routed-forward Replay is deliberately retained for every generation after
  the first saved B-sized generation.
- Split-expert BF16 routed-Wgrad partials can be accumulated in a different
  order, so the opt-in minibatch schedule is tolerance-correct but not bitwise
  deterministic.
- Near the one-generation boundary, the default macrobatch schedule can be
  faster; users should tune legal `macrobatch_size`, `minibatch_size`, and
  communication-SM counts for their workload.

Future agent releases should add a sibling directory named
`docs/updates/YYYY-MM-DD-<short-topic>/` with a concise `README.md` and link it
from the repository README.
