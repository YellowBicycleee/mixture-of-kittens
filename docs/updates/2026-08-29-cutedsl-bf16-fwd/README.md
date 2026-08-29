# Optional CuTe DSL BF16 forward

This update adds an explicit `MoKConfig(fwd_backend="cutedsl")` forward
specialization without replacing the original CUDA implementation. The default
remains `fwd_backend="cuda"`; unsupported CuTe configurations fail closed, and
backward continues to use the existing CUDA BF16 kernel.

## Supported configuration

- NVIDIA SM103, validated on one node with eight B300 GPUs
- EP8, 64 local experts, H=4096, I=1024, top-k=10
- BF16 routed weights and unclamped SwiGLU
- `macrobatch_size=32768`, `minibatch_size=4096`
- `fwd_num_comm_sms=40`
- `nvidia-cutlass-dsl[cu13]==4.6.2` and
  `quack-kernels[cu13]==0.6.4`

Install the optional dependencies with:

```bash
pip install ".[cutedsl]" --no-build-isolation
```

Other shapes, precisions, architectures, and clamped SwiGLU are not silently
redirected to CUDA.

## Validation result

The release gate used 102400 tokens per rank with the fixed
`1234 + EP rank` input seeds. Each leg ran forward immediately followed by the
default deterministic macrobatch backward before cloning its results. Three
comparisons covered output, seven public forward-context tensors, and eight
gradients: CUDA1 versus CUDA2, CUDA1 versus CuTe, and CUDA2 versus CuTe. All 48
records were finite and bitwise exact (`max_abs=0`, `relative_l1=0`). The
acceptance contract was `max_abs <= 0.5` and `relative_l1 <= 0.01`.

CUDA Events bracketed `mok.functional.forward` only. Schedule construction,
input generation, and the single CuTe compilation were outside the events.
Each ABBA block used 10 warmups and 30 samples; every sample was reduced to the
maximum same-index latency across EP8 before summarization.

| ABBA block | Backend | Median rank-max (ms) | CV |
|---:|:---|---:|---:|
| 0 | CUDA | 35.633520 | 0.594% |
| 1 | CuTe DSL | 34.990688 | 1.285% |
| 2 | CuTe DSL | 35.009071 | 0.740% |
| 3 | CUDA | 35.677649 | 0.558% |

The two-block summaries were 35.652945 ms for CUDA and 34.994831 ms for CuTe
DSL: CuTe was 1.018806x as fast, or 1.846% lower latency. CUDA bookend drift
was 0.123840%. This is a measured fixed-shape B300 result, not a claim about
other MoE shapes or GPUs.

## Why the original CuTe version was slow

- **Source:** every routed Gate/Up or Down task repeatedly scanned all 64
  expert counts to recover its expert and row tile, and that work appeared in
  multiple persistent roles.
- **Measured:** replacing only that decoder with the prefix/fixed-six path
  reduced the fixed-shape CuTe latency from 54.119167 ms to 35.385664 ms;
  pipeline-depth changes were much smaller.
- **Inference:** for this tested EP8/E64/BF16/SM103 shape, repeated decoding was
  the dominant observed gap. The A/B result argues against BF16 MMA throughput
  or an inherent CuTe DSL ceiling as the explanation for that gap, but no
  per-role hardware profile was collected to attribute individual cycles.

The accepted change builds a 65-entry expert row-block prefix once per CTA in
existing shared-memory padding, then uses a fixed six-comparison upper bound
for each routed task. It does not change the route map, task order, readiness
counters, communication, or GEMM math.

The checkpoint timings below came from the isolated direct single-launch
tuning harness. They explain the search decision; the public API performance
claim is the r59 result above.

| DFS checkpoint | CuTe median (ms) | Outcome |
|:---|---:|:---|
| Initial O3 | 56.737263 | Baseline; about 21 ms behind CUDA |
| A/B pipeline 4 to 6 | 54.521215 | Kept; 3.91% improvement |
| Accumulator pipeline 1 to 2 | 54.119167 | Kept; 0.74% improvement |
| Fixed-six prefix decode | 35.385664 | Kept; 34.62% improvement from the prior checkpoint |
| DSL 4.6.2 / QuACK 0.6.4 migration | 35.066065 | Kept after parity and ABBA |

Compact capacity and O2 produced only marginal changes; full-N produced no
gain, the rmem2 variant emitted identical PTX/CUBIN, C1 was slower, and the
fixed-K64 FC1 variant regressed. Those branches were rolled back rather than
combined. QuACK 0.6.4 additionally required the pipeline-owned accumulator
release election (`elect_one_release=True`,
`syncwarp_before_release=False`).

## Provenance and evidence boundary

- Source HEAD: `cf2bc8067797d8fbbf21f9f3792dcd314cf1fe6d`
- Frozen source archive SHA256:
  `e21c3f8415884d2dfdebb4138da13584125eb99389c2909559ba1eba670939ea`
- Validation runner SHA256:
  `fc9806cb4eb891d2373bea5303fd062077be15e4eecb86336fdfa4d8b4d497ce`
- Result JSON SHA256:
  `4c15c37663a65fa124fe8445440370fc2dc8b0797ec55e2d818e3b027ef82b38`
- Queue result / SUCCESS / stdout SHA256:
  `a689236b2f383195b6020239de47d264e20d7284838b6a8361d205cad7607cb2`,
  `a745fc23955d706561cd8ea9aba2fd70809eaaa471fbb4d28288680c0e18ac39`,
  `327e10f05d61b7e164f3cb8af7801c54f77de9ed0d0bcc4ab345152860641253`
- Runtime: Python 3.12.3, PyTorch 2.12.0a0+5aff3928d8.nv26.05,
  CUDA 13.2, CUTLASS DSL 4.6.2, QuACK 0.6.4, TVM-FFI enabled

The subsequent promotion cleanup removes an unused host read from schedule
construction and corrects the repository benchmark and documentation. It does
not change the persistent device kernel or the measured `functional.forward`
path. The hashes above identify the exact source and runner used for the GPU
claim; future kernel changes require new correctness and timing evidence.
