# Mixture-of-Kittens (MoK)

Mixture-of-Kittens (MoK) is a mixture-of-experts (MoE) training megakernel built from first principles for NVL72s. Its default execution path, including macrobatch-ordered backward weight-gradient accumulation, is fully deterministic. The opt-in minibatch backward schedule described below is not bitwise deterministic. MoK fuses all MoE computation and communication into a single kernel, overlapping compute and inter-GPU networking at configurable granularity, and fully eliminates CPU-GPU synchronization. It supports BF16 and MXFP8, covers both forward and backward passes, and powers production training of Composer at Cursor.

For a deep dive, read our [blog post](https://cursor.com/blog/mixture-of-kittens).

## Performance

We evaluated MoK at two levels: standalone MoE layers, with all benchmark code available in [`benchmarks`](./benchmarks), and end-to-end training on our internal production stack across multiple NVL72 racks. The following figure summarizes the standalone MoE layer benchmarks:

![MoE benchmark results](./figures/moe_benchmarks.png)

Compared with the fastest baseline, MoK is up to 2.37x faster for the MXFP8 forward, 1.78x for the MXFP8 backward, 1.92x for the BF16 forward, and 1.58x for the BF16 backward.

See our [blog post](https://cursor.com/blog/mixture-of-kittens) for the methodology and full results.

## Setup

### Requirements

- NVIDIA Blackwell SM100 or SM103 GPUs (e.g., GB200 NVL72 or GB300 NVL72)
- Python 3.12 or later
- PyTorch 2.10 or later
- CUDA toolkit 13.0 or later

### PyTorch and CUDA

The installed PyTorch build must target CUDA 13.0+, and its CUDA version must match the major and minor version of the system CUDA toolkit. MoK is built and tested against CUDA 13.0, but later versions should work as well.

For example, to install PyTorch 2.10 built against CUDA 13.0:

```bash
python -m pip install "torch==2.10.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```

### Installation

By default, MoK builds for SM103. Install from the repository root with either:

```bash
pip install . --no-build-isolation
```

or:

```bash
python setup.py install
```

To build for SM100 instead, set the `MOK_ARCH` environment variable:

```bash
MOK_ARCH=SM100 pip install . --no-build-isolation
```

To verify the installation:

```bash
python -c "import mok; print(mok.__version__)"
```

### Development

For development, install MoK in editable mode once, then use `make` for fast rebuilds:

```bash
pip install -e . --no-build-isolation
make
```

#### Unit tests

Launch multi-GPU unit tests through `torchrun` and `pytest`:

```bash
torchrun --standalone --nproc-per-node=<num-gpus> -m pytest -s <test-path>
```

## Getting Started

MoK provides two layers of abstraction:

- **Ops layer** (`mok/ops.py`): the low-level API that lets you call our CUDA kernel implementations directly with minimal overhead. With this, you fully manage your own data (e.g., symmetric tensors) and coordinate the kernel calls yourself to implement the MoE layer's computation and communication.
- **Functional layer** (`mok/functional.py`): the higher-level API that handles more for you, such as maintaining scratch memory and coordinating kernel launches. It consists of workspace creation functions, a schedule function, and forward/backward functions.

The functional layer is our choice for production training, so we recommend it unless you have specific needs. We explain only the functional layer from here.

With the functional API, MoK is simple to use: call `schedule(...)` once to build the dispatch/combine schedule, then pass that same schedule to `forward(...)` and `backward(...)`. Before doing so, however, you need to define and manage a **config** and a **workspace**.

### Config

MoK exposes several settings that affect computation/communication overlap.
Optimal values depend on the model, routing distribution, EP size, precision,
GPU, and token count, so tune them together on the actual workload. Do not use
a fixed minibatch or communication-SM range as a universal recommendation.

- `fwd_num_comm_sms` (default: `40`): positive even communication-SM count used
  by forward. Tune it against forward latency and leave at least one compute
  SM.
- `bwd_num_comm_sms` (default: `28`): positive even communication-SM count used
  by backward. Tune it independently from the forward count and leave at least
  one compute SM.
- `minibatch_size` (default: `4096`): computation/communication overlap
  granularity. It must be positive, divisible by 256, and divide the
  `macrobatch_size` used by that call. Sweep legal divisors instead of assuming
  that finer is always faster.
- `macrobatch_size` (default: `131072`): physical routed ring capacity. Larger
  rings reduce reuse and legacy routed-forward replay, while smaller rings may
  improve overlap and reduce backward-gradient buffer capacity. Sweep it
  jointly with `minibatch_size`, communication-SM counts, and `bwd_schedule`,
  and include the full memory footprint in the decision.
- `bwd_schedule` (default: `"macrobatch"`): the default preserves deterministic
  weight-gradient accumulation order. `"minibatch"` enables fine ring-slot
  retirement and can overlap dependent work sooner, but split-expert
  weight-gradient partials may be added in a different order, so it is not
  bitwise deterministic. All ranks in an expert-parallel group must use the
  same setting.
- `schedule_capacity_multiplier` (default: `0.5`): positive finite capacity
  factor for the routed schedule. Size it for the worst expected expert
  imbalance; lowering it does not materially reduce the main activation or
  gradient-buffer footprint.
- `all_gather_top_experts_chunk_bytes` (default: `2048`): positive multiple of
  16 that divides one rank's route-buffer bytes and fits the device's dynamic
  shared-memory limit. Keep the default unless the routing all-gather is being
  tuned explicitly.

The same `MoKConfig` can be used for ordinary forward and backward calls. The
experimental EP8/MXFP8 dual-context path is the exception: forward may use
`macrobatch_size=C` to retain every padded routed row, while backward uses a
separately tuned `macrobatch_size=B<C` with `bwd_schedule="minibatch"`. `C`
must cover the maximum padded routed-row count across ranks. At `B=C`, select
the tuned legacy macrobatch path rather than forcing the fine specialization.
Every EP rank must use the same selected forward configuration and the same
selected backward configuration.

See [EP8 MXFP8 backward: full-context forward and a fine backward
ring](docs/bwd-ep8-minibatch-pipeline.md) for the execution contract, memory
cost, complete per-macrobatch measurements, bandwidth definition, and current
limitations.

See the [BF16 EP8 B-sized Replay backward update](docs/updates/2026-08-25-bf16-ep8-replay-bwd/README.md)
for the minibatch scheduling changes, correctness contract, measured B300
results, and current limitations.

Set these values when constructing `MoKConfig`, and pass the selected
configuration for that call to the functional layer.

### Workspace

MoK relies on PyTorch symmetric memory to allocate and manage inter-GPU symmetric buffers, or identically sized memory allocations across many GPUs (i.e., all GPUs in an EP group). These buffers serve as the source/destination of token dispatch/combine, along with a few other purposes. We call the entire collection of symmetric memory, along with the other metadata and scratchpads MoK needs, the **workspace**. We provide a data structure and functions for creating and destroying workspaces.

Because symmetric buffers are expensive to allocate, we recommend creating a single workspace per model and reusing it across layers. For this, we provide the `get_workspace(...)` function, which automatically caches workspaces with identical properties (EP group, device, model shapes, etc.). If you prefer to manage a workspace's lifetime yourself, the `create_workspace(...)` function does not cache.

### MXFP8

To run MoK in MXFP8 mode, pass the activations as-is in BF16 while prequantizing the weights to MXFP8. We *could* quantize the weights inside our kernels, but prequantizing leaves better opportunities for things like FSDP, so we keep it separate and provide the `mxfp8_quantize(...)` function at the ops layer so you can prequantize the weights yourself.

### Example (MXFP8 forward and backward using the functional layer)

The following is a canonical example of implementing MoE forward and backward with MoK in MXFP8 mode:

```python
import torch.distributed as dist

from mok import functional, ops

# Inputs:
#   num_local_tokens:    int
#   hidden_size:         int
#   intermediate_size:   int
#   topk:                int
#   num_local_experts:   int
#   x:                   torch.bfloat16 [num_local_tokens, hidden_size]
#   topk_experts:        torch.int64    [num_local_tokens, topk]
#   router_weights:      torch.float32  [num_local_tokens, topk]
#   w_shared_gate:       torch.bfloat16 [intermediate_size, hidden_size]
#   w_shared_up:         torch.bfloat16 [intermediate_size, hidden_size]
#   w_shared_down:       torch.bfloat16 [hidden_size, intermediate_size]
#   w_routed_gate:       torch.bfloat16 [num_local_experts, intermediate_size, hidden_size]
#   w_routed_up:         torch.bfloat16 [num_local_experts, intermediate_size, hidden_size]
#   w_routed_down:       torch.bfloat16 [num_local_experts, hidden_size, intermediate_size]
#   d_output:            torch.bfloat16 [num_local_tokens, hidden_size]

config = functional.MoKConfig() # tune for your workload
workspace = functional.get_workspace(
    config,
    dist.group.WORLD, # replace with your expert-parallel process group
    device=x.device,
    num_local_tokens=num_local_tokens,
    hidden_size=hidden_size,
    topk=topk,
)


########################
# Weight quantization
########################

(
    w_routed_gate_fp8,
    w_routed_gate_sc,
    w_routed_gate_t_fp8,
    w_routed_gate_t_sc,
) = ops.mxfp8_quantize(w_routed_gate, True, True)
(
    w_routed_up_fp8,
    w_routed_up_sc,
    w_routed_up_t_fp8,
    w_routed_up_t_sc,
) = ops.mxfp8_quantize(w_routed_up, True, True)
(
    w_routed_down_fp8,
    w_routed_down_sc,
    w_routed_down_t_fp8,
    w_routed_down_t_sc,
) = ops.mxfp8_quantize(w_routed_down, True, True)


########################
# Forward
########################

schedule = functional.build_schedule(
    workspace,
    config,
    topk_experts,
    num_local_experts=num_local_experts,
)
output, forward_context = functional.forward(
    config,
    workspace,
    schedule,
    x,
    router_weights,
    w_shared_gate,
    w_shared_up,
    w_shared_down,
    (w_routed_gate_fp8, w_routed_gate_sc),
    (w_routed_up_fp8, w_routed_up_sc),
    (w_routed_down_fp8, w_routed_down_sc),
)

# Save `schedule` and `forward_context` for the backward


########################
# Backward
########################

(
    d_x,
    d_router_weights,
    d_w_routed_gate,
    d_w_routed_up,
    d_w_routed_down,
    d_w_shared_gate,
    d_w_shared_up,
    d_w_shared_down,
) = functional.backward(
    config,
    workspace,
    schedule,
    forward_context,
    d_output,
    x,
    router_weights,
    w_shared_gate,
    w_shared_up,
    w_shared_down,
    (w_routed_gate_fp8, w_routed_gate_sc, w_routed_gate_t_fp8, w_routed_gate_t_sc),
    (w_routed_up_fp8, w_routed_up_sc, w_routed_up_t_fp8, w_routed_up_t_sc),
    (w_routed_down_t_fp8, w_routed_down_t_sc),
)
```

### Manual activation checkpointing

The example above saves the first macrobatch's forward intermediates for the backward pass. While we recommend this pattern, you can avoid storing them by discarding the returned context and rebuilding it immediately before backward with `functional.recompute_forward_context`.

```python
# Forward pass
output, discarded_forward_context = functional.forward(...)  # same call as above
del discarded_forward_context

# Backward pass
forward_context = functional.recompute_forward_context(
    config,
    workspace,
    schedule,
    x,
    w_shared_gate,
    w_shared_up,
    (w_routed_gate_fp8, w_routed_gate_sc),
    (w_routed_up_fp8, w_routed_up_sc),
)
gradients = functional.backward(...)  # same backward call as above
```

## Contributing

Contributions are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

Mixture-of-Kittens is released under the Apache 2.0 License. See [`LICENSE`](LICENSE).

## Citation

If you use this work, please cite:

```
Stuart H. Sul, Nash Brown, Henry Wildermuth, William Lin, and Federico Cassano. "Mixture-of-Kittens: MoE Megakernel for NVL72s." Cursor Research, Aug 2026. https://github.com/cursor/mixture-of-kittens
```

Or in BibTeX:

```bibtex
@misc{sul2026mok,
  title        = {Mixture-of-Kittens: {MoE} Megakernel for {NVL72s}},
  author       = {Stuart H. Sul and Nash Brown and Henry Wildermuth and William Lin and Federico Cassano},
  organization = {Cursor Research},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/cursor/mixture-of-kittens}},
}
```
