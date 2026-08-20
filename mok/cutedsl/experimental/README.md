# CuTe DSL Stage 0

This directory is a narrow feasibility spike, not a second MoE implementation.
It leaves the CUDA C++ backend and public `mok` API unchanged.

The kernel is pinned to `nvidia-cutlass-dsl==4.6.2` and fixed to the Qwen
BF16/EP8 boundary (`H=4096`, `top-k=10`). It proves these four mechanisms in
one Blackwell cluster launch:

1. Consume all eight `MoKWorkspace.x_buffer_ptrs` without reallocating the
   symmetric workspace.
2. Select the source peer from `schedule_peer_rank` at runtime and TMA-load the
   complete row named by `schedule_peer_token_idx`.
3. Publish/wait on paired-CTA counters with release/acquire semantics at both
   system and GPU scope.
4. Run communication and paired-compute logical roles from the same 2-CTA CLC
   dynamic work queue.

The implementation follows the v4.6.2 CUTLASS examples at Git revision
`6c65a175668952f09bcbf66cb97a8de1b734b4a0`, specifically
`all_reduce_tma.py` and `fp16_gemm_3_1.py`.

CPU contract:

```bash
python -m unittest tests/test_cutedsl_stage0_contract.py
```

B300 smoke, after building the existing MoK extension and installing the
optional dependency:

```bash
pip install -e '.[cutedsl]'
torchrun --standalone --nproc-per-node=8 \
  benchmarks/cutedsl/run_stage0_b300.py
```

The B300 runner makes deterministic BF16 rows on every rank and compares every
copied element exactly. It separately checks the CLC role log, system-scope
counter, GPU-scope counter, and task-completion log.

Not implemented in Stage 0: GEMM, SwiGLU, combine, production minibatching,
macrobatch replay, backward, MXFP8, performance measurement, or public backend
selection. These remain `N/A` until the primitive smoke passes on B300.
