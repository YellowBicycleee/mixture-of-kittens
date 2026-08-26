"""Opt-in CuTe DSL implementations for MoK.

This package is imported lazily only for ``MoKConfig(fwd_backend="cutedsl")``;
the CUDA C++ backend remains the default implementation.

The BF16 pipeline-v2 stage fuses shared and routed Gate+Up+SwiGLU through
QuACK 0.6.4. Shared work always materializes Gate/Up; routed work does so only
for macro zero to preserve the existing backward replay ABI. The CUDA C++
backend and all MXFP8 behavior remain untouched.
"""
