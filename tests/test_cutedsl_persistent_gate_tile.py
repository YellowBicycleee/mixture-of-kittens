from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CUTEDSL_ROOT = _REPO_ROOT / "mok" / "cutedsl"


def _load_persistent_forward():
    package_name = "mok_cutedsl_gate_tile_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_CUTEDSL_ROOT)]
    sys.modules[package_name] = package

    loaded = None
    for name in (
        "_tma_1d",
        "forward_contract",
        "persistent_forward_contract",
        "persistent_forward",
    ):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{name}",
            _CUTEDSL_ROOT / f"{name}.py",
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    assert loaded is not None
    return loaded


def test_shared_gate_tile_bf16_cold_jit(tmp_path, monkeypatch) -> None:
    assert torch.cuda.is_available(), "Shared Gate tile test requires CUDA"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device) == (10, 3), (
        "Shared Gate tile test requires B300/SM103"
    )

    cache_dir = tmp_path / "cute-dsl-cache"
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(cache_dir))
    persistent_forward = _load_persistent_forward()

    torch.cuda.manual_seed(20260823)
    A = torch.empty(
        (256, 64),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.25, 0.25)
    W_nk = torch.empty(
        (256, 64),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.25, 0.25)
    ref = (A.float() @ W_nk.float().T).to(torch.bfloat16)
    out = torch.full(
        (256, 256),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    ready = torch.zeros((1,), dtype=torch.int32, device=device)

    persistent_forward._run_shared_gate_tile_bf16(
        A,
        W_nk,
        out,
        ready,
    )
    torch.cuda.synchronize(device)

    assert torch.isfinite(out).all().item(), "collective left non-finite output"
    abs_error = (out.float() - ref.float()).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    print(
        f"max_abs_error={max_abs_error:.8f} "
        f"mean_abs_error={mean_abs_error:.8f}"
    )
    torch.testing.assert_close(
        out,
        ref,
        atol=0.02,
        rtol=0.02,
    )
    assert ready.item() == 2
