import torch

from mok.union_ops import union_x_fc1_k64


def test_union_x_fc1_k64_matches_fp32_partial_reference(
    context: tuple[int, int, torch.device],
) -> None:
    _, world_size, device = context
    if world_size != 8:
        return

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260828)

    def make_bf16(shape: tuple[int, ...]) -> torch.Tensor:
        host = torch.randn(shape, generator=generator, dtype=torch.float32)
        return (host * 0.0625).to(device=device, dtype=torch.bfloat16)

    union_x = make_bf16((128, 4096))
    gate = make_bf16((128, 4096))
    up = make_bf16((128, 4096))
    # Cover repeats across both CTA row halves and explicit padding.
    route_to_union = torch.arange(256, dtype=torch.int32)
    route_to_union.remainder_(128)
    route_to_union[31] = -1
    route_to_union[159] = -1
    route_to_union[200:208] = 7
    route_to_union = route_to_union.to(device=device)

    actual = union_x_fc1_k64(union_x, route_to_union, gate, up)
    torch.cuda.synchronize(device)

    gathered = torch.zeros(256, 64, dtype=torch.float32, device=device)
    valid = route_to_union >= 0
    gathered[valid] = union_x[route_to_union[valid].long(), :64].float()
    reference_gate = gathered @ gate[:, :64].float().transpose(0, 1)
    reference_up = gathered @ up[:, :64].float().transpose(0, 1)
    reference = torch.cat((reference_gate, reference_up), dim=1).contiguous()
    torch.cuda.synchronize(device)

    assert actual.dtype == torch.float32
    assert tuple(actual.shape) == (256, 256)
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)
    assert torch.count_nonzero(actual[31]).item() == 0
    assert torch.count_nonzero(actual[159]).item() == 0
