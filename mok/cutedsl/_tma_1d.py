"""Descriptor-free 1D TMA helpers for runtime-selected peer addresses.

MoK already owns symmetric-memory addresses for every EP rank.  A descriptor
cannot be selected with a runtime peer index, so the communication kernels use
the raw-address bulk-copy PTX form for one contiguous BF16 row chunk.
"""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op


_TMA_CACHE_HINT_EVICT_NORMAL = 0x1000000000000000
_TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000


@dsl_user_op
def tma_load_1d_raw(
    dst_smem,
    src_gmem_addr: Int64,
    mbar_smem,
    num_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one runtime-address GMEM-to-SMEM bulk copy."""

    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_gmem_addr.ir_value(),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_FIRST).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )

@dsl_user_op
def tma_store_1d_raw(
    dst_gmem_addr: Int64,
    src_smem,
    num_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one runtime-address SMEM-to-GMEM bulk copy."""

    llvm.inline_asm(
        None,
        [
            dst_gmem_addr.ir_value(),
            src_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_NORMAL).ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
        "[$0], [$1], $2, $3;",
        "l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
