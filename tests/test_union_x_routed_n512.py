"""CPU contracts for the routed Union-X adjacent-N pairing child."""


MLP_M = 256
LOGICAL_N = 128
PACKED_N = 2 * LOGICAL_N
CLUSTER_CTAS = 2


def test_pairing_covers_every_old_logical_tile_once() -> None:
    for row_blocks in (1, 3, 8, 64):
        for intermediate in (256, 512, 1024, 2048):
            assert intermediate % PACKED_N == 0
            old_tiles = {
                (row, logical_col)
                for row in range(row_blocks)
                for logical_col in range(intermediate // LOGICAL_N)
            }
            paired_tiles = {
                (row, pair_col * 2 + logical_n_offset)
                for row in range(row_blocks)
                for pair_col in range(intermediate // PACKED_N)
                for logical_n_offset in range(2)
            }
            assert paired_tiles == old_tiles
            assert (
                row_blocks * (intermediate // PACKED_N) * 2
                == len(old_tiles)
            )


def test_paired_task_preserves_down_ready_count() -> None:
    for intermediate in (256, 512, 1024, 2048):
        paired_tasks_per_row = intermediate // PACKED_N
        # Each paired cluster has two CTAs, and each CTA publishes once for
        # each old N128 hidden subtile after its store completes.
        actual_arrivals = paired_tasks_per_row * CLUSTER_CTAS * 2
        legacy_down_expected = CLUSTER_CTAS * (intermediate // LOGICAL_N)
        assert actual_arrivals == legacy_down_expected


def test_pairing_halves_a_gathers_without_changing_math() -> None:
    hidden = 4096
    for intermediate in (256, 1024, 2048):
        k64_stages = hidden // 64
        old_tasks = intermediate // LOGICAL_N
        paired_tasks = intermediate // PACKED_N
        assert paired_tasks * k64_stages * 2 == old_tasks * k64_stages
        assert paired_tasks * k64_stages < old_tasks * k64_stages
