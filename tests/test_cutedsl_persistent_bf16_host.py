from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


_HOST_PATH = (
    Path(__file__).resolve().parents[1]
    / "mok"
    / "cutedsl"
    / "persistent_bf16.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "mok_cutedsl_persistent_bf16_host", _HOST_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
host = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = host
_SPEC.loader.exec_module(host)
_HOST_SOURCE = _HOST_PATH.read_text()


class PersistentBf16HostTest(unittest.TestCase):
    def test_plan_is_the_requested_cuda_specialization(self) -> None:
        plan = host.PersistentBf16Plan()
        plan.validate()
        self.assertEqual(plan.macrobatch_size, 32768)
        self.assertEqual(plan.minibatch_size, 4096)
        self.assertEqual(plan.num_comm_sms, 40)
        self.assertEqual(plan.clc_depth, 1)
        self.assertEqual(plan.gate_task_group_size, 1)
        self.assertEqual(plan.down_task_group_size, 1)
        with self.assertRaises(NotImplementedError):
            host.PersistentBf16Plan(macrobatch_size=65536).validate()

    def test_nine_output_shapes_match_cuda_bf16_wrapper(self) -> None:
        self.assertEqual(
            host.persistent_bf16_output_shapes(102400),
            (
                (32768, 4096),
                (102400, 1024),
                (32768, 1024),
                (102400, 1024),
                (32768, 1024),
                (102400, 1024),
                (32768, 1024),
                (102400, 4096),
                (32768, 4096),
            ),
        )

    def test_counter_lengths_match_forward_cuh_allocations(self) -> None:
        lengths = host.persistent_bf16_counter_lengths(102400, 2097152)
        shared_rows = 102400 // 256
        routed_rows = 2097152 // 256
        self.assertEqual(
            lengths.gate_up_tile_ready,
            (shared_rows + routed_rows) * 8,
        )
        self.assertEqual(
            lengths.hidden_row_block_ready,
            shared_rows + routed_rows,
        )
        self.assertEqual(lengths.x_routed_ready, 2097152 // 4096)
        self.assertEqual(lengths.y_routed_ready, 2097152 // 4096)
        self.assertEqual(lengths.y_routed_done, 2097152 // 128)

    def test_invalid_capacity_and_token_alignment_fail_closed(self) -> None:
        for tokens in (0, 256, 511, 513, 102401):
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    host.persistent_bf16_output_shapes(tokens)
        for capacity in (0, 255, 257, 4095, 4097):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    host.persistent_bf16_counter_lengths(102400, capacity)

    def test_partial_minibatch_capacity_uses_cuda_ceil(self) -> None:
        lengths = host.persistent_bf16_counter_lengths(512, 10240)
        self.assertEqual(lengths.x_routed_ready, 3)
        self.assertEqual(lengths.y_routed_ready, 3)
        self.assertEqual(lengths.y_routed_done, 80)

    def test_optional_swiglu_clamp_matches_cuda_host_contract(self) -> None:
        self.assertEqual(host._normalize_swiglu_limit(None), (0.0, False))
        self.assertEqual(host._normalize_swiglu_limit(4), (4.0, True))
        self.assertEqual(host._normalize_swiglu_limit(0.0), (0.0, True))
        for invalid in (-1, "4"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    host._normalize_swiglu_limit(invalid)

    def test_compile_and_run_expand_the_same_device_abi(self) -> None:
        tree = ast.parse(_HOST_SOURCE)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        compile_calls = [
            node
            for node in ast.walk(functions["compile_persistent_forward_bf16"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_mega_bf16"
        ]
        self.assertEqual(len(compile_calls), 1)
        call = compile_calls[0]
        self.assertEqual(len(call.args), 14)
        self.assertEqual(
            {keyword.arg for keyword in call.keywords},
            {
                "num_local_tokens",
                "schedule_capacity",
                "macrobatch_size",
                "minibatch_size",
                "num_comm_sms",
                "swiglu_limit",
                "is_clamped",
                "stream",
            },
        )
        run_calls = [
            node
            for node in ast.walk(functions["run_persistent_forward_bf16"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compile_persistent_forward_bf16"
        ]
        self.assertEqual(len(run_calls), 1)

    def test_run_resets_every_counter_on_the_launch_stream(self) -> None:
        tree = ast.parse(_HOST_SOURCE)
        compiled_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CompiledPersistentBf16Forward"
        )
        run = next(
            node
            for node in compiled_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        zero_calls = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "zero_"
        ]
        self.assertEqual(len(zero_calls), 1)
        self.assertIn("with torch.cuda.stream(self.stream):", _HOST_SOURCE)

    def test_public_executor_cache_rebinds_addresses_state_and_stream(self) -> None:
        class FakeDevice:
            def __init__(self, index: int) -> None:
                self.index = index

        class FakeStream:
            def __init__(self, name: str) -> None:
                self.name = name

        class FakeTensor:
            def __init__(self, tag: str, shape, dtype: str) -> None:
                self.tag = tag
                self.shape = tuple(shape)
                self.dtype = dtype
                self.zero_calls = 0
                self.recorded_streams = []

            def stride(self):
                strides = []
                running = 1
                for extent in reversed(self.shape):
                    strides.append(running)
                    running *= extent
                return tuple(reversed(strides))

            def zero_(self):
                self.zero_calls += 1
                return self

            def record_stream(self, stream) -> None:
                self.recorded_streams.append(stream)

        class FakeCuda:
            def __init__(self) -> None:
                self.current = None
                self.synchronized = []

            def current_stream(self, _device):
                return self.current

            def current_device(self) -> int:
                return 0

            def stream(self, _stream):
                return nullcontext()

            def synchronize(self, device) -> None:
                self.synchronized.append(device)

        fake_cuda = FakeCuda()
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        state_serial = []

        def make_state(*, num_local_tokens, schedule_capacity, device):
            del device
            serial = len(state_serial)
            outputs = tuple(
                FakeTensor(f"state{serial}.output{index}", shape, "torch.bfloat16")
                for index, shape in enumerate(
                    host.persistent_bf16_output_shapes(num_local_tokens)
                )
            )
            lengths = host.persistent_bf16_counter_lengths(
                num_local_tokens,
                schedule_capacity,
            )
            counters = tuple(
                FakeTensor(f"state{serial}.counter{index}", (length,), "torch.int32")
                for index, length in enumerate(
                    (
                        lengths.gate_up_tile_ready,
                        lengths.hidden_row_block_ready,
                        lengths.x_routed_ready,
                        lengths.y_routed_ready,
                        lengths.y_routed_done,
                    )
                )
            )
            state = host.PersistentBf16State(*outputs, *counters)
            state_serial.append(state)
            return state

        def make_workspace(serial: int):
            base = 100_000_000 + serial * 1_000_000
            return SimpleNamespace(
                device=FakeDevice(0),
                ep_size=8,
                num_local_tokens=512,
                hidden_size=4096,
                topk=10,
                schedule_capacity=4096,
                x_buffer=FakeTensor(
                    f"workspace{serial}.x", (512, 4096), "torch.bfloat16"
                ),
                combine_buffer=FakeTensor(
                    f"workspace{serial}.combine", (5120, 4096), "torch.bfloat16"
                ),
                x_buffer_ptrs=[base + index * 16 for index in range(8)],
                combine_buffer_ptrs=[
                    base + 500_000 + index * 16 for index in range(8)
                ],
            )

        def make_schedule(serial: int):
            return SimpleNamespace(
                peer_rank=FakeTensor(f"schedule{serial}.rank", (4096,), "torch.int32"),
                peer_token_idx=FakeTensor(
                    f"schedule{serial}.token", (4096,), "torch.int32"
                ),
                num_tokens=FakeTensor(f"schedule{serial}.count", (1,), "torch.int32"),
                tokens_per_expert=FakeTensor(
                    f"schedule{serial}.expert", (64,), "torch.int32"
                ),
            )

        def make_weights(serial: int):
            return (
                FakeTensor(f"weights{serial}.sg", (1024, 4096), "torch.bfloat16"),
                FakeTensor(
                    f"weights{serial}.rg", (64, 1024, 4096), "torch.bfloat16"
                ),
                FakeTensor(f"weights{serial}.su", (1024, 4096), "torch.bfloat16"),
                FakeTensor(
                    f"weights{serial}.ru", (64, 1024, 4096), "torch.bfloat16"
                ),
                FakeTensor(f"weights{serial}.sd", (4096, 1024), "torch.bfloat16"),
                FakeTensor(
                    f"weights{serial}.rd", (64, 4096, 1024), "torch.bfloat16"
                ),
            )

        compile_calls = []
        make_arg_calls = []
        launches = []

        def fake_make_mega_args(*args, **kwargs):
            state = args[-1]
            marker = (
                state.abi_outputs[0].tag,
                args[0][0],
                kwargs["stream"].name,
            )
            make_arg_calls.append(marker)
            return tuple((marker, index) for index in range(34)) + (marker,)

        def fake_executor(*runtime_args):
            launches.append(runtime_args)

        def fake_prepare(*args, **kwargs):
            compile_calls.append((args, kwargs))
            cute_args = fake_make_mega_args(*args, **kwargs)
            return fake_executor, cute_args[:27] + cute_args[-1:]

        workspace_a, workspace_b = make_workspace(0), make_workspace(1)
        schedule_a, schedule_b = make_schedule(0), make_schedule(1)
        weights_a, weights_b = make_weights(0), make_weights(1)
        stream_a, stream_b = FakeStream("stream-a"), FakeStream("stream-b")
        runtime_environment = (
            ("nvidia-cutlass-dsl", "4.6.2"),
            ("quack-kernels", "0.6.4"),
            ("CUTE_DSL_ENABLE_TVM_FFI", "1"),
            ("CUTE_DSL_ARCH", "sm_103a"),
            ("compile_options", ("cute.compile.default-options",)),
        )

        host.clear_persistent_bf16_executor_cache(synchronize=False)
        patches = (
            mock.patch.object(host, "prepare_persistent_bf16_state", make_state),
            mock.patch.object(host, "validate_persistent_bf16_call"),
            mock.patch.object(
                host,
                "_prepare_public_runtime_environment",
                return_value=runtime_environment,
            ),
            mock.patch.object(host, "_current_cuda_context_key", return_value=77),
            mock.patch.object(
                host,
                "_load_mega_runtime",
                return_value=(fake_make_mega_args, fake_prepare, 27),
            ),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            fake_cuda.current = stream_a
            result_a = host.forward_bf16(
                workspace_a, schedule_a, *weights_a,
                macrobatch_size=32768,
                minibatch_size=4096,
                swiglu_limit=None,
                num_comm_sms=40,
            )
            fake_cuda.current = stream_b
            result_b = host.forward_bf16(
                workspace_b, schedule_b, *weights_b,
                macrobatch_size=32768,
                minibatch_size=4096,
                swiglu_limit=None,
                num_comm_sms=40,
            )

        self.assertEqual(len(compile_calls), 1)
        self.assertEqual(len(host._EXECUTOR_CACHE), 1)
        self.assertEqual(tuple(host._EXECUTOR_CACHE.values()), (fake_executor,))
        self.assertEqual(len(state_serial), 2)
        self.assertIsNot(state_serial[0], state_serial[1])
        self.assertEqual(len(make_arg_calls), 2)
        self.assertEqual(
            make_arg_calls,
            [
                ("state0.output0", workspace_a.x_buffer_ptrs[0], "stream-a"),
                ("state1.output0", workspace_b.x_buffer_ptrs[0], "stream-b"),
            ],
        )
        self.assertEqual(len(launches), 2)
        self.assertEqual(launches[0][0][0], make_arg_calls[0])
        self.assertEqual(launches[1][0][0], make_arg_calls[1])
        self.assertEqual(result_a, state_serial[0].abi_outputs)
        self.assertEqual(result_b, state_serial[1].abi_outputs)
        for state, stream in zip(state_serial, (stream_a, stream_b)):
            self.assertTrue(all(counter.zero_calls == 1 for counter in state.counters))
            self.assertTrue(
                all(tensor.recorded_streams == [stream] for tensor in (*state.abi_outputs, *state.counters))
            )
        for tensors, workspace, schedule, stream in (
            (weights_a, workspace_a, schedule_a, stream_a),
            (weights_b, workspace_b, schedule_b, stream_b),
        ):
            owners = (
                workspace.x_buffer,
                workspace.combine_buffer,
                schedule.peer_rank,
                schedule.peer_token_idx,
                schedule.num_tokens,
                schedule.tokens_per_expert,
                *tensors,
            )
            self.assertTrue(all(tensor.recorded_streams == [stream] for tensor in owners))
        cache_key_text = repr(next(iter(host._EXECUTOR_CACHE)))
        self.assertIn("r54-ab6-acc2-prefixdecode-fixed6-dsl462-quack064", cache_key_text)
        self.assertIn("persistent-bf16-nine-output-five-counter-v1", cache_key_text)
        self.assertIn("nvidia-cutlass-dsl", cache_key_text)
        self.assertNotIn(str(workspace_a.x_buffer_ptrs[0]), cache_key_text)
        self.assertNotIn(str(workspace_b.x_buffer_ptrs[0]), cache_key_text)
        host.clear_persistent_bf16_executor_cache(synchronize=False)

    def test_public_forward_rejects_nonfrozen_config_before_runtime_import(self) -> None:
        workspace = SimpleNamespace(ep_size=8, hidden_size=4096, topk=10)
        shared_gate = SimpleNamespace(shape=(1024, 4096))
        routed_gate = SimpleNamespace(shape=(64, 1024, 4096))
        with mock.patch.object(
            host,
            "_prepare_public_runtime_environment",
            side_effect=AssertionError("runtime must remain lazy"),
        ):
            with self.assertRaises(NotImplementedError):
                host.forward_bf16(
                    workspace,
                    object(),
                    shared_gate,
                    routed_gate,
                    object(),
                    object(),
                    object(),
                    object(),
                    macrobatch_size=131072,
                    minibatch_size=4096,
                    swiglu_limit=None,
                    num_comm_sms=40,
                )
            with self.assertRaises(NotImplementedError):
                host.forward_bf16(
                    workspace,
                    object(),
                    shared_gate,
                    routed_gate,
                    object(),
                    object(),
                    object(),
                    object(),
                    macrobatch_size=32768,
                    minibatch_size=4096,
                    swiglu_limit=1.0,
                    num_comm_sms=40,
                )

    def test_public_forward_rejects_noncurrent_workspace_device_before_runtime_import(self) -> None:
        workspace = SimpleNamespace(
            device=SimpleNamespace(index=0),
            ep_size=8,
            hidden_size=4096,
            topk=10,
        )
        shared_gate = SimpleNamespace(shape=(1024, 4096))
        routed_gate = SimpleNamespace(shape=(64, 1024, 4096))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(current_device=lambda: 1),
        )
        with (
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
            mock.patch.object(
                host,
                "_prepare_public_runtime_environment",
                side_effect=AssertionError("runtime must remain lazy"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "workspace CUDA device"):
                host.forward_bf16(
                    workspace,
                    object(),
                    shared_gate,
                    routed_gate,
                    object(),
                    object(),
                    object(),
                    object(),
                    macrobatch_size=32768,
                    minibatch_size=4096,
                    swiglu_limit=None,
                    num_comm_sms=40,
                )

    def test_executor_cache_clear_synchronizes_known_devices(self) -> None:
        class FakeCuda:
            def __init__(self) -> None:
                self.synchronized = []

            def synchronize(self, device) -> None:
                self.synchronized.append(device)

        fake_cuda = FakeCuda()
        host._EXECUTOR_CACHE[("executor",)] = object()
        host._EXECUTOR_CACHE_DEVICES.update((3, 1))
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=fake_cuda)}):
            host.clear_persistent_bf16_executor_cache()
        self.assertEqual(fake_cuda.synchronized, [1, 3])
        self.assertEqual(host._EXECUTOR_CACHE, {})
        self.assertEqual(host._EXECUTOR_CACHE_DEVICES, set())

    def test_public_runtime_selects_tvm_ffi_before_cutlass_import(self) -> None:
        versions = {
            "nvidia-cutlass-dsl": "4.6.2",
            "quack-kernels": "0.6.4",
        }
        with (
            mock.patch.object(host, "sys", SimpleNamespace(modules={})),
            mock.patch.dict(host.os.environ, {}, clear=True),
            mock.patch.object(
                host.metadata,
                "version",
                side_effect=lambda package: versions[package],
            ),
        ):
            signature = host._prepare_public_runtime_environment()
            self.assertEqual(host.os.environ["CUTE_DSL_ENABLE_TVM_FFI"], "1")
            self.assertEqual(host.os.environ["CUTE_DSL_ARCH"], "sm_103a")
        self.assertIn(("nvidia-cutlass-dsl", "4.6.2"), signature)
        self.assertIn(("quack-kernels", "0.6.4"), signature)
        self.assertIn(("compile_options", ("cute.compile.default-options",)), signature)


if __name__ == "__main__":
    unittest.main()
