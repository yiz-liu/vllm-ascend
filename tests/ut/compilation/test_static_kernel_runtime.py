# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.compilation import static_kernel


@pytest.fixture
def runtime(monkeypatch):
    runtime = ModuleType("npugraph_ex")
    runtime._finalize = lambda: None
    concrete_graph = ModuleType("npugraph_ex._acl_concrete_graph")
    vendor_kernel = ModuleType("npugraph_ex._acl_concrete_graph.static_kernel")
    vendor_kernel._use_process_isolation = True
    vendor_kernel._set_static_kernel_blacklist = Mock()
    concrete_graph.static_kernel = vendor_kernel
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(sys.modules, concrete_graph.__name__, concrete_graph)
    monkeypatch.setitem(sys.modules, vendor_kernel.__name__, vendor_kernel)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "")
    monkeypatch.delenv("LOCAL_WORLD_SIZE")
    return vendor_kernel


def configure_dp4(monkeypatch, hosts, overrides=None, world_size=64):
    cpu_group = object()
    dist = static_kernel.torch.distributed
    assert len(hosts) == world_size
    overrides = [None] * world_size if overrides is None else overrides
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(range(world_size)))
    gather = Mock(side_effect=lambda output, value, group: output.__setitem__(slice(None), overrides))
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(static_kernel, "get_node_count", lambda: len(set(hosts)))
    monkeypatch.setattr(static_kernel, "get_world_group", lambda: SimpleNamespace(cpu_group=cpu_group), raising=False)
    locality = Mock(side_effect=lambda group, source_rank=0: [host == hosts[source_rank] for host in hosts])
    monkeypatch.setattr(static_kernel, "in_the_same_node_as", locality, raising=False)

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=4,
            data_parallel_size_local=1,
            local_world_size=16,
            data_parallel_external_lb=False,
            world_size_across_dp=64,
        ),
        speculative_config=None,
    )
    return config, locality


@pytest.mark.parametrize("configured", [None, "16"])
def test_capture_uses_dp4_host_local_groups(monkeypatch, runtime, configured):
    if configured is not None:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", configured)
    config, locality = configure_dp4(monkeypatch, [rank // 16 for rank in range(64)], [configured] * 64)

    static_kernel.initialize_static_kernel_runtime(config, capture=True)

    assert os.environ["LOCAL_WORLD_SIZE"] == "16"
    assert [call.args[1] for call in locality.call_args_list] == [0, 16, 32, 48]
    runtime._set_static_kernel_blacklist.assert_called_once()


def test_capture_rejects_interleaved_hosts(monkeypatch, runtime):
    config, _ = configure_dp4(monkeypatch, [rank % 4 for rank in range(64)])

    with pytest.raises(RuntimeError, match="contiguous host-local rank blocks"):
        static_kernel.initialize_static_kernel_runtime(config, capture=True)
    runtime._set_static_kernel_blacklist.assert_not_called()


def test_capture_rejects_uneven_host_sizes(monkeypatch, runtime):
    config, _ = configure_dp4(monkeypatch, [0] * 16 + [1] * 15 + [2] * 17 + [3] * 16)

    with pytest.raises(RuntimeError, match="contiguous host-local rank blocks"):
        static_kernel.initialize_static_kernel_runtime(config, capture=True)


@pytest.mark.parametrize("override", ["64", "not-an-int"])
def test_capture_rejects_invalid_override_on_another_rank(monkeypatch, runtime, override):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "16")
    config, _ = configure_dp4(monkeypatch, [rank // 16 for rank in range(64)], ["16"] * 63 + [override])

    with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE"):
        static_kernel.initialize_static_kernel_runtime(config, capture=True)


def test_capture_rejects_incomplete_decode_world(monkeypatch, runtime):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "16")
    config, locality = configure_dp4(monkeypatch, [0] * 16, world_size=16)

    with pytest.raises(RuntimeError, match="complete DP worker world"):
        static_kernel.initialize_static_kernel_runtime(config, capture=True)
    locality.assert_not_called()


@pytest.mark.parametrize("capture", [False, True])
def test_single_dp_worker_world_keeps_existing_path(monkeypatch, runtime, capture):
    config, locality = configure_dp4(monkeypatch, [0] * 16, world_size=16)
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.world_size_across_dp = 16

    static_kernel.initialize_static_kernel_runtime(config, capture=capture)

    assert os.environ["LOCAL_WORLD_SIZE"] == "16"
    locality.assert_not_called()
