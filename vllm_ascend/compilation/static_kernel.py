# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import atexit
import logging
import os
import signal
from collections.abc import Callable

import torch
from vllm import envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_node_count, get_world_group, in_the_same_node_as

from vllm_ascend.ascend_config import AscendCompilationConfig

logger = logging.getLogger(__name__)

# TODO: Temporary workaround. Remove this static kernel blacklist once these ops are supported.
STATIC_KERNEL_BLACKLIST = (
    "TopKV2",  # minimax-m3 910C
    "GenericBlockSparseAttention",  # minimax-m3 910C
    "QuantBatchMatmulV3",  # glm5.2 910C
    "TransposeQuantBatchMatMul",  # dsv4 950
    "QuantLightningIndexer",  # glm5.2 950
    "PoolKeyIndexer",  # glm5.3-flash
)


def _resolve_dp_capture_local_world_size(parallel: ParallelConfig) -> int:
    dist = torch.distributed
    world_size = dist.get_world_size()
    if world_size != parallel.world_size_across_dp:
        raise RuntimeError("Static kernel warmup requires the complete DP worker world")
    cpu_group = get_world_group().cpu_group
    if dist.get_process_group_ranks(cpu_group) != list(range(world_size)):
        raise RuntimeError("Static kernel warmup requires a CPU group spanning the complete DP worker world")

    # torch_npu shares the installed package path within consecutive blocks
    # of LOCAL_WORLD_SIZE ranks. Check actual shared-memory locality first.
    first_host = in_the_same_node_as(cpu_group, 0)
    local_world_size = sum(first_host)
    if not local_world_size or world_size % local_world_size:
        raise RuntimeError("Static kernel warmup requires equal worker counts in contiguous host-local rank blocks")
    for start in range(0, world_size, local_world_size):
        actual = first_host if start == 0 else in_the_same_node_as(cpu_group, start)
        expected = [start <= rank < start + local_world_size for rank in range(world_size)]
        if actual != expected:
            raise RuntimeError("Static kernel warmup requires contiguous host-local rank blocks")

    # Check overrides collectively so a misconfigured rank cannot leave its
    # peers waiting when the vendor compiler creates the install groups.
    overrides: list[str | None] = [None] * world_size
    dist.all_gather_object(overrides, os.getenv("LOCAL_WORLD_SIZE"), group=cpu_group)
    try:
        matches = all(value is None or int(value) == local_world_size for value in overrides)
    except ValueError as error:
        raise RuntimeError("Static kernel warmup requires integer LOCAL_WORLD_SIZE overrides") from error
    if not matches:
        raise RuntimeError(f"Static kernel warmup requires LOCAL_WORLD_SIZE={local_world_size} on every DP worker")
    return local_world_size


def initialize_static_kernel_runtime(vllm_config: VllmConfig, *, capture: bool = False) -> None:
    if capture:
        parallel = vllm_config.parallel_config
        if parallel.pipeline_parallel_size > 1:
            raise RuntimeError("Static kernel compilation during graph warmup requires PP=1")
        spec = vllm_config.speculative_config
        if spec is not None and not spec.enforce_eager:
            draft_parallel = spec.draft_parallel_config
            if draft_parallel is not None and draft_parallel.tensor_parallel_size != parallel.tensor_parallel_size:
                raise RuntimeError("Static kernel compilation requires target and draft to use the same TP ranks")
        if parallel.data_parallel_size > 1:
            os.environ["LOCAL_WORLD_SIZE"] = str(_resolve_dp_capture_local_world_size(parallel))

        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            import npugraph_ex
            from npugraph_ex._acl_concrete_graph import static_kernel
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        atexit.unregister(npugraph_ex._finalize)
        if not static_kernel._use_process_isolation:
            raise RuntimeError("Capture-time static kernels require isolated installation support")

    if "LOCAL_WORLD_SIZE" not in os.environ:
        parallel_config = vllm_config.parallel_config
        local_world_size = parallel_config.local_world_size * parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_external_lb:
            world_size = torch.distributed.get_world_size()
            if world_size != parallel_config.world_size_across_dp:
                raise RuntimeError("Static kernels require the complete DP worker world")
            local_world_size, remainder = divmod(world_size, get_node_count())
            if remainder:
                raise RuntimeError("Static kernels require equal worker counts on each node")
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

    from npugraph_ex._acl_concrete_graph.static_kernel import _set_static_kernel_blacklist

    _set_static_kernel_blacklist(list(STATIC_KERNEL_BLACKLIST))


def prepare_static_kernels(forward_fn: Callable[[CUDAGraphMode], None], config: AscendCompilationConfig) -> None:
    from npugraph_ex._acl_concrete_graph.static_kernel import compile_static_kernel

    options = {
        "force_eager": True,
        "static_kernel_compile": True,
        "super_kernel_optimize": config.enable_super_kernel,
        "super_kernel_optimize_options": {"dcci_after_kernel_end": [".*"]},
        "disable_static_kernel_compile_cache": envs.VLLM_DISABLE_COMPILE_CACHE,
    }
    forward_started = False

    def collect() -> None:
        nonlocal forward_started
        forward_started = True
        forward_fn(CUDAGraphMode.NONE)
        torch.npu.synchronize()

    compile_static_kernel(collect, config=options)
    if not forward_started:
        logger.warning("Static kernel compilation skipped operator collection. Running ordinary graph warmup.")
        forward_fn(CUDAGraphMode.NONE)


def release_static_kernels() -> None:
    """Release installed kernels after all participating workers have released their graphs."""
    from npugraph_ex._acl_concrete_graph.static_kernel import uninstall_static_kernel

    uninstall_static_kernel()
