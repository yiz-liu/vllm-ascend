# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CreateForwardFn, CudaGraphManager

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.static_kernel import prepare_static_kernels
from vllm_ascend.worker.v2.aclgraph_utils import AscendGraphManagerMixin

P = ParamSpec("P")


def _wrap_capture(
    capture: Callable[Concatenate[CudaGraphManager, CreateForwardFn, P], None],
) -> Callable[Concatenate[CudaGraphManager, CreateForwardFn, P], None]:
    @wraps(capture)
    def capture_with_static_kernels(
        self: CudaGraphManager, create_forward_fn: CreateForwardFn, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if not isinstance(self, AscendGraphManagerMixin):
            return capture(self, create_forward_fn, *args, **kwargs)
        config = get_ascend_config().ascend_compilation_config
        enabled = self.static_kernel_warmup_enabled and config.enable_static_kernel and not config.enable_npugraph_ex
        logger.info(
            "ACL graph preparation: static_kernel=%s, npugraph_ex=%s, compilation_mode=%s, capture_preparation=%s",
            config.enable_static_kernel,
            config.enable_npugraph_ex,
            self.vllm_config.compilation_config.mode,
            enabled,
        )
        if not enabled:
            return capture(self, create_forward_fn, *args, **kwargs)

        def create_prepared_forward(desc: BatchExecutionDescriptor, warmup: bool) -> Callable[[CUDAGraphMode], None]:
            forward_fn = create_forward_fn(desc, warmup=warmup)
            if not warmup:
                return forward_fn

            def forward(mode: CUDAGraphMode) -> None:
                if mode == CUDAGraphMode.NONE:
                    prepare_static_kernels(forward_fn, config)
                else:
                    forward_fn(mode)

            return forward

        return capture(self, create_prepared_forward, *args, **kwargs)

    return capture_with_static_kernels


def _wrap_end_segment(
    end_segment: Callable[[BreakableCUDAGraphCapture], None],
) -> Callable[[BreakableCUDAGraphCapture], None]:
    @wraps(end_segment)
    def end_segment_with_super_kernel(self: BreakableCUDAGraphCapture) -> None:
        graph = self._current_graph if self._capturing else None
        end_segment(self)
        if graph is not None and sys.exception() is None:
            if get_ascend_config().ascend_compilation_config.enable_super_kernel:
                graph.super_kernel_optimize(optimize_options={"dcci_after_kernel_end": [".*"]})
                logger.info_once("Super kernel optimization is enabled for Breakable ACL graph segments.")

    return end_segment_with_super_kernel


CudaGraphManager.capture = _wrap_capture(CudaGraphManager.capture)
BreakableCUDAGraphCapture._end_segment = _wrap_end_segment(BreakableCUDAGraphCapture._end_segment)
