#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import os

from vllm.logger import logger

_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. E2E test
    conftest hooks do not run there, so global patches that affect scheduler
    and engine code must also be applied through these plugin entry points.
    """
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the NPU platform."""

    # Upstream auto-enables breakable CUDAGraph for selected architectures
    # only when the environment variable is absent. Keep it opt-in on Ascend
    # while preserving any value explicitly set by users.
    value = os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    logger.info_once(
        "Breakable CUDAGraph on Ascend is opt-in; using "
        "VLLM_USE_BREAKABLE_CUDAGRAPH=%s.",
        value,
        scope="process",
    )
    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_connector()
    register_engine()


def register_model_loader():
    _ensure_global_patch()

    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    _ensure_global_patch()

    from vllm_ascend.patch.hunyuan_vl_processor_compat import (
        install_hunyuan_vl_processor_compat,
    )

    install_hunyuan_vl_processor_compat()
    from .models import register_model

    register_model()


import vllm_ascend.logger  # noqa: E402, F401
