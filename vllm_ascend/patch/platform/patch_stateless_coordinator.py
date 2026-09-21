#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# Swap ``StatelessGroupCoordinator`` for the Ascend subclass that registers
# the stateless HCCL/gloo torch process groups into torch's global
# ``_world``, so ``torch.distributed`` module-level APIs can resolve them
# (required by e.g. the async EPLB gloo staging communicator during
# elastic EP). See the patch entry in ``vllm_ascend/patch/__init__.py``.

import vllm.distributed.stateless_coordinator as _stateless_coordinator

from vllm_ascend.distributed.stateless_coordinator import (
    AscendStatelessGroupCoordinator,
)

_stateless_coordinator.StatelessGroupCoordinator = AscendStatelessGroupCoordinator
