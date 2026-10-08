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

from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


def _accelerator_synchronize() -> None:
    # torch.accelerator APIs raise on CPU-only builds; the mask manager is
    # unit-testable without a device, so guard the synchronize.
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


logger = init_logger(__name__)


class _NpuAll2AllManager:
    """All2All-manager adapter for the CANN MegaMoe backend.

    Implements the upstream ``All2AllManagerBase`` mask API consumed by the
    Elastic EP executor: EP membership changes become writes on the MegaMoe
    symm buffer's device-side rank mask, so captured ACL graphs survive
    them. Mutable state is class-level (mirroring upstream
    ``NixlEPAll2AllManager._buffer``) because the binding must outlive the
    executor's process-group switches.
    """

    # Class-level shared state: survives process-group switches.
    _mega_moe_buffer: Any = None
    _ep_to_mc2: tuple[int, ...] = ()
    _ep_world_size: int = 0
    _dead: set[int] = set()
    _mask: torch.Tensor | None = None

    def __init__(self, ep_world_size: int | None = None):
        if ep_world_size is not None and ep_world_size > 0:
            _NpuAll2AllManager.bind_ep_world_size(ep_world_size)

    @classmethod
    def bind_ep_world_size(cls, ep_world_size: int) -> None:
        # First binding wins: graphs stay in the startup rank space.
        if cls._ep_world_size == 0:
            cls._ep_world_size = ep_world_size

    @property
    def uses_mega_moe(self) -> bool:
        return _NpuAll2AllManager._mega_moe_buffer is not None

    @torch.inference_mode()
    def bind_mega_moe_buffer(self, buffer, ep_to_mc2: list[int]) -> None:
        """Attach the rank mask before any ACL graph is captured.

        The CPU ``_dead`` set stays authoritative; binding replays already
        dead ranks into the device mask.
        """
        ep_world_size = _NpuAll2AllManager._ep_world_size
        if ep_world_size == 0:
            ep_world_size = getattr(buffer, "ep_world_size", len(ep_to_mc2))
            _NpuAll2AllManager.bind_ep_world_size(ep_world_size)
        if sorted(ep_to_mc2) != list(range(ep_world_size)):
            raise ValueError(
                "Elastic EP graph reuse requires the EP and MC2 rank sets "
                f"to match (got ep_to_mc2={ep_to_mc2}, ep_world_size={ep_world_size})."
            )
        if self.uses_mega_moe:
            # Re-binding must never resurrect an already dead peer.
            return
        buffer.clean_mask_buffer()
        for rank in sorted(_NpuAll2AllManager._dead):
            buffer.update_mask_buffer(ep_to_mc2[rank], True)
        _NpuAll2AllManager._ep_to_mc2 = tuple(ep_to_mc2)
        _NpuAll2AllManager._mega_moe_buffer = buffer
        logger.info(
            "Bound MegaMoe rank-mask buffer for %d EP ranks before graph capture.",
            ep_world_size,
        )

    @torch.inference_mode()
    def update_mask(self, rank: int, masked: bool = True) -> None:
        """Mark an EP rank dead/alive on both the CPU set and device mask."""
        ep_world_size = _NpuAll2AllManager._ep_world_size
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < ep_world_size:
            raise ValueError(f"EP rank must be in [0, {ep_world_size}), got {rank}.")
        if self.uses_mega_moe:
            if not masked and rank in _NpuAll2AllManager._dead:
                # Recovery must stop outstanding device work before unmasking.
                self.clean_buffers()
            _NpuAll2AllManager._mega_moe_buffer.update_mask_buffer(_NpuAll2AllManager._ep_to_mc2[rank], masked)
            _accelerator_synchronize()
        if masked:
            _NpuAll2AllManager._dead.add(rank)
        else:
            _NpuAll2AllManager._dead.discard(rank)

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        # MegaMoe kernels do not self-detect faults; masking is executor-driven.
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        """EP-order liveness mask (0 = alive), refreshed from ``_dead``.

        Stable address, so a tensor captured in a graph keeps observing
        the latest membership.
        """
        ep_world_size = _NpuAll2AllManager._ep_world_size
        if ep_world_size == 0:
            return torch.zeros(1, dtype=torch.bool, device="cpu")
        if _NpuAll2AllManager._mask is None or _NpuAll2AllManager._mask.numel() < ep_world_size:
            _NpuAll2AllManager._mask = torch.zeros(ep_world_size, dtype=torch.int32)
        mask = _NpuAll2AllManager._mask[:ep_world_size]
        mask.fill_(0)
        for rank in _NpuAll2AllManager._dead:
            mask[rank] = 1
        return mask

    @torch.inference_mode()
    def clean_buffers(self) -> None:
        """Clear fused-communication flags (dead ranks stay masked)."""
        if self.uses_mega_moe:
            _NpuAll2AllManager._mega_moe_buffer.get_local_buffer_tensor(torch.uint8).zero_()
            _accelerator_synchronize()

    # Upstream stage/commit hooks; CANN has no incremental rank-connect for
    # the symm buffer, so masking is driven by the EEP executor instead.
    def stage_ep_size(self) -> None:
        pass

    def commit_ep_size(self) -> None:
        pass


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
        stateless: bool | None = None,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all,
            stateless=stateless,
        )
        self.device = torch.npu.current_device()

        # Create pyhccl_comm handle for batch_transfer_weights in elastic_ep
        self.pyhccl_comm: PyHcclCommunicator | None = None
        if self.world_size > 1 and tcp_store_group is not None:
            self.pyhccl_comm = PyHcclCommunicator(group=tcp_store_group, device=self.device, warmup=False)

        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        self.all2all_manager = _NpuAll2AllManager(self.world_size)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.pyhccl_comm is not None:
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()
            input_size = input_.size()
            # Use concat-style all-gather: stack-style has torch.compile
            # compatibility issues (pytorch/pytorch#138795).
            output_size = (input_size[0] * self.world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
            # All-gather.
            output_tensor = self.pyhccl_comm.all_gather(input_, output_tensor)
            # Reshape
            output_tensor = output_tensor.reshape((self.world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :]
            )
            return output_tensor
        else:
            return super().all_gather(input_, dim)

    def destroy(self):
        if self.pyhccl_comm is not None:
            self.pyhccl_comm.destroy()
            self.pyhccl_comm = None

    def batch_isend_irecv(self, p2p_ops: list):
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is not None and not pyhccl_comm.disabled:
            pyhccl_comm.batch_isend_irecv(p2p_ops)
        else:
            raise ValueError("No PyHccl communicator found")
