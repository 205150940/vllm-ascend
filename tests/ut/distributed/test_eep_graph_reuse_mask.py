# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Unit tests for the MegaMoe rank-mask manager used by Elastic EP scale-down
graph reuse (``_NpuAll2AllManager``).

The mask state lives at class level so that it survives the process-group
switch performed by the executor: manager instances are created per
coordinator, but the MegaMoe symmetric buffer (and the mask on it) must stay
alive for captured ACL graphs to keep replaying. These tests pin that
contract on CPU with a fake symm buffer.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.device_communicators.npu_communicator import _NpuAll2AllManager


class FakeSymmBuffer:
    """CPU stand-in for the CANN MegaMoe symmetric buffer mask API."""

    def __init__(self, world_size: int = 4):
        self.ep_world_size = world_size
        self.mask_buffer = None
        self.ccl = torch.ones(32, dtype=torch.uint8)
        self.destroyed = False

    def clean_mask_buffer(self):
        if self.mask_buffer is None:
            self.mask_buffer = torch.zeros(self.ep_world_size, dtype=torch.int32)
        else:
            self.mask_buffer.zero_()

    def update_mask_buffer(self, rank: int, masked: bool):
        self.mask_buffer[rank] = int(masked)

    def get_local_buffer_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        assert dtype == torch.uint8
        return self.ccl

    def destroy(self):
        self.destroyed = True


@pytest.fixture(autouse=True)
def _reset_manager_state():
    _NpuAll2AllManager.reset_for_test()
    yield
    _NpuAll2AllManager.reset_for_test()


def _bind(world_size: int = 4, ep_to_mc2: list[int] | None = None) -> FakeSymmBuffer:
    manager = _NpuAll2AllManager(world_size)
    buffer = FakeSymmBuffer(world_size)
    manager.bind_mega_moe_buffer(buffer, ep_to_mc2 or list(range(world_size)))
    return buffer


def test_mask_state_survives_manager_replacement():
    """Group switch = a new manager instance; the mask binding must not.

    This is the property that keeps captured graphs valid across the
    Elastic EP switch: the buffer object and its mask are shared state.
    """
    first = _NpuAll2AllManager(4)
    buffer = FakeSymmBuffer(4)
    first.bind_mega_moe_buffer(buffer, [0, 1, 2, 3])
    first.update_mask(2, masked=True)

    # A smaller standby coordinator (scale-down) constructs a new manager.
    second = _NpuAll2AllManager(2)
    assert second.uses_mega_moe
    assert second.query_active_mask().tolist() == [0, 0, 1, 0]
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 0]

    # Masking through the new instance updates the same device buffer.
    second.update_mask(3, masked=True)
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 1]


def test_scale_down_mask_semantics_and_storage_stability():
    buffer = _bind(4)
    address = buffer.mask_buffer.data_ptr()
    manager = _NpuAll2AllManager()

    # Scale-down 8->4 style: mask the upper half of the EP rank space.
    manager.update_mask(2, masked=True)
    manager.update_mask(3, masked=True)
    assert manager.query_active_mask().tolist() == [0, 0, 1, 1]
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 1]

    # The mask buffer storage must never move: captured graphs reference it.
    manager.clean_buffers()
    assert not buffer.ccl.any()
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 1]
    assert buffer.mask_buffer.data_ptr() == address


def test_mask_rank_translation_uses_ep_to_mc2():
    buffer = _bind(4, ep_to_mc2=[0, 2, 1, 3])
    manager = _NpuAll2AllManager()
    manager.update_mask(1, masked=True)
    manager.update_mask(3, masked=True)
    # EP space liveness vs MC2 space device bytes.
    assert manager.query_active_mask().tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 1]


def test_unmask_clears_stale_comm_flags_first():
    buffer = _bind(4)
    manager = _NpuAll2AllManager()
    manager.update_mask(1, masked=True)
    buffer.ccl.fill_(1)

    update_calls = []
    original_update = buffer.update_mask_buffer

    def spying_update(rank, masked):
        update_calls.append((rank, masked))
        original_update(rank, masked)

    buffer.update_mask_buffer = spying_update
    manager.update_mask(1, masked=False)
    assert not buffer.ccl.any(), "recovery must zero fused comm flags before unmasking"
    assert manager.query_active_mask().tolist() == [0, 0, 0, 0]
    assert buffer.mask_buffer.tolist() == [0, 0, 0, 0]


def test_rebind_never_resurrects_dead_peers():
    buffer = _bind(4)
    manager = _NpuAll2AllManager()
    manager.update_mask(2, masked=True)
    manager.bind_mega_moe_buffer(buffer, [0, 1, 2, 3])
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 0]


def test_bind_rejects_mismatched_rank_sets():
    manager = _NpuAll2AllManager(4)
    buffer = FakeSymmBuffer(4)
    with pytest.raises(ValueError, match="rank sets"):
        manager.bind_mega_moe_buffer(buffer, [0, 1, 2])  # not a permutation
    assert not manager.uses_mega_moe


def test_update_mask_validates_rank_bounds():
    _bind(4)
    manager = _NpuAll2AllManager()
    with pytest.raises(ValueError, match="EP rank"):
        manager.update_mask(4, masked=True)
    with pytest.raises(ValueError, match="EP rank"):
        manager.update_mask(-1, masked=True)
    with pytest.raises(ValueError, match="EP rank"):
        manager.update_mask(True)  # bool is not a rank


def test_unbound_manager_is_inert():
    manager = _NpuAll2AllManager(4)
    assert not manager.uses_mega_moe
    manager.update_mask(1, masked=True)  # CPU-set only, no device write
    assert manager.query_active_mask().tolist() == [0, 1, 0, 0]
    manager.clean_buffers()  # must not raise without a bound buffer


def test_mask_remote_ranks_round_trip():
    buffer = _bind(4)
    manager = _NpuAll2AllManager()
    manager.update_mask(3, masked=True)  # pre-existing dead rank
    with manager.mask_remote_ranks():
        assert buffer.mask_buffer.tolist() == [1, 1, 1, 1]
    # Dead rank stays masked after the context exits.
    assert buffer.mask_buffer.tolist() == [0, 0, 0, 1]


def test_mask_remote_ranks_noop_when_unbound():
    manager = _NpuAll2AllManager()
    with manager.mask_remote_ranks():
        pass  # must not raise


def test_manager_plays_the_upstream_all2all_hook_contract():
    """The executor calls these hooks regardless of the reuse path."""
    _bind(4)
    manager = _NpuAll2AllManager()
    manager.stage_ep_size()
    manager.commit_ep_size()
    assert manager.support_fault_tolerance is False
    assert manager.query_fault().tolist() == [False]


def test_query_active_mask_keeps_a_stable_tensor_address():
    _bind(4)
    manager = _NpuAll2AllManager()
    first = manager.query_active_mask()
    ptr = first.data_ptr()
    manager.update_mask(0, masked=True)
    second = manager.query_active_mask()
    assert second.data_ptr() == ptr, "mask tensor address must be capture-stable"
    assert second.tolist() == [1, 0, 0, 0]


def test_npu_communicator_binds_ep_world_size():
    """NPUCommunicator must forward its world size so the first coordinator
    establishes the (graph-space) EP rank bounds."""
    from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator

    communicator = Mock(spec=NPUCommunicator)
    communicator.world_size = 8
    _NpuAll2AllManager(communicator.world_size)
    assert _NpuAll2AllManager._ep_world_size == 8
