# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Register stateless process groups in torch's global ``_world``.

Upstream ``stateless_init_torch_distributed_process_group`` creates
process groups that are deliberately not registered in torch's global
``_world`` state, so ``torch.distributed`` module-level APIs cannot
resolve them. Ascend registers the stateless groups that HAVE such
consumers (see ``register_stateless_coordinator_pgs`` for the list):
- the EPLB group's CPU (gloo) process group, consumed by the gloo
  staged EPLB communicator (``dist.get_global_rank`` +
  ``batch_isend_irecv``) and by the dynamic-EPLB P2P transfer, and
- the MC2 group's torch PGs when CANN MegaMoe is active: its
  symm-buffer handshake resolves ranks via ``dist.get_rank(group=...)``
  on the HCCL device group.

The world/dp/ep groups talk through coordinator methods (PyHccl / TCP
store) and never consult ``_world``, so they stay unregistered to keep
the blast radius small.

Ordering constraint: ``DeviceCommunicatorBase.__init__`` treats a
``cpu_group`` as stateless iff it is absent from ``_world.pg_map``.
Registration must therefore happen *after* the coordinator (and its
device communicator) has been constructed, or the communicator would be
misclassified as stateful and take the ``dist.get_rank(group=...)``
path. Call sites:
- ``NPUWorker._init_worker_distributed_environment`` for the startup
  EPLB group, and
- ``AscendElasticEPScalingExecutor.prepare_reconfiguration`` for the
  standby EPLB group created on scale-up preparation.
Unregistration is paired in
``AscendElasticEPScalingExecutor._destroy_retired_groups``.
"""

from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import BackendConfig, _world
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator


def _register_pg(
    pg: ProcessGroup,
    backend: str,
    group_ranks: dict[int, int],
) -> None:
    """Register a stateless PG into torch's global ``_world``.

    ``group_ranks`` is the real ``{global_rank: group_rank}`` table derived
    from ``coordinator.ranks`` (group members of elastic groups are not
    contiguous global ranks, so an identity map would translate wrongly).
    """
    _world.pg_group_ranks[pg] = group_ranks
    _world.pg_map[pg] = (backend, pg.get_group_store())
    _world.pg_names[pg] = pg.group_name
    _world.pg_backend_config[pg] = str(BackendConfig(backend))


def _unregister_pg(pg: ProcessGroup) -> None:
    """Mirror ``_register_pg``: drop the group from ``_world``."""
    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)


def register_stateless_coordinator_pgs(
    coordinator: StatelessGroupCoordinator,
    include_device_group: bool = False,
) -> None:
    """Register a stateless coordinator's torch PGs into ``_world``.

    The CPU (gloo) group is always registered: the gloo staged EPLB
    communicator and the dynamic-EPLB P2P transfer resolve ranks via
    ``dist.get_global_rank`` / ``batch_isend_irecv``. The HCCL device
    group is additionally registered when a consumer resolves ranks on
    it through torch.distributed module-level APIs — currently the CANN
    MegaMoe symm-buffer handshake (``dist.get_rank(group=...)``); pass
    ``include_device_group=True`` for the MC2 group in that case.

    Ordering constraint: ``DeviceCommunicatorBase.__init__`` treats a
    group as stateless iff it is absent from ``_world.pg_map``. Call
    right after the coordinator (and its device communicator) has been
    constructed, or the communicator would be misclassified as stateful.
    """
    rank_map = {global_rank: idx for idx, global_rank in enumerate(coordinator.ranks)}
    if include_device_group and coordinator.device_group is not None:
        _register_pg(coordinator.device_group, coordinator.backend, rank_map)
    if coordinator.cpu_group is not None:
        _register_pg(coordinator.cpu_group, "gloo", rank_map)


def unregister_stateless_coordinator_pgs(
    coordinator: StatelessGroupCoordinator,
) -> None:
    """Mirror registration; both groups are attempted (no-op if absent)."""
    if coordinator.device_group is not None:
        _unregister_pg(coordinator.device_group)
    if coordinator.cpu_group is not None:
        _unregister_pg(coordinator.cpu_group)
