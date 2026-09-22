# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Register stateless process groups in torch's global ``_world``.

Upstream ``stateless_init_torch_distributed_process_group`` creates
process groups that are not registered in torch's global ``_world``
state, so ``torch.distributed`` module-level APIs cannot find them.
This is required by e.g. ``broadcast``/``send``/``recv`` global-rank
translation on the stateless world/dp/ep groups and by the async EPLB
communicator issuing ``batch_isend_irecv`` on the stateless gloo group
during elastic EP.

Registration is done at the vLLM Ascend-owned call sites right after
the coordinators are created:
- ``NPUWorker._init_worker_distributed_environment`` for the startup
  world/dp/ep/eplb groups, and
- ``AscendElasticEPScalingExecutor.prepare_reconfiguration`` for the
  standby groups created on scale-up preparation.
Unregistration is paired in
``AscendElasticEPScalingExecutor._destroy_retired_groups``.
"""

from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import BackendConfig, _world
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator


def _register_pg(pg: ProcessGroup, backend: str) -> None:
    """Register a stateless PG into torch's global ``_world``.

    Each rank of a stateless group maps 1:1 to itself (rank i in the
    group is global rank i).
    """
    _world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
    _world.pg_map[pg] = (backend, pg.get_group_store())
    _world.pg_names[pg] = pg.group_name
    _world.pg_backend_config[pg] = str(BackendConfig(backend))

    # The WORLD group is used as torch's default process group.
    if "WORLD" in (pg.group_name or ""):
        _world.default_pg = pg


def _unregister_pg(pg: ProcessGroup) -> None:
    """Mirror ``_register_pg``: drop the group from ``_world``."""
    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)


def register_stateless_coordinator_pgs(
    coordinator: StatelessGroupCoordinator,
) -> None:
    """Register a stateless coordinator's torch PGs into ``_world``.

    Call right after the coordinator is created; pair with
    ``unregister_stateless_coordinator_pgs`` when it is destroyed.
    """
    if coordinator.device_group is not None:
        _register_pg(coordinator.device_group, coordinator.backend)
    if coordinator.cpu_group is not None:
        _register_pg(coordinator.cpu_group, "gloo")


def unregister_stateless_coordinator_pgs(
    coordinator: StatelessGroupCoordinator,
) -> None:
    """Mirror ``register_stateless_coordinator_pgs`` on destruction."""
    if coordinator.device_group is not None:
        _unregister_pg(coordinator.device_group)
    if coordinator.cpu_group is not None:
        _unregister_pg(coordinator.cpu_group)
