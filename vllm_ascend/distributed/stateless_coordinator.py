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

Two wiring paths exist for the same registration:
- ``NPUPlatform.on_stateless_process_group_created`` /
  ``on_stateless_process_group_destroyed`` platform lifecycle hooks
  (preferred when the supported vLLM revision provides them), and
- ``AscendStatelessGroupCoordinator`` (below), a subclass swapped in by
  ``vllm_ascend/patch/platform/patch_stateless_coordinator.py`` for
  revisions without the hooks.
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


class AscendStatelessGroupCoordinator(StatelessGroupCoordinator):
    """Stateless group coordinator whose torch PGs are registered in
    torch's global ``_world`` (see module docstring for why).

    The device (HCCL) and CPU (gloo) groups are registered right after
    they are created, and unregistered when the coordinator is
    destroyed, mirroring the lifecycle of the platform hooks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.device_group is not None:
            _register_pg(self.device_group, self.backend)
        if self.cpu_group is not None:
            _register_pg(self.cpu_group, "gloo")

    def destroy(self) -> None:
        try:
            super().destroy()
        finally:
            if self.device_group is not None:
                _unregister_pg(self.device_group)
            if self.cpu_group is not None:
                _unregister_pg(self.cpu_group)
