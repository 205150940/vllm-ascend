# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the upstream FT recovery connection-setup steps.

The upstream ``WorkerSentinel.retry`` rebuilds the recovery TCP store
connection and the DP/TP/EP/EPLB Gloo cpu_groups through module-level
helpers. This patch wraps those helpers so their durations join the active
scale-down timer (see ``ft_scale_down_timing``) without modifying upstream
vLLM. With no scale-down round in flight the wrappers are pass-through.
"""

from functools import wraps

import vllm.v1.worker.sentinel.gpu_worker_sentinel as _gpu_sentinel
from vllm.logger import init_logger

from vllm_ascend.worker.sentinel.ft_scale_down_timing import get_active_timer

logger = init_logger(__name__)

_orig_reinit_cpu_group = _gpu_sentinel._reinit_cpu_group
_orig_get_cached_tcp_store_client = _gpu_sentinel.get_cached_tcp_store_client


def _stage_name_for_group_key(key: str) -> str:
    """Derive a stage name from a recovery port key.

    Upstream keys look like ``ft_dp_port_<round>_<dp_index>``,
    ``ft_tp_port_<round>_<rank>``, ``ft_ep_port_<round>`` and
    ``ft_eplb_port_<round>``; map them to ``reinit_dp_cpu_group`` etc.
    """
    group_name = key.split("_port_")[0].removeprefix("ft_")
    return f"reinit_{group_name}_cpu_group"


@wraps(_orig_reinit_cpu_group)
def _timed_reinit_cpu_group(group, store, key, host, rank, size):
    timer = get_active_timer()
    if timer is None:
        return _orig_reinit_cpu_group(group, store, key, host, rank, size)
    with timer.stage(_stage_name_for_group_key(key)):
        return _orig_reinit_cpu_group(group, store, key, host, rank, size)


@wraps(_orig_get_cached_tcp_store_client)
def _timed_get_cached_tcp_store_client(host, port):
    timer = get_active_timer()
    if timer is None:
        return _orig_get_cached_tcp_store_client(host, port)
    with timer.stage("connect_recovery_store"):
        return _orig_get_cached_tcp_store_client(host, port)


def _apply_patch() -> None:
    _gpu_sentinel._reinit_cpu_group = _timed_reinit_cpu_group
    # The store client is bound into the sentinel module's namespace, so the
    # retry flow resolves the wrapper here; vllm.distributed.utils stays
    # untouched for other callers.
    _gpu_sentinel.get_cached_tcp_store_client = _timed_get_cached_tcp_store_client
    logger.debug("Applied FT scale-down timing patch on gpu_worker_sentinel")


_apply_patch()
