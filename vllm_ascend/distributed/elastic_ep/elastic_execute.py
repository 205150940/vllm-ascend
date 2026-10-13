# Adapted from vLLM's elastic_execute.py with Ascend-specific changes:
# NPU/ACL graphs, quantized weight transfer, MC2 comm groups, PyHccl EPLB.

import gc
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributed import P2POp
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import set_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.distributed.elastic_ep.standby_state import get_standby_eplb_group
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.platforms import current_platform
from vllm.utils import is_moe_layer
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
)
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    pop_ascend_standby_groups,
)
from vllm_ascend.distributed.eplb.eplb_communicator import AscendHixlEplbCommunicator
from vllm_ascend.distributed.parallel_state import (
    GroupCoordinator,
    _replace_ascend_active_groups,
    get_mc2_group,
)
from vllm_ascend.distributed.stateless_coordinator import (
    register_stateless_coordinator_pgs,
    unregister_stateless_coordinator_pgs,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod


def setup_moe_comm_and_quant_method(module: nn.Module) -> None:
    if isinstance(
        quant_method := getattr(module.routed_experts.quant_method, "quant_method", None),
        AscendW8A8DynamicFusedMoEMethod,
    ):
        try:
            device_group = get_mc2_group().device_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""
    setup_moe_comm_method(module.moe_config)


def batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None:
    # Ascend HCCL P2P weight transfer. Differs from upstream: collects params
    # from __dict__/AttentionImplBase and negotiates param names via the TCP
    # store. HCCL, like PyNccl, transfers flat memory (data_ptr + numel) and
    # does not honor tensor strides, so non-contiguous params are materialized
    # into a contiguous copy and copied back on the receive side.
    device_comm = dp_group.device_communicator
    tcp_store_group = dp_group.tcp_store_group
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            if isinstance(weight, torch.Tensor):
                expert_weights_set.add(weight.data_ptr())
            else:
                expert_weights_set.update(w.data_ptr() for w in weight)

    state_dict = model.state_dict()
    all_params = []
    all_params_ptrs = set()
    all_params_name = []

    for name, param in state_dict.items():
        if name.endswith("expert_map"):
            continue
        ptr = param.data_ptr()
        if ptr not in all_params_ptrs and ptr not in expert_weights_set:
            if param.device.type == "npu":
                all_params.append(param.data)
                all_params_ptrs.add(ptr)
                all_params_name.append(name)

    def handle_sub_module(submodule, submodule_name):
        for attr_name, attr_value in submodule.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                data_ptr = attr_value.data_ptr()
                if data_ptr not in all_params_ptrs and data_ptr not in expert_weights_set:
                    if attr_value.device.type == "npu":
                        all_params.append(attr_value)
                        all_params_ptrs.add(data_ptr)
                        all_params_name.append(submodule_name + "." + attr_name)
            if isinstance(attr_value, AttentionImplBase):
                handle_sub_module(attr_value, submodule_name + "." + attr_name)

    for module_name, module in model.named_modules():
        handle_sub_module(module, module_name)

    if is_sender:
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
    else:
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)

    if len(all_params_name) == len(peer_rank_all_params_name):
        assert all_params_name == peer_rank_all_params_name, (
            "Elastic EP weight transfer: sender/receiver parameter lists differ"
        )
        common_names = all_params_name
    else:
        common_names = sorted(set(all_params_name) & set(peer_rank_all_params_name))
        assert common_names, "Elastic EP weight transfer: sender/receiver parameter lists have no names in common"

    name_to_param = dict(zip(all_params_name, all_params))
    all_params = [name_to_param[name] for name in common_names]

    assert len(all_params) > 0
    p2p_ops = []
    for param in all_params:
        transfer_param = param.contiguous()
        op = object.__new__(P2POp)
        op.op = torch.distributed.isend if is_sender else torch.distributed.irecv
        op.tensor = transfer_param
        op.group_peer = peer_rank
        p2p_ops.append(op)
        if transfer_param is not param:
            device_comm.batch_isend_irecv(p2p_ops)
            p2p_ops.clear()
            if not is_sender:
                param.copy_(transfer_param)
    if p2p_ops:
        device_comm.batch_isend_irecv(p2p_ops)


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    def _batch_transfer_weights(
        self,
        model: nn.Module,
        is_sender: bool,
        peer_rank: int,
        dp_group: StatelessGroupCoordinator,
        expert_weights: Sequence[Iterable[torch.Tensor]],
    ) -> None:
        # Ascend HCCL P2P transfer; see the module-level
        # ``batch_transfer_weights`` for how it differs from upstream.
        batch_transfer_weights(
            model=model,
            is_sender=is_sender,
            peer_rank=peer_rank,
            dp_group=dp_group,
            expert_weights=expert_weights,
        )

    def _create_backend_standby_groups(
        self,
        reconfig_request: ReconfigureDistributedRequest,
        new_dp_size: int,
        new_world_size_across_dp: int,
    ) -> None:
        # The standby EPLB group is the only stateless group whose torch PGs
        # are consumed through torch.distributed module-level APIs (the gloo
        # staged EPLB communicator and the dynamic-EPLB P2P transfer pass
        # global ranks); register it so those calls resolve the group. Paired
        # with unregistration in _destroy_retired_groups.
        standby_eplb_group = get_standby_eplb_group()
        if standby_eplb_group is not None:
            register_stateless_coordinator_pgs(standby_eplb_group)
        # The MC2 standby rendezvous completes only once every member has
        # joined, so it must be created after the upstream world/dp/ep/eplb
        # groups, matching the new worker's boot order.
        create_ascend_standby_groups(
            new_dp_size=new_dp_size,
            new_world_size_across_dp=new_world_size_across_dp,
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
        )

    def _release_cuda_graphs(self) -> None:
        if isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        ACLGraphWrapper.clear_all_graphs()

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        reset_graph_params()

        mgr = self.worker.model_runner.cudagraph_manager
        if mgr is not None:
            mgr.graphs.clear()
            mgr._graphs_captured = False
            # NPU graph pools cache old allocations; a fresh pool is
            # required before re-capture (NPUCachingAllocator.cpp:2106).
            mgr.pool = current_platform.graph_pool_handle()
            if hasattr(mgr, "capture_sizes"):
                capture_sizes = mgr.capture_sizes
                if self.worker.model_runner.use_aclgraph:
                    set_graph_params(capture_sizes)
                    if self.worker.model_runner.speculative_config:
                        set_draft_graph_params(capture_sizes)

        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def _destroy_retired_groups(self, groups: tuple[GroupCoordinator | None, ...]) -> None:
        # Pair of the ``_register_pg`` call in prepare_reconfiguration:
        # drop the retired stateless groups from torch's ``_world`` before
        # their PGs are shut down, so stale entries don't keep the PG
        # objects (and their HCCL/gloo comms) alive across scaling rounds.
        # Unregistration must never skip the actual destroy.
        try:
            for group in groups:
                if isinstance(group, StatelessGroupCoordinator):
                    unregister_stateless_coordinator_pgs(group)
        finally:
            super()._destroy_retired_groups(groups)

    def switch_and_remove(self) -> None:
        super().switch_and_remove()
        retired_mc2 = _replace_ascend_active_groups(mc2=None)
        if retired_mc2 is not None:
            retired_mc2.destroy()

    def _expert_load_view(self, eplb_model_state) -> torch.Tensor:
        # The reconfigured active view, not the maximum-capacity buffer, so
        # set_layer_state derives the per-layer local expert range from the
        # active slot count.
        return eplb_model_state.expert_load_pass

    def _on_topology_switched(
        self,
        retired_groups: tuple[GroupCoordinator | None, ...],
        moe_modules: list[nn.Module],
    ) -> tuple[GroupCoordinator | None, ...]:
        # Swap the MC2 standby group in and refresh the MoE communication
        # methods against the switched groups.
        retired_mc2 = _replace_ascend_active_groups(mc2=pop_ascend_standby_groups())
        self.worker.model_runner.dp_size = self.worker.parallel_config.data_parallel_size
        self.worker.model_runner.dp_rank = self.worker.parallel_config.data_parallel_rank
        for module in moe_modules:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()
        self._setup_moe_comm_and_quant_method()
        if retired_mc2 is not None:
            # The retired MC2 group joins the upstream retired groups so the
            # executor's async cleanup thread destroys it after the switch.
            return (*retired_groups, retired_mc2)
        return retired_groups

    def _close_hixl_communicator(self) -> None:
        """Deregister the active HIXL segments before EEP frees memory.

        HIXL register_mem pins the NPU segments at the driver level: while
        the registration is alive, releasing those segments (the switch
        rebuilds the expert weights and empties the cache) fails with
        AclrtUnmapMem Resource_Busy. The reshuffle is the retired
        communicator's last use, so deregister right after it.
        """
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is None:
            return
        model_state = eplb_state.model_states.get(self.worker.model_runner.model_config.compute_hash())
        if model_state is None:
            return
        retired = model_state.communicator
        if isinstance(retired, AscendHixlEplbCommunicator):
            retired._close()

    def _after_scale_down_reshuffle(self) -> None:
        self._close_hixl_communicator()

    def commit_scale_up(self, is_existing_worker: bool) -> None:
        # The retired communicator is idle across the commit; deregister its
        # HIXL segments before the switch frees the registered weights.
        self._close_hixl_communicator()
        super().commit_scale_up(is_existing_worker)

    def receive_expert_mapping(self) -> torch.Tensor:
        mapping = super().receive_expert_mapping()
        self._setup_moe_comm_and_quant_method()
        return mapping

    def _warm_target_groups(self, dp_group, ep_group) -> None:
        # No-op on Ascend: the dp/ep device_group carries no HCCL traffic
        # (DP sync on cpu_group, MoE on MC2, EPLB on gloo), so skip the warm.
        return

    def _can_reuse_fused_moe_kernel(self) -> bool:
        # Reuse requires the nixl_ep all2all backend (CUDA-only); NPU always
        # takes the non-reuse path and re-captures ACL graphs.
        return False

    def warmup_new_worker(self) -> None:
        # Upstream warms local kernels here (and re-captures graphs on the
        # kernel-reuse path); Ascend skips the kernel warmup.
        pass

    def warm_and_capture(self) -> None:
        # No need to save/clear/restore the KV-cache block tables like the
        # upstream warm_and_capture: the V2 runner's dummy attention uses
        # all-zero block tables (reserved null block) and PAD_SLOT_ID slot
        # mappings, so real KV-cache blocks are never written during the
        # dummy run.
        runner = self.worker.model_runner
        self._release_cuda_graphs()
        unlock_workspace()
        runner._dummy_run(runner.max_num_tokens, is_profile=True, skip_eplb=True)
        self.worker.compile_or_warm_up_model()
        lock_workspace()

    def _setup_moe_comm_and_quant_method(self) -> None:
        moe_modules = [module for module in self.worker.get_model().modules() if is_moe_layer(module)]
        for module in moe_modules:
            with set_current_vllm_config(self.worker.vllm_config):
                setup_moe_comm_and_quant_method(module)
