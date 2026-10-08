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
from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups,
    get_standby_ep_group,
    get_standby_eplb_group,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.platforms import current_platform
from vllm.utils import is_moe_layer
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

from vllm_ascend.ascend_forward_context import use_cann_megamoe
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
from vllm_ascend.distributed.parallel_state import (
    GroupCoordinator,
    _replace_ascend_active_groups,
    get_mc2_group,
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
        batch_transfer_weights(
            model=model,
            is_sender=is_sender,
            peer_rank=peer_rank,
            dp_group=dp_group,
            expert_weights=expert_weights,
        )

    def prepare_reconfiguration(self, reconfig_request: ReconfigureDistributedRequest, use_all2all: bool) -> None:
        # Ascend variant of the upstream preparation: creates the MC2
        # standby group between the upstream groups and transfer_weights.
        # The order is load-bearing — stateless rendezvous completes only
        # when all members join, and new workers boot their groups as
        # world -> dp -> ep -> eplb -> mc2; any other order deadlocks
        # scale-up.
        self._wait_for_group_cleanup()
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        old_dp_size = get_dp_group().world_size
        parallel_config = self.worker.vllm_config.parallel_config
        world_size = parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        create_standby_groups(
            new_dp_size=new_dp_size,
            new_world_size_across_dp=new_world_size_across_dp,
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
            use_all2all=use_all2all,
            enable_eplb=parallel_config.enable_eplb,
        )
        if not self._can_reuse_fused_moe_kernel():
            create_ascend_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=new_world_size_across_dp,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                coord_store_port=reconfig_request.coord_store_port,
            )
        # else (graph reuse): keep the current MC2 group — its HCCL comm
        # name and symm buffer are baked into the captured graphs.
        # Non-reuse only: stage the standby quant methods (mirrors upstream).
        standby_ep_group = get_standby_ep_group()
        assert standby_ep_group is not None
        all2all_manager = get_ep_all2all_manager(standby_ep_group)
        all2all_manager.stage_ep_size()
        if not self._can_reuse_fused_moe_kernel():
            self.stage_standby_moe_quant_methods(all2all_manager)
        self._prepare_eplb_communicator(get_standby_eplb_group())
        if new_dp_size > old_dp_size:
            self.transfer_weights(old_dp_size, new_dp_size)

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

    def switch_and_remove(self) -> None:
        super().switch_and_remove()
        retired_mc2 = _replace_ascend_active_groups(mc2=None)
        if retired_mc2 is not None:
            retired_mc2.destroy()

    def switch_and_prepare(self) -> tuple[GroupCoordinator | None, ...]:
        reuse = self._can_reuse_fused_moe_kernel()
        # Upstream's reuse branch skips _release_cuda_graphs.
        retired_groups = super().switch_and_prepare()
        self.worker.model_runner.dp_size = self.worker.parallel_config.data_parallel_size
        self.worker.model_runner.dp_rank = self.worker.parallel_config.data_parallel_rank
        if reuse:
            # Graphs bake in the old MC2 comm name and symm buffer, so keep
            # the group and comm methods; membership changed via mask writes.
            moe_modules = [module for module in self.worker.model_runner.model.modules() if is_moe_layer(module)]
            for module in moe_modules:
                module.moe_config.tp_group = get_tp_group()
                module.moe_config.dp_group = get_dp_group()
                module.moe_config.ep_group = get_ep_group()
                # Stale on purpose: graphs reference the pre-switch group.
                # module.moe_config.mc2_group NOT refreshed.
            return retired_groups
        retired_mc2 = _replace_ascend_active_groups(**pop_ascend_standby_groups())
        moe_modules = [module for module in self.worker.model_runner.model.modules() if is_moe_layer(module)]
        for module in moe_modules:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()
        self._setup_moe_comm_and_quant_method()
        if retired_mc2 is not None:
            # Destroyed by the executor's async cleanup thread.
            return (*retired_groups, retired_mc2)
        return retired_groups

    def receive_expert_mapping(self) -> torch.Tensor:
        mapping = super().receive_expert_mapping()
        self._setup_moe_comm_and_quant_method()
        return mapping

    def _warm_target_groups(self, dp_group, ep_group) -> None:
        # No-op on Ascend: the dp/ep device_group carries no HCCL traffic
        # (DP sync on cpu_group, MoE on MC2, EPLB on gloo), so skip the warm.
        return

    def _can_reuse_fused_moe_kernel(self) -> bool:
        # Graph reuse: MegaMoe only, scale-down only — the symm buffer's
        # rank mask lets captured graphs skip removed ranks; scale-up needs
        # a latecomer connect that CANN does not expose.
        if not use_cann_megamoe(self.worker.vllm_config):
            return False
        if not get_ep_all2all_manager().uses_mega_moe:
            # Mask buffer binds lazily at the first MegaMoe forward.
            return False
        reconfig_request = self.reconfig_request
        if reconfig_request is None:
            return False
        return reconfig_request.new_data_parallel_size < get_dp_group().world_size

    def _mask_removed_ep_ranks(self, new_dp_size: int) -> None:
        """Mask the removed EP ranks on the MegaMoe buffer (in place).

        Runs after the EPLB reshuffle moved experts off them and before
        switch_and_prepare updates the parallel config (the old EP size is
        derived from the still-active DP group). Graphs re-read the mask
        on replay.
        """
        tp_size = self.worker.vllm_config.parallel_config.tensor_parallel_size
        old_ep_size = get_dp_group().world_size * tp_size
        new_ep_size = new_dp_size * tp_size
        manager = get_ep_all2all_manager()
        assert manager.uses_mega_moe, "Scale-down graph reuse requires the MegaMoe mask buffer bound at warmup."
        for ep_rank in range(new_ep_size, old_ep_size):
            manager.update_mask(ep_rank, masked=True)
        torch.npu.synchronize()

    def commit_scale_down(self, new_dp_size: int, removing: bool) -> None:
        if not removing and self._can_reuse_fused_moe_kernel():
            # Graph reuse: move experts off the removed ranks, mask them,
            # switch control-plane groups only, skip re-capture.
            self.perform_scale_down_eplb_reshuffle(new_dp_size)
            self._mask_removed_ep_ranks(new_dp_size)
            retired_groups = self.switch_and_prepare()
            self._start_group_cleanup(retired_groups)
            # Publish mask/routing updates before the next graph replay.
            torch.npu.synchronize()
            return
        super().commit_scale_down(new_dp_size, removing)

    def warmup_new_worker(self) -> None:
        # No kernel warmup on Ascend.
        pass

    def warm_and_capture(self) -> None:
        # V2 dummy attention uses null block tables, so real KV blocks are
        # never written — no save/clear/restore like upstream.
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
