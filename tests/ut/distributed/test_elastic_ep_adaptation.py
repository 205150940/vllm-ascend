from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

from vllm_ascend.distributed.elastic_ep.elastic_execute import (
    AscendElasticEPScalingExecutor,
)


def test_ascend_executor_uses_current_upstream_hooks():
    current_hooks = {
        "_transfer_non_expert_weights",
        "_prepare_standby_groups",
        "_get_eplb_expert_load",
        "switch_and_prepare",
        "commit_scale_down",
    }
    obsolete_hooks = {
        "_batch_transfer_weights",
        "_create_backend_standby_groups",
        "_expert_load_view",
        "_on_topology_switched",
        "_after_scale_down_reshuffle",
    }

    for hook in current_hooks:
        assert hook in AscendElasticEPScalingExecutor.__dict__
        assert hook in ElasticEPScalingExecutor.__dict__

    assert obsolete_hooks.isdisjoint(AscendElasticEPScalingExecutor.__dict__)
