import inspect
from types import SimpleNamespace

from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

import vllm_ascend.utils
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
from vllm_ascend.distributed.elastic_ep.elastic_execute import (
    AscendElasticEPScalingExecutor,
)
from vllm_ascend.utils import AscendDeviceType


def test_ascend_executor_overrides_current_upstream_hooks():
    for hook in ("_transfer_non_expert_weights", "_prepare_standby_groups"):
        assert hook in AscendElasticEPScalingExecutor.__dict__
        assert hook in ElasticEPScalingExecutor.__dict__

    assert "_batch_transfer_weights" not in AscendElasticEPScalingExecutor.__dict__


def test_elastic_ep_validation_does_not_require_removed_runner_flag(monkeypatch):
    monkeypatch.setattr(vllm_ascend.utils, "get_ascend_device_type", lambda: AscendDeviceType.A3)

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_elastic_ep=True,
            prefill_context_parallel_size=1,
        )
    )

    AscendConfig._validate_elastic_ep(object.__new__(AscendConfig), config)
    assert not hasattr(config, "use_v2_model_runner")


def test_npu_communicator_uses_current_device_communicator_signature():
    parameters = inspect.signature(NPUCommunicator).parameters

    assert "stateless" not in parameters
    assert "tcp_store_group" in parameters
