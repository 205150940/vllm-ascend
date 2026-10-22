# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""External-LB Elastic EP e2e test: scale dp 4 -> 3 -> 4.

Every DP rank is an independent ``vllm serve`` process. Scale-down retires the
last rank process; scale-up relaunches it with
``VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=1``. Each stage verifies the parallel config
on every rank and runs factual-prompt accuracy checks once.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
import torch
from vllm.utils.network_utils import get_open_ports_list

from tests.e2e.conftest import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-30B-A3B"
NUM_EXPERTS = 128
INITIAL_REDUNDANT_EXPERTS = 128

MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 16
INITIAL_DP_SIZE = 4
SCALE_DOWN_DP_SIZE = 3
MAX_DP_SIZE = 4
TP_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.7
HCCL_BUFFSIZE = 1024
RANK_STOP_SETTLE_SECONDS = 5

_ANSWERS = [
    ("The capital of France is", "Paris"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The first month of the year is", "January"),
]


def _base_args() -> list[str]:
    return [
        "--trust-remote-code",
        "--enable-eplb",
        "--eplb-config.log_balancedness",
        "true",
        "--eplb-config.log_balancedness_interval",
        "10",
        "--eplb-config.use_async",
        "false",
        "--eplb-config.num_redundant_experts",
        str(INITIAL_REDUNDANT_EXPERTS),
        "--enable-elastic-ep",
        "--elastic-ep-max-dp-size",
        str(MAX_DP_SIZE),
        "--enable-expert-parallel",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--enforce-eager",
    ]


def _start_rank(rank: int, dp_size: int, port: int, dp_rpc_port: int, scale_up: bool = False) -> RemoteOpenAIServer:
    visible = [d for d in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if d]
    start = rank * TP_SIZE
    devices = visible[start : start + TP_SIZE] or [str(i) for i in range(start, start + TP_SIZE)]
    env = {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "BENCHMARK_HOME": "./benchmark",
        "HCCL_BUFFSIZE": str(HCCL_BUFFSIZE),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(devices),
    }
    if os.environ.get("VLLM_USE_MODELSCOPE", "").lower() not in ("", "0", "false"):
        env["VLLM_USE_MODELSCOPE"] = "true"
    if scale_up:
        env["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "1"

    args = [
        *_base_args(),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--data-parallel-size",
        str(dp_size),
        "--data-parallel-rank",
        str(rank),
        "--data-parallel-size-local",
        "1",
        "--data-parallel-address",
        "127.0.0.1",
        "--data-parallel-rpc-port",
        str(dp_rpc_port),
        "--tensor-parallel-size",
        str(TP_SIZE),
    ]
    server = RemoteOpenAIServer(
        MODEL,
        args,
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env,
        auto_port=False,
        max_wait_seconds=1800,
    )
    print(f"[external-lb] rank {rank} started on port {port}")
    return server


def _start_ranks(ranks, dp_size: int, ports: list[int], dp_rpc_port: int) -> dict[int, RemoteOpenAIServer]:
    servers: dict[int, RemoteOpenAIServer] = {}
    errors: list[Exception] = []

    def start(rank: int) -> None:
        try:
            servers[rank] = _start_rank(rank, dp_size, ports[rank], dp_rpc_port)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=len(ranks)) as executor:
        list(executor.map(start, ranks))
    if errors:
        _stop_all(servers)
        raise errors[0]
    return servers


def _stop(server: RemoteOpenAIServer) -> None:
    server.__exit__(None, None, None)


def _stop_all(servers: dict[int, RemoteOpenAIServer]) -> None:
    while servers:
        _, server = servers.popitem()
        try:
            _stop(server)
        except Exception as exc:  # noqa: BLE001
            print(f"[external-lb] error stopping server: {exc}")


def _scale(
    servers: dict[int, RemoteOpenAIServer],
    new_dp_size: int,
    dp_rpc_port: int,
    port: int | None = None,
) -> None:
    """POST the scale request to every rank; optionally launch the new rank."""
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = [
            executor.submit(
                requests.post,
                server.url_for("scale_elastic_ep"),
                json={"new_data_parallel_size": new_dp_size},
                timeout=1000,
            )
            for server in servers.values()
        ]
        if port is not None:
            while not all(f.running() for f in futures):
                if any(f.done() for f in futures):
                    pytest.fail("Scale request finished before the new rank was launched")
                time.sleep(0.1)
            servers[new_dp_size - 1] = _start_rank(
                new_dp_size - 1,
                new_dp_size,
                port,
                dp_rpc_port,
                scale_up=True,
            )

        for future in futures:
            response = future.result(timeout=1000)
            assert response.status_code == 200, response.text


def _redundant_experts(cur_dp: int, new_dp: int, cur_redundant: int) -> int:
    return (NUM_EXPERTS + cur_redundant) * new_dp // cur_dp - NUM_EXPERTS


def _check_stage(servers: dict[int, RemoteOpenAIServer], dp_size: int, redundant: int, stage: str) -> None:
    assert sorted(servers) == list(range(dp_size))
    for server in servers.values():
        config = requests.get(server.url_for("server_info"), params={"config_format": "json"})
        config.raise_for_status()
        parallel = config.json()["vllm_config"]["parallel_config"]
        assert parallel["data_parallel_size"] == dp_size
        assert parallel["eplb_config"]["num_redundant_experts"] == redundant

    client = servers[0].get_client()
    for prompt, answer in _ANSWERS:
        text = client.completions.create(model=MODEL, prompt=prompt, max_tokens=16, temperature=0.0).choices[0].text
        print(f"[{stage}][accuracy] {prompt!r}: {text!r}")
        assert answer.lower() in text.lower(), f"[{stage}] expected {answer!r}, got {text!r}"


def _has_npus() -> bool:
    try:
        return torch.npu.is_available() and torch.npu.device_count() >= MAX_DP_SIZE
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_npus(), reason="Requires at least 4 NPUs for external LB Elastic EP 4 -> 3 -> 4")
def test_elastic_ep_external_lb_scaling_qwen3_30b() -> None:
    dp_rpc_port, *api_ports = get_open_ports_list(MAX_DP_SIZE + 2)
    servers = _start_ranks(range(INITIAL_DP_SIZE), INITIAL_DP_SIZE, api_ports, dp_rpc_port)
    redundant = INITIAL_REDUNDANT_EXPERTS
    try:
        _check_stage(servers, INITIAL_DP_SIZE, redundant, "Initial")

        # Scale down 4 -> 3, then retire the last rank process.
        _scale(servers, SCALE_DOWN_DP_SIZE, dp_rpc_port)
        _stop(servers.pop(SCALE_DOWN_DP_SIZE))
        time.sleep(RANK_STOP_SETTLE_SECONDS)
        redundant = _redundant_experts(INITIAL_DP_SIZE, SCALE_DOWN_DP_SIZE, redundant)
        _check_stage(servers, SCALE_DOWN_DP_SIZE, redundant, "After scale down")

        # Scale up 3 -> 4, relaunching rank 3 once the existing ranks prepare.
        _scale(servers, MAX_DP_SIZE, dp_rpc_port, port=api_ports[MAX_DP_SIZE])
        redundant = _redundant_experts(SCALE_DOWN_DP_SIZE, MAX_DP_SIZE, redundant)
        _check_stage(servers, MAX_DP_SIZE, redundant, "After scale up")
    finally:
        _stop_all(servers)
