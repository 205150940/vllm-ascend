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
"""
End-to-end Elastic EP scaling tests for vllm-ascend.

Launches a vLLM serve instance with Elastic EP enabled and performs a
scale-down (dp=4 -> dp=3) followed by a scale-up (dp=3 -> dp=4), validating
that inference quality is preserved with factual-prompt accuracy checks
after each scaling stage.

Scaling can additionally run under continuous request traffic
(``_scale_with_traffic``), which asserts that scaling only produces 200/503
responses, that the commit is observed via the ``/is_scaling_elastic_ep``
probe, and that requests keep completing during the preparation phase.

External load balancing (``--data-parallel-rank``) is covered separately:
each DP rank runs as an independent ``vllm serve`` process and a new rank is
launched during scale-up, mirroring how an external load balancer orchestrates
Elastic EP workers.
"""

import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest
import regex as re
import requests
import torch
from vllm.utils.network_utils import get_open_port, get_open_ports_list

from tests.e2e.conftest import RemoteOpenAIServer

# ---------------------------------------------------------------------------
# Server / model constants
# ---------------------------------------------------------------------------

QWEN3_30B_A3B_MODEL = "Qwen/Qwen3-30B-A3B"
"""HuggingFace / ModelScope identifier of the MoE model used for testing."""

MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 16

QWEN3_30B_A3B_NUM_EXPERTS = 128
QWEN3_30B_NUM_REDUNDANT_EXPERTS = 128

# External LB scenario: start four independent DP ranks, retire one for a
# scale-down, then relaunch it for a scale-up, mirroring an external
# orchestrator that removes and re-adds rank processes.
EXTERNAL_LB_INITIAL_DP_SIZE = 4
EXTERNAL_LB_SCALE_DOWN_DP_SIZE = 3
EXTERNAL_LB_MAX_DP_SIZE = 4
EXTERNAL_LB_TP_SIZE = 1
# Scale transitions keep the old and new stateless HCCL groups alive briefly,
# so leave more NPU headroom than a steady-state server and cap the CCL buffer.
EXTERNAL_LB_GPU_MEMORY_UTILIZATION = 0.7
EXTERNAL_LB_HCCL_BUFFSIZE = 1024
EXTERNAL_LB_RANK_STOP_SETTLE_SECONDS = 5

# How long (seconds) to wait after a scale request before evaluating,
# giving the Elastic EP state machine time to finish reconfiguration.
_SCALE_DELAY_SECONDS = 30

# Post-scaling accuracy check: after each scaling stage, the server must
# answer these factual prompts correctly. Each expected answer is a single
# word, matched case-insensitively on word boundaries anywhere in the
# completion, so harmless phrasing variations do not fail the check.
_ANSWER_CASES = [
    ("The capital of France is", ("Paris",)),
    ("The largest planet in our solar system is", ("Jupiter",)),
    ("The first month of the year is", ("January",)),
]
_ANSWER_MAX_TOKENS = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_ray_between_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force-stop any lingering Ray processes between tests."""
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)

    # monkeypatch.setenv so the vars are visible to `ray start --head` (Ray
    # workers inherit the daemon env) and are restored after the test.
    for key, value in _make_env_dict().items():
        monkeypatch.setenv(key, value)

    subprocess.run(["ray", "start", "--head"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


def _send_scale_command(server, new_dp_size: int) -> bool:
    """POST a scale request to the server's Elastic EP endpoint.

    Returns ``True`` on HTTP 200, ``False`` otherwise.
    """
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=1000)
        code = response.status_code
        if code != 200:
            print(f"[scale] HTTP {code}: {response.text}")
        return code == 200
    except requests.exceptions.RequestException as exc:
        print(f"[scale] Request failed: {exc}")
        return False


def _post_scale_request(server: RemoteOpenAIServer, new_dp_size: int) -> requests.Response:
    """POST a scale request and return the raw response for external LB tests."""
    return requests.post(
        server.url_for("scale_elastic_ep"),
        json={"new_data_parallel_size": new_dp_size},
        timeout=1000,
    )


def _wait_for_scale_requests_to_start(scale_requests: list[Future], timeout: float = 30) -> None:
    """Wait until every scale request is in flight before launching a new rank.

    External LB scale-up requires the existing ranks to enter the preparation
    phase and open their handshake listener before the new rank starts.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for request in scale_requests:
            if request.done():
                if request.exception() is not None:
                    pytest.fail(f"Scale request failed before the new rank was launched: {request.exception()}")
                response = request.result()
                pytest.fail(
                    f"Scale request finished before the new rank was launched: {response.status_code} {response.text}"
                )

        if all(request.running() for request in scale_requests):
            return
        time.sleep(0.1)

    pytest.fail("Timed out waiting for all scale requests to start")


def _get_parallel_config(server: RemoteOpenAIServer) -> dict:
    response = requests.get(server.url_for("server_info"), params={"config_format": "json"})
    response.raise_for_status()
    return response.json()["vllm_config"]["parallel_config"]


def _traffic_loop(
    server: RemoteOpenAIServer,
    dp_rank: int | None,
    ready: threading.Barrier,
    stop: threading.Event,
    finished: threading.Event,
    model_name: str,
    is_probe: bool = False,
) -> list[tuple[float, float, int | None]]:
    """Send requests continuously until the scale finishes.

    The probe (``is_probe=True``) polls ``/is_scaling_elastic_ep`` instead of
    sending inference requests, so it can observe the exact 503 -> 200
    transition caused by the scale commit.
    """
    url = server.url_for("is_scaling_elastic_ep" if is_probe else "v1/completions")
    payload = {"model": model_name, "prompt": "Hello", "max_tokens": 4}
    headers = None if dp_rank is None else {"X-data-parallel-rank": str(dp_rank)}
    request_payload = None if is_probe else payload
    responses = []
    is_ready = False
    while not stop.is_set():
        request_start = time.perf_counter()
        try:
            response = requests.post(url, json=request_payload, headers=headers, timeout=120)
            status_code = response.status_code
        except requests.exceptions.RequestException:
            status_code = None
        responses.append((request_start, time.perf_counter(), status_code))
        if status_code == 200:
            if not is_ready:
                ready.wait(timeout=120)
                is_ready = True
            if finished.is_set():
                return responses
        time.sleep(0.05)
    return responses


def _downtime(responses: list[tuple[float, float, int | None]]) -> float:
    """Compute the downtime window from the probe's responses.

    Downtime is the time between the first 503 (scale commit) and the first
    200 after it (service recovered). Returns 0 if no 503 was observed.
    """
    rejected = [end for _, end, status in responses if status == 503]
    if not rejected:
        return 0
    recovered = next(
        (end for _, end, status in responses if status == 200 and end > rejected[-1]),
        None,
    )
    assert recovered is not None, "service did not recover after scale commit"
    return recovered - rejected[0]


def _scale_with_traffic(
    server: RemoteOpenAIServer,
    source_dp_size: int,
    new_dp_size: int,
    model_name: str,
    traffic_mode: str,
) -> None:
    """Scale the server while continuously sending requests.

    Verifies that scaling under traffic only produces 200/503 statuses, that
    the probe observes the commit (503), and that requests completed before
    the scale. Prints the scale duration and the downtime window.
    """
    traffic_clients: list[int | None] = []
    if traffic_mode == "light":
        traffic_clients = [0]
    elif traffic_mode == "heavy":
        traffic_clients = [None] * source_dp_size
    clients = [(None, True)] + [(rank, False) for rank in traffic_clients]
    ready = threading.Barrier(len(clients) + 1)
    stop = threading.Event()
    finished = threading.Event()

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [
            executor.submit(
                _traffic_loop,
                server,
                rank,
                ready,
                stop,
                finished,
                model_name,
                is_probe,
            )
            for rank, is_probe in clients
        ]
        try:
            ready.wait(timeout=120)
            start_time = time.perf_counter()
            assert _send_scale_command(server, new_dp_size)
            scale_seconds = time.perf_counter() - start_time
            finished.set()
            probe_result, *results = [future.result(timeout=120) for future in futures]
        finally:
            stop.set()

    bad_statuses = {
        status for responses in [probe_result, *results] for _, _, status in responses if status not in (200, 503)
    }
    assert not bad_statuses, f"traffic got unexpected statuses {bad_statuses}"
    probe_503 = [start for start, _, status in probe_result if status == 503]
    assert probe_503, "Scaling probe did not observe commit"
    assert not results or any(
        status == 200 and start_time <= request_start and request_end < probe_503[0]
        for responses in results
        for request_start, request_end, status in responses
    ), "No request completed successfully during preparation"

    print(
        f"[Elastic EP timing][{source_dp_size}->{new_dp_size}]"
        f"[traffic={traffic_mode}] "
        f"scale_seconds={scale_seconds:.3f} "
        f"downtime_seconds={_downtime(probe_result):.3f}"
    )


def _assert_correct_answers(server, model_name: str, stage: str) -> None:
    """Send factual prompts to the server and assert the answer appears.

    Greedy decoding; case-insensitive whole-word match, not an exact-match
    golden, so the check is independent of environment numerics.
    """
    client = server.get_client()
    for prompt, answers in _ANSWER_CASES:
        resp = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=_ANSWER_MAX_TOKENS,
            temperature=0.0,
        )
        completion = resp.choices[0].text
        matched = any(re.search(rf"\b{re.escape(answer)}\b", completion, re.IGNORECASE) for answer in answers)
        print(f"[{stage}][accuracy] prompt {prompt!r}: {completion!r}")
        assert matched, f"[{stage}] answered {prompt!r} incorrectly: expected one of {answers!r}, got {completion!r}"


def _make_env_dict() -> dict[str, str]:
    """Return the common environment-variable overrides for the server."""
    env = {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "BENCHMARK_HOME": "./benchmark",
        "HCCL_BUFFSIZE": "2048",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    if os.environ.get("VLLM_USE_MODELSCOPE", "").lower() not in ("", "0", "false"):
        env["VLLM_USE_MODELSCOPE"] = "true"
    return env


def _device_ids_for_rank(rank: int, tp_size: int = EXTERNAL_LB_TP_SIZE) -> str:
    """Select the NPUs visible to one externally managed DP rank.

    CI usually exports the allocated cards through
    ``ASCEND_RT_VISIBLE_DEVICES``; keep that mapping so rank ``r`` gets
    ``[r * tp_size, (r + 1) * tp_size)`` of the allocated cards.
    """
    visible_devices = [device for device in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if device]
    start = rank * tp_size
    selected = visible_devices[start : start + tp_size]
    if len(selected) == tp_size:
        return ",".join(selected)
    return ",".join(str(device_id) for device_id in range(start, start + tp_size))


def _build_external_lb_base_args() -> list[str]:
    """Build the arguments shared by every externally managed DP rank."""
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
        str(QWEN3_30B_NUM_REDUNDANT_EXPERTS),
        "--enable-elastic-ep",
        "--elastic-ep-max-dp-size",
        str(EXTERNAL_LB_MAX_DP_SIZE),
        "--enable-expert-parallel",
        "--gpu-memory-utilization",
        str(EXTERNAL_LB_GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--enforce-eager",
    ]


class ExternalLBServerManager:
    """Manage independent ``vllm serve`` processes for external LB tests."""

    def __init__(
        self,
        model_name: str,
        dp_size: int,
        base_server_args: list[str],
        tp_size: int = EXTERNAL_LB_TP_SIZE,
    ) -> None:
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.base_server_args = base_server_args
        self.dp_address = "127.0.0.1"
        # Reserve the coordination port and every rank's API port up front so
        # the coordination port cannot collide with an API server port. The
        # extra API port covers relaunching a rank after a scale-down.
        max_ranks = max(dp_size, EXTERNAL_LB_MAX_DP_SIZE)
        ports = get_open_ports_list(max_ranks + 2)
        self.dp_rpc_port = ports[0]
        self.api_ports = ports[1:]
        self._api_port_lock = threading.Lock()
        self._next_api_port = 0
        self.servers: dict[int, RemoteOpenAIServer] = {}
        self.server_threads: list[threading.Thread] = []

    def start_rank(
        self,
        rank: int,
        dp_size: int | None = None,
        elastic_ep_scale_up_launch: bool = False,
    ) -> RemoteOpenAIServer:
        """Start one externally managed DP rank on its own NPU and port."""
        dp_size = self.dp_size if dp_size is None else dp_size
        with self._api_port_lock:
            port = self.api_ports[self._next_api_port]
            self._next_api_port += 1
        server_args = [
            *self.base_server_args,
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
            self.dp_address,
            "--data-parallel-rpc-port",
            str(self.dp_rpc_port),
            "--tensor-parallel-size",
            str(self.tp_size),
        ]

        env_dict = _make_env_dict()
        env_dict["VLLM_SERVER_DEV_MODE"] = "1"
        env_dict["ASCEND_RT_VISIBLE_DEVICES"] = _device_ids_for_rank(rank, self.tp_size)
        env_dict["HCCL_BUFFSIZE"] = str(EXTERNAL_LB_HCCL_BUFFSIZE)
        if elastic_ep_scale_up_launch:
            env_dict["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "1"

        server = RemoteOpenAIServer(
            self.model_name,
            server_args,
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env_dict,
            auto_port=False,
            max_wait_seconds=1800,
        )
        print(f"[external-lb] rank {rank} started on port {port}")
        self.servers[rank] = server
        return server

    def stop_rank(self, rank: int) -> None:
        """Terminate a retired rank so the orchestrator can remove it."""
        server = self.servers.pop(rank, None)
        if server is None:
            return
        print(f"[external-lb] stopping rank {rank}")
        server.__exit__(None, None, None)
        # Give the driver time to tear down HCCL state and release NPU memory
        # before this rank is relaunched during scale-up.
        time.sleep(EXTERNAL_LB_RANK_STOP_SETTLE_SECONDS)

    def __enter__(self) -> dict[int, RemoteOpenAIServer]:
        """Start the initial DP ranks in parallel."""
        start_errors: list[Exception] = []

        def start_server(rank: int) -> None:
            try:
                self.start_rank(rank)
            except Exception as exc:
                print(f"[external-lb] failed to start rank {rank}: {exc}")
                start_errors.append(exc)

        for rank in range(self.dp_size):
            thread = threading.Thread(target=start_server, args=(rank,))
            thread.start()
            self.server_threads.append(thread)

        for thread in self.server_threads:
            thread.join()

        if start_errors:
            self.__exit__(None, None, None)
            raise RuntimeError("Failed to start external LB servers") from start_errors[0]

        if len(self.servers) != self.dp_size:
            self.__exit__(None, None, None)
            raise RuntimeError(f"Expected {self.dp_size} external LB servers, got {len(self.servers)}")

        return self.servers

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Terminate all rank processes."""
        servers = list(self.servers.values())
        self.servers.clear()
        for server in servers:
            try:
                server.__exit__(exc_type, exc_value, traceback)
            except Exception as exc:
                print(f"[external-lb] error stopping server: {exc}")


def _scale_external_lb(
    server_manager: ExternalLBServerManager,
    new_dp_size: int,
    *,
    launch_new_rank: bool = False,
) -> None:
    """Drive one external LB scaling step across all existing API ranks."""
    existing_servers = list(server_manager.servers.values())
    with ThreadPoolExecutor(max_workers=len(existing_servers)) as executor:
        scale_requests = [executor.submit(_post_scale_request, server, new_dp_size) for server in existing_servers]
        _wait_for_scale_requests_to_start(scale_requests)

        if launch_new_rank:
            server_manager.start_rank(
                rank=new_dp_size - 1,
                dp_size=new_dp_size,
                elastic_ep_scale_up_launch=True,
            )

        for request in scale_requests:
            response = request.result(timeout=1000)
            assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Configuration dataclasses and test runner
# ---------------------------------------------------------------------------


@dataclass
class ScaleSequence:
    """Defines a sequence of scaling operations."""

    name: str
    steps: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class ElasticEPTestConfig:
    """Configuration for an Elastic EP test."""

    name: str
    data_parallel_size: int
    data_parallel_size_local: int
    tensor_parallel_size: int
    gpu_memory_utilization: float = 0.9
    num_redundant_experts: int = 0
    scale_sequence: ScaleSequence = field(
        default_factory=lambda: ScaleSequence(
            name="4_to_3_to_4",
            steps=[
                (3, "Scale down (dp=4 -> dp=3)"),
                (4, "Scale up (dp=3 -> dp=4)"),
            ],
        )
    )


CONFIG_QWEN3_30B_DEFAULT = ElasticEPTestConfig(
    name="Qwen3-30B-A3B, Default Graph",
    data_parallel_size=4,
    data_parallel_size_local=4,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    num_redundant_experts=QWEN3_30B_NUM_REDUNDANT_EXPERTS,
)


def _build_vllm_args(config: ElasticEPTestConfig) -> list[str]:
    """Build vLLM server arguments from configuration."""
    args = [
        "--host",
        "0.0.0.0",
        "--port",
        str(get_open_port()),
        "--trust-remote-code",
        "--data-parallel-size",
        str(config.data_parallel_size),
        "--data-parallel-size-local",
        str(config.data_parallel_size_local),
        "--data-parallel-backend",
        "ray",
        "--enable-eplb",
        "--eplb-config.log_balancedness",
        "true",
        "--eplb-config.log_balancedness_interval",
        "10",
        "--eplb-config.use_async",
        "false",
        "--enable-elastic-ep",
        "--enable-expert-parallel",
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
    ]

    args.extend(["--eplb-config.num_redundant_experts", str(config.num_redundant_experts)])

    return args


def _run_elastic_ep_test(
    config: ElasticEPTestConfig,
    model_name: str,
    traffic_mode: str | None = None,
) -> None:
    """Run the scaling sequence with factual-prompt accuracy checks.

    Args:
        config: The Elastic EP test configuration to use.
        model_name: Identifier of the model to serve.
        traffic_mode: When set ("none"/"light"/"heavy"), each scaling step is
            executed under continuous request traffic via
            ``_scale_with_traffic`` (measuring scale duration / downtime).
            When ``None``, each step uses the plain send-command-and-sleep
            sequence.
    """
    vllm_serve_args = _build_vllm_args(config)
    env_dict = _make_env_dict()

    # Extract port from args (last port specification)
    port_index = vllm_serve_args.index("--port") + 1
    server_port = int(vllm_serve_args[port_index])

    with RemoteOpenAIServer(
        model_name,
        vllm_serve_args,
        server_host="127.0.0.1",
        server_port=server_port,
        env_dict=env_dict,
        auto_port=False,
        max_wait_seconds=1800,
    ) as server:
        print(f"Server started on port {server.port}")

        # Run initial baseline accuracy check
        initial_stage = f"Initial (dp={config.data_parallel_size})"
        _assert_correct_answers(server, model_name, initial_stage)

        # Run scaling steps
        current_dp_size = config.data_parallel_size
        for new_dp_size, stage_description in config.scale_sequence.steps:
            if traffic_mode is None:
                assert _send_scale_command(server, new_dp_size), f"{stage_description} failed"
                time.sleep(_SCALE_DELAY_SECONDS)
            else:
                _scale_with_traffic(server, current_dp_size, new_dp_size, model_name, traffic_mode)
            current_dp_size = new_dp_size
            _assert_correct_answers(server, model_name, stage_description)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def has_npu_elastic_ep_capability() -> bool:
    """Require at least 4 visible NPUs for dp=4 Elastic EP scaling tests."""
    if not torch.npu.is_available():
        return False
    try:
        return torch.npu.device_count() >= 4
    except Exception:
        return False


def has_npu_external_lb_capability() -> bool:
    """Require at least 4 visible NPUs for external LB 4 -> 3 -> 4 tests."""
    if not torch.npu.is_available():
        return False
    try:
        return torch.npu.device_count() >= EXTERNAL_LB_MAX_DP_SIZE
    except Exception:
        return False


def _expected_num_redundant_experts(cur_dp_size: int, new_dp_size: int, cur_redundant_experts: int) -> int:
    """Mirror ExternalElasticEPScaleCoordinator._calculate_num_redundant_experts."""
    return (QWEN3_30B_A3B_NUM_EXPERTS + cur_redundant_experts) * new_dp_size // cur_dp_size - QWEN3_30B_A3B_NUM_EXPERTS


def _assert_external_lb_state(
    server_manager: ExternalLBServerManager,
    expected_dp_size: int,
    expected_redundant_experts: int,
    stage: str,
) -> None:
    assert sorted(server_manager.servers) == list(range(expected_dp_size))
    for rank in range(expected_dp_size):
        server = server_manager.servers[rank]
        parallel_config = _get_parallel_config(server)
        assert parallel_config["data_parallel_size"] == expected_dp_size
        assert parallel_config["eplb_config"]["num_redundant_experts"] == expected_redundant_experts
        _assert_correct_answers(
            server,
            QWEN3_30B_A3B_MODEL,
            f"{stage} (external lb, dp={expected_dp_size}, rank={rank})",
        )


@pytest.mark.skipif(
    not has_npu_elastic_ep_capability(),
    reason="Requires at least 4 NPUs for dp=4 Elastic EP scaling tests",
)
def test_elastic_ep_scaling_qwen3_30b() -> None:
    """Scale dp 4 -> 3 -> 4 (tp=1) with Default Graph."""
    _run_elastic_ep_test(CONFIG_QWEN3_30B_DEFAULT, QWEN3_30B_A3B_MODEL)


@pytest.mark.parametrize(
    "traffic_mode",
    [
        pytest.param("none", id="none"),
        pytest.param("light", id="light"),
        pytest.param("heavy", id="heavy"),
    ],
)
@pytest.mark.skipif(
    not has_npu_elastic_ep_capability(),
    reason="Requires at least 4 NPUs for dp=4 Elastic EP scaling tests",
)
def test_elastic_ep_scaling_qwen3_30b_with_traffic(traffic_mode: str) -> None:
    """Scale dp 4 -> 3 -> 4 under continuous request traffic.

    "none" runs the probe only (measures scale duration / downtime),
    "light" adds a single traffic client pinned to dp rank 0, "heavy" adds
    as many unbound traffic clients as the source DP size (requests are
    routed freely, so no client pins a rank that scale-down may retire).
    Factual-prompt accuracy is still checked after every scaling stage.
    """
    _run_elastic_ep_test(CONFIG_QWEN3_30B_DEFAULT, QWEN3_30B_A3B_MODEL, traffic_mode)


@pytest.mark.skipif(
    not has_npu_external_lb_capability(),
    reason="Requires at least 4 NPUs for external LB Elastic EP 4 -> 3 -> 4",
)
def test_elastic_ep_external_lb_scaling_qwen3_30b() -> None:
    """Scale external LB dp 4 -> 3 -> 4 across independent rank processes.

    Each rank owns one NPU and exposes its own API server. For scale-down all
    four ranks receive the scale request and the retired rank process is then
    stopped, like an orchestrator removing it from the load balancer. For
    scale-up the remaining three ranks reconfigure first, then rank 3 starts
    with ``VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=1``. Every stage checks the new DP
    size, recalculated EPLB redundancy, and factual-prompt accuracy.
    """
    server_manager = ExternalLBServerManager(
        QWEN3_30B_A3B_MODEL,
        EXTERNAL_LB_INITIAL_DP_SIZE,
        _build_external_lb_base_args(),
    )
    current_dp_size = EXTERNAL_LB_INITIAL_DP_SIZE
    current_redundant_experts = QWEN3_30B_NUM_REDUNDANT_EXPERTS

    with server_manager:
        _assert_external_lb_state(
            server_manager,
            current_dp_size,
            current_redundant_experts,
            "Initial",
        )

        # Scale down 4 -> 3: every rank participates in the commit, then the
        # external orchestrator terminates the retired rank process.
        _scale_external_lb(server_manager, EXTERNAL_LB_SCALE_DOWN_DP_SIZE)
        server_manager.stop_rank(EXTERNAL_LB_INITIAL_DP_SIZE - 1)
        current_redundant_experts = _expected_num_redundant_experts(
            current_dp_size,
            EXTERNAL_LB_SCALE_DOWN_DP_SIZE,
            current_redundant_experts,
        )
        current_dp_size = EXTERNAL_LB_SCALE_DOWN_DP_SIZE
        _assert_external_lb_state(
            server_manager,
            current_dp_size,
            current_redundant_experts,
            "After scale down (dp=4 -> dp=3)",
        )

        # Scale up 3 -> 4: start rank 3 once all existing ranks are preparing.
        _scale_external_lb(server_manager, EXTERNAL_LB_MAX_DP_SIZE, launch_new_rank=True)
        current_redundant_experts = _expected_num_redundant_experts(
            current_dp_size,
            EXTERNAL_LB_MAX_DP_SIZE,
            current_redundant_experts,
        )
        current_dp_size = EXTERNAL_LB_MAX_DP_SIZE
        _assert_external_lb_state(
            server_manager,
            current_dp_size,
            current_redundant_experts,
            "After scale up (dp=3 -> dp=4)",
        )
