# [debug-trace] shared probe switch + trace-file logger.
#
# Probes that insert torch.npu.synchronize() (module forward hooks, MC2 op and
# HCCL collective timing) must NEVER run during aclgraph capture at startup:
# synchronize inside a capturing stream is illegal. Capture happens during
# engine init via runner-internal dummy runs, which bypass NPUWorker, so the
# first wave-time NPUWorker.execute_model / execute_dummy_batch call is
# guaranteed to be post-capture — that call arms the probes.

import logging
import os

_ARMED = False

_TRACE_LOG_PATH = os.environ.get("VLLM_TRACE_DEBUG_FILE", "/tmp/vllm_debug_trace.log")
_trace_logger = logging.getLogger("vllm.debug_trace.file")
_trace_logger.setLevel(logging.INFO)
_trace_logger.propagate = False


def arm() -> None:
    global _ARMED
    _ARMED = True


def is_armed() -> bool:
    return _ARMED


def trace(msg: str, *args) -> None:
    """Append to the shared trace file; never raise."""
    try:
        if not _trace_logger.handlers:
            _handler = logging.FileHandler(_TRACE_LOG_PATH)
            _handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s.%(msecs)03d pid=%(process)d %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _trace_logger.addHandler(_handler)
        _trace_logger.info(msg, *args)
    except Exception:
        pass
