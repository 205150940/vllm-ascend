# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage timing for the FT scale-down flow.

Splits one scale-down round into timed stages, reports each stage to the log
as it completes, prints a summary table at the end, and dumps the per-stage
breakdown to a CSV file on disk for offline analysis.
"""

import csv
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.logger import logger

# Columns of the dumped CSV table.
CSV_HEADER = ("stage", "depth", "start_offset_ms", "duration_ms", "percent_of_total")

# Indentation per nesting level in the logged summary table.
_LOG_INDENT = "  "

# Column width of the logged summary table.
_STAGE_COL_WIDTH = 44

# Timer of the in-flight scale-down round, if any. Module-level so the
# upstream-facing patch (patch_sentinel_timing) can join the same round
# without threading the timer through upstream vLLM signatures.
_active_timer: "FtScaleDownTimer | None" = None


def get_active_timer() -> "FtScaleDownTimer | None":
    """Return the timer of the in-flight scale-down round, or None."""
    return _active_timer


def set_active_timer(timer: "FtScaleDownTimer | None") -> None:
    """Activate/deactivate scale-down timing for this process."""
    global _active_timer
    _active_timer = timer


class FtScaleDownTimer:
    """Nested stage stopwatch for one scale-down round.

    Stages may nest freely: ``stage()`` blocks maintain a stack, and each
    completed stage records its nesting depth, start offset relative to the
    timer creation, wall duration, and share of the total time.
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._stack: list[str] = []
        # (depth, name, start_offset_ms, duration_ms) in completion order.
        self._rows: list[tuple[int, str, float, float]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named stage; records the row even if the body raises."""
        depth = len(self._stack)
        self._stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stack.pop()
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._rows.append((depth, name, (start - self._t0) * 1000.0, duration_ms))
            logger.info("[FT] scale_down stage '%s' took %.1f ms", name, duration_ms)

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def format_table(self) -> str:
        """Render the stage breakdown as a fixed-width summary table."""
        total_ms = self.total_ms()
        header = f"{'stage':<{_STAGE_COL_WIDTH}} {'depth':>5} {'start_ms':>10} {'dur_ms':>10} {'pct':>6}"
        lines = [header]
        for depth, name, start_ms, duration_ms in self._rows:
            stage_col = _LOG_INDENT * depth + name
            lines.append(
                f"{stage_col:<{_STAGE_COL_WIDTH}} {depth:>5} "
                f"{start_ms:>10.1f} {duration_ms:>10.1f} "
                f"{self._percent(duration_ms, total_ms):>5.1f}%"
            )
        lines.append(f"{'TOTAL':<{_STAGE_COL_WIDTH}} {'':>5} {'':>10} {total_ms:>10.1f} {'100.0%':>6}")
        return "\n".join(lines)

    def dump_to_disk(self, rank: int, file_tag: str = "") -> str | None:
        """Write the stage table as CSV; returns the path or None on failure.

        The file is written to the current working directory as
        ``ft_scale_down_timing_rank<rank>[_<file_tag>].csv``.
        """
        directory = os.getcwd()
        filename = f"ft_scale_down_timing_rank{rank}"
        if file_tag:
            filename = f"{filename}_{file_tag}"
        path = os.path.join(directory, f"{filename}.csv")
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                total_ms = self.total_ms()
                for depth, name, start_ms, duration_ms in self._rows:
                    writer.writerow(
                        (
                            name,
                            depth,
                            f"{start_ms:.3f}",
                            f"{duration_ms:.3f}",
                            f"{self._percent(duration_ms, total_ms):.2f}",
                        )
                    )
        except OSError as exc:
            logger.warning("[FT] Failed to dump scale-down timing table to %s: %s", path, exc)
            return None
        return path

    @staticmethod
    def _percent(duration_ms: float, total_ms: float) -> float:
        if total_ms <= 0.0:
            return 0.0
        return duration_ms / total_ms * 100.0
