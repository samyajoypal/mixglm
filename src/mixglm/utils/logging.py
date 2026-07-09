# src/mixglm/utils/logging.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time
import sys


@dataclass
class Logger:
    """
    Lightweight logger with verbosity control and timestamps.

    Designed to be dependency-free and fast enough for EM loops
    and model selection routines.
    """
    verbose: bool = False
    stream: any = sys.stdout
    prefix: str = "[mixglm]"

    def log(self, msg: str) -> None:
        if not self.verbose:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.stream.write(f"{self.prefix} {ts} | {msg}\n")
        self.stream.flush()

    def section(self, title: str) -> None:
        if not self.verbose:
            return
        self.stream.write("\n")
        self.log("=" * len(title))
        self.log(title)
        self.log("=" * len(title))

    def progress(self, it: int, total: Optional[int] = None, *, every: int = 1, msg: str = "") -> None:
        """
        Print periodic progress updates.

        Example:
            logger.progress(it, total=100, every=10, msg="EM iterations")
        """
        if not self.verbose:
            return
        if it % every != 0:
            return
        if total is not None:
            self.log(f"{msg} {it}/{total}")
        else:
            self.log(f"{msg} {it}")
