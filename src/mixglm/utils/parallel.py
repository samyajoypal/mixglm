# src/mixglm/utils/parallel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Any, Optional
import os


@dataclass
class ParallelConfig:
    """
    Small parallelization config.

    We keep it optional and dependency-light. If joblib is available, we use it.
    Otherwise we fall back to serial execution.
    """
    n_jobs: int = 1          # 1 means serial
    backend: str = "loky"    # joblib backend if available
    prefer: str = "processes"  # "threads" or "processes"

    def effective_jobs(self) -> int:
        if self.n_jobs is None or self.n_jobs == 0:
            return 1
        if self.n_jobs < 0:
            # joblib convention: -1 means all cores
            return max(os.cpu_count() or 1, 1)
        return max(int(self.n_jobs), 1)


def parallel_map(
    func: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    cfg: Optional[ParallelConfig] = None,
) -> List[Any]:
    """
    Apply func to items possibly in parallel.

    If joblib is installed and cfg.n_jobs != 1, uses joblib.
    Otherwise, runs serially.

    Returns list of results in the same order as items.
    """
    cfg = cfg or ParallelConfig(n_jobs=1)
    items = list(items)
    n_jobs = cfg.effective_jobs()

    if n_jobs == 1:
        return [func(x) for x in items]

    try:
        from joblib import Parallel, delayed
    except Exception:
        # fall back to serial if joblib not installed
        return [func(x) for x in items]

    return Parallel(n_jobs=n_jobs, backend=cfg.backend, prefer=cfg.prefer)(
        delayed(func)(x) for x in items
    )
