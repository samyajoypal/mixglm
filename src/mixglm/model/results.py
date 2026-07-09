# src/mixglm/model/results.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

Array = np.ndarray


@dataclass
class MixtureGLMResult:
    """
    Container for fitted mixture-GLM results.
    """
    converged: bool
    n_iter: int
    loglik: float
    bic: float
    aic: float
    icl: Optional[float]

    pi: Array
    betas: List[Array]
    extras: List[Dict[str, Any]]
    responsibilities: Array

    history: Dict[str, List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": bool(self.converged),
            "n_iter": int(self.n_iter),
            "loglik": float(self.loglik),
            "bic": float(self.bic),
            "aic": float(self.aic),
            "icl": float(self.icl) if self.icl is not None else None,
            "pi": self.pi.tolist(),
            "betas": [b.tolist() for b in self.betas],
            "extras": self.extras,
            "responsibilities": self.responsibilities.tolist(),
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MixtureGLMResult":
        return cls(
            converged=bool(d["converged"]),
            n_iter=int(d["n_iter"]),
            loglik=float(d["loglik"]),
            bic=float(d["bic"]),
            aic=float(d["aic"]),
            icl=float(d["icl"]) if d.get("icl", None) is not None else None,
            pi=np.asarray(d["pi"], dtype=float),
            betas=[np.asarray(b, dtype=float) for b in d["betas"]],
            extras=list(d.get("extras", [])),
            responsibilities=np.asarray(d["responsibilities"], dtype=float),
            history=dict(d.get("history", {"loglik": [], "obj": []})),
        )

    def summary_str(self) -> str:
        K = int(self.pi.size)
        p = int(self.betas[0].size) if self.betas else 0
        lines = []
        lines.append("MixtureGLMResult")
        lines.append(f"  converged: {self.converged}")
        lines.append(f"  n_iter:    {self.n_iter}")
        lines.append(f"  K:         {K}")
        lines.append(f"  p:         {p}")
        lines.append(f"  loglik:    {self.loglik:.6f}")
        lines.append(f"  AIC:       {self.aic:.6f}")
        lines.append(f"  BIC:       {self.bic:.6f}")
        if self.icl is not None:
            lines.append(f"  ICL:       {self.icl:.6f}")
        lines.append(f"  pi:        {np.array2string(self.pi, precision=4)}")
        return "\n".join(lines)
