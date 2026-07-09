# src/mixglm/utils/standardize.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass
class Standardizer:
    """
    Column-wise standardizer that leaves an intercept column unchanged.

    If intercept_col=0, we assume X[:,0] is intercept and do not standardize it.
    For j>0:
      X_std[:,j] = (X[:,j] - mean[j]) / std[j]
    """
    intercept_col: int = 0
    mean_: Array | None = None
    scale_: Array | None = None
    fitted_: bool = False

    def fit(self, X: Array) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        mean = np.zeros(p, dtype=float)
        scale = np.ones(p, dtype=float)

        for j in range(p):
            if j == self.intercept_col:
                mean[j] = 0.0
                scale[j] = 1.0
                continue
            col = X[:, j]
            mean[j] = float(np.mean(col))
            s = float(np.std(col, ddof=0))
            scale[j] = s if s > 0 else 1.0

        self.mean_ = mean
        self.scale_ = scale
        self.fitted_ = True
        return self

    def transform(self, X: Array) -> Array:
        if not self.fitted_:
            raise RuntimeError("Standardizer must be fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: Array) -> Array:
        return self.fit(X).transform(X)

    def beta_to_original_scale(self, beta_std: Array) -> Array:
        """
        Convert coefficients learned on standardized X into coefficients for raw X.

        eta = X_std @ beta_std = X_raw @ beta_raw

        For j != intercept:
          beta_raw[j] = beta_std[j] / scale[j]
        Intercept:
          beta_raw[intercept] = beta_std[intercept] - sum_{j!=int} beta_std[j] * mean[j] / scale[j]
        """
        if not self.fitted_:
            raise RuntimeError("Standardizer must be fit() before beta_to_original_scale().")
        b = np.asarray(beta_std, dtype=float).copy()
        mean = self.mean_
        scale = self.scale_

        b_raw = np.zeros_like(b)
        for j in range(b.size):
            if j == self.intercept_col:
                continue
            b_raw[j] = b[j] / scale[j]

        ic = self.intercept_col
        adj = 0.0
        for j in range(b.size):
            if j == ic:
                continue
            adj += b[j] * mean[j] / scale[j]
        b_raw[ic] = b[ic] - adj
        return b_raw

    def beta_to_fit_scale(self, beta_raw: Array) -> Array:
        """
        Convert coefficients for raw X into coefficients for standardized X.

        This is the inverse operation of beta_to_original_scale().

        eta = X_raw @ beta_raw = X_std @ beta_std

        For j != intercept:
          beta_std[j] = beta_raw[j] * scale[j]
        Intercept:
          beta_std[intercept] = beta_raw[intercept] + sum_{j!=int} beta_raw[j] * mean[j]
        """
        if not self.fitted_:
            raise RuntimeError("Standardizer must be fit() before beta_to_fit_scale().")
        b = np.asarray(beta_raw, dtype=float).copy()
        mean = self.mean_
        scale = self.scale_

        b_std = np.zeros_like(b)
        ic = self.intercept_col
        for j in range(b.size):
            if j == ic:
                continue
            b_std[j] = b[j] * scale[j]

        b_std[ic] = b[ic]
        for j in range(b.size):
            if j == ic:
                continue
            b_std[ic] += b[j] * mean[j]
        return b_std
