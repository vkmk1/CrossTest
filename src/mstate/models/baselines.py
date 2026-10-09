"""Context baselines (ARCHITECTURE.md §3): none, hmm, pca.

Each produces a *market-level* context, one row per date, which the context builder broadcasts
to every asset. All are fitted on train-period dates only and are causal at inference: the row
for date t depends only on market rows with date <= t.

Outputs are zero-padded to the shared context dimension ``dim`` so the forecaster (including its
FiLM MLP) has the same shape and parameter count under every ablation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from mstate.data.features import mkt_columns


def market_frame(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per date with the mkt_* columns (identical across assets at a date)."""
    cols = list(columns or mkt_columns())
    return df.groupby("date")[cols].first().sort_index().astype(np.float64)


def _pad(arr: np.ndarray, dim: int) -> np.ndarray:
    if arr.shape[1] > dim:
        raise ValueError(f"context has {arr.shape[1]} columns > dim {dim}")
    return np.pad(arr, ((0, 0), (0, dim - arr.shape[1])))


class NoneContext:
    def __init__(self, dim: int = 16):
        self.dim = dim

    def fit(self, mkt: pd.DataFrame) -> NoneContext:
        return self

    def transform(self, mkt: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.zeros((len(mkt), self.dim)), index=mkt.index,
                            columns=[f"c_{i}" for i in range(self.dim)])


class HMMContext:
    """Gaussian HMM on a market series; context = filtered posterior P(s_t | x_{<=t}).

    States are ordered by their mean (state 0 = lowest). ``predict_proba`` is never used for the
    context because it smooths with future data.
    """

    def __init__(self, n_states: int = 3, dim: int = 16, series: str | Sequence[str] = "mkt_median_lrv",
                 n_init: int = 5, n_iter: int = 200, seed: int = 0):
        self.n_states, self.dim = n_states, dim
        self.series = [series] if isinstance(series, str) else list(series)
        self.n_init, self.n_iter, self.seed = n_init, n_iter, seed

    def _x(self, mkt: pd.DataFrame) -> np.ndarray:
        return mkt[self.series].to_numpy(dtype=np.float64)

    def fit(self, mkt_train: pd.DataFrame) -> HMMContext:
        x = self._x(mkt_train)
        best, best_ll = None, -np.inf
        for k in range(self.n_init):
            m = GaussianHMM(n_components=self.n_states, covariance_type="full",
                            n_iter=self.n_iter, random_state=self.seed + k)
            m.fit(x)
            ll = m.score(x)
            if ll > best_ll:
                best, best_ll = m, ll
        assert best is not None
        self.model = best
        self.order = np.argsort(best.means_[:, 0])
        return self

    def filtered(self, x: np.ndarray) -> np.ndarray:
        """Forward algorithm in log space: ``[T, K]`` filtered posteriors (model state order)."""
        m = self.model
        logb = np.column_stack([
            multivariate_normal(m.means_[k], m.covars_[k], allow_singular=True).logpdf(x)
            for k in range(self.n_states)
        ]).reshape(len(x), self.n_states)
        log_a = np.log(m.transmat_ + 1e-300)
        out = np.empty_like(logb)
        alpha = np.log(m.startprob_ + 1e-300) + logb[0]
        out[0] = alpha - logsumexp(alpha)
        for t in range(1, len(x)):
            alpha = logsumexp(out[t - 1][:, None] + log_a, axis=0) + logb[t]
            out[t] = alpha - logsumexp(alpha)
        return np.exp(out)

    def transform(self, mkt: pd.DataFrame) -> pd.DataFrame:
        post = self.filtered(self._x(mkt))[:, self.order]
        return pd.DataFrame(_pad(post, self.dim), index=mkt.index,
                            columns=[f"c_{i}" for i in range(self.dim)])


class PCAContext:
    """PCA of the market vector (mkt_median_f, mkt_mad_f), stacked with ``lags`` causal lags.

    The 5 base features give a 10-dim market vector, fewer than d = 16, so the vector at t is
    concatenated with its values at t-1..t-lags to reach the context dimension. Inputs are
    z-scored with train moments and components are whitened on train.
    """

    def __init__(self, dim: int = 16, lags: int = 1, columns: Sequence[str] | None = None):
        self.dim, self.lags = dim, lags
        self.columns = list(columns or mkt_columns())

    def _stack(self, mkt: pd.DataFrame) -> np.ndarray:
        base = mkt[self.columns]
        parts = [base] + [base.shift(k) for k in range(1, self.lags + 1)]
        x = pd.concat(parts, axis=1).bfill()  # only the first `lags` dates of the calendar
        return x.to_numpy(dtype=np.float64)

    def fit(self, mkt_train: pd.DataFrame) -> PCAContext:
        x = self._stack(mkt_train)
        if x.shape[1] < self.dim:
            raise ValueError(f"PCA input dim {x.shape[1]} < context dim {self.dim}; raise lags")
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        self.pca = PCA(n_components=self.dim, whiten=True, random_state=0).fit((x - self.mu) / self.sd)
        return self

    def transform(self, mkt: pd.DataFrame) -> pd.DataFrame:
        z = self.pca.transform((self._stack(mkt) - self.mu) / self.sd)
        return pd.DataFrame(z, index=mkt.index, columns=[f"c_{i}" for i in range(self.dim)])


class JumpContext:
    """Nystrup-Kolm-Lindström jump model. Phase 1 (EXPERIMENTS.md); not built for Phase 0."""

    def __init__(self, *args: object, **kwargs: object):
        raise NotImplementedError("jump baseline is scheduled for Phase 1")


def make_market_context(cfg_context: dict) -> NoneContext | HMMContext | PCAContext:
    src, dim = cfg_context["source"], int(cfg_context["dim"])
    if src == "none":
        return NoneContext(dim)
    if src == "hmm":
        return HMMContext(int(cfg_context.get("n_states", 3)), dim,
                          cfg_context.get("series", "mkt_median_lrv"))
    if src == "pca":
        return PCAContext(dim, int(cfg_context.get("lags", 1)))
    if src == "jump":
        return JumpContext()
    raise ValueError(f"{src} is not a market-level baseline")
