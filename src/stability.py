"""
stability.py

Stability and drift-aware checks:
- PSI (Population Stability Index) for numeric features across time windows
- KS test for distribution shifts (optional heuristic)
- Segment-based stability checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class StabilityConfig:
    psi_bins: int = 10
    psi_epsilon: float = 1e-6
    psi_warn: float = 0.1
    psi_alert: float = 0.25


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Compute PSI for two numeric arrays."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Bin edges based on expected
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.nan

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e_dist = e_counts / max(e_counts.sum(), 1)
    a_dist = a_counts / max(a_counts.sum(), 1)

    e_dist = np.clip(e_dist, eps, 1)
    a_dist = np.clip(a_dist, eps, 1)

    psi = np.sum((a_dist - e_dist) * np.log(a_dist / e_dist))
    return float(psi)


def numeric_drift_report(
    df: pd.DataFrame,
    time_col: str,
    split_time: str,
    features: Optional[List[str]] = None,
    cfg: Optional[StabilityConfig] = None,
) -> pd.DataFrame:
    """
    Compute PSI and KS-test p-values for numeric features comparing
    pre vs post split_time.
    """
    cfg = cfg or StabilityConfig()
    if time_col not in df.columns:
        return pd.DataFrame()

    t = pd.to_datetime(df[time_col], errors="coerce")
    split = pd.to_datetime(split_time)

    pre = df[t < split]
    post = df[t >= split]

    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if features:
        num = [c for c in num if c in features]

    rows = []
    for col in num:
        e = pre[col].to_numpy(dtype=float)
        a = post[col].to_numpy(dtype=float)

        psi_val = _psi(e, a, bins=cfg.psi_bins, eps=cfg.psi_epsilon)
        ks_p = np.nan
        try:
            e2 = e[~np.isnan(e)]
            a2 = a[~np.isnan(a)]
            if len(e2) > 20 and len(a2) > 20:
                ks_p = float(ks_2samp(e2, a2).pvalue)
        except Exception:
            pass

        flag = "ok"
        if not np.isnan(psi_val):
            if psi_val >= cfg.psi_alert:
                flag = "alert"
            elif psi_val >= cfg.psi_warn:
                flag = "warn"

        rows.append(
            {
                "feature": col,
                "psi": psi_val,
                "ks_pvalue": ks_p,
                "flag": flag,
                "pre_n": int(pre[col].notna().sum()),
                "post_n": int(post[col].notna().sum()),
            }
        )

    out = pd.DataFrame(rows).sort_values(["flag", "psi"], ascending=[True, False])
    return out


def segment_stability(
    df: pd.DataFrame,
    segment_col: str,
    target_col: Optional[str] = None,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Quick segment check:
    - segment sizes
    - optional target rate by segment (if binary-like target)
    """
    if segment_col not in df.columns:
        return pd.DataFrame()

    seg = df[segment_col].astype("object")
    counts = seg.value_counts().head(top_k)
    out = pd.DataFrame({"segment": counts.index.astype(str), "count": counts.values})
    out["pct"] = out["count"] / max(out["count"].sum(), 1)

    if target_col and target_col in df.columns:
        # try binary-ish
        y = df[target_col]
        y2 = y.astype(str).str.lower().str.strip()
        mapping = {"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0}
        if set(y2.dropna().unique()).issubset(set(mapping.keys())):
            ybin = y2.map(mapping).astype(float)
            tmp = pd.DataFrame({"seg": seg, "y": ybin}).dropna()
            rate = tmp.groupby("seg")["y"].mean()
            out["target_rate"] = out["segment"].map(rate).astype(float)

    return out
