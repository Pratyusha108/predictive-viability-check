"""
profiling.py

Lightweight data profiling utilities for pre-modeling feasibility checks.
Focus: missingness, duplicates, basic stats, cardinality, and quick sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class ProfileConfig:
    max_unique_preview: int = 10
    high_cardinality_threshold: float = 0.8  # unique_ratio above this is "high-cardinality"
    low_variance_threshold: float = 0.01     # fraction of most common value above this is "low variance"
    sample_rows: int = 5


def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except Exception:
        return 0


def basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic dataset info."""
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
    }


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness by column."""
    total = df.shape[0]
    miss = df.isna().sum()
    pct = (miss / total).replace([np.inf, np.nan], 0.0)
    out = pd.DataFrame({"missing_count": miss, "missing_pct": pct}).sort_values("missing_pct", ascending=False)
    return out


def duplicates_report(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """Duplicate stats overall and optionally by subset keys."""
    dup_all = int(df.duplicated().sum())
    result = {"duplicate_rows_all_columns": dup_all}

    if subset:
        missing_cols = [c for c in subset if c not in df.columns]
        if missing_cols:
            result["subset_error"] = f"Subset columns not found: {missing_cols}"
        else:
            dup_subset = int(df.duplicated(subset=subset).sum())
            result["duplicate_rows_subset"] = dup_subset
            result["subset"] = subset
    return result


def type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize column dtypes and basic characteristics."""
    rows = df.shape[0]
    records = []
    for col in df.columns:
        s = df[col]
        nunique = _safe_nunique(s)
        unique_ratio = (nunique / rows) if rows else 0.0
        missing_pct = float(s.isna().mean())

        # low variance proxy: share of the most common value
        top_share = 0.0
        try:
            vc = s.value_counts(dropna=True, normalize=True)
            if len(vc) > 0:
                top_share = float(vc.iloc[0])
        except Exception:
            pass

        records.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_pct": missing_pct,
                "nunique": nunique,
                "unique_ratio": unique_ratio,
                "top_value_share": top_share,
            }
        )

    out = pd.DataFrame(records).sort_values(["missing_pct", "unique_ratio"], ascending=[False, False])
    return out


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric descriptive stats."""
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    desc["missing_pct"] = num.isna().mean()
    return desc


def categorical_preview(df: pd.DataFrame, config: Optional[ProfileConfig] = None) -> Dict[str, List[str]]:
    """Preview top categories for object/category columns."""
    cfg = config or ProfileConfig()
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    preview: Dict[str, List[str]] = {}
    for col in cats:
        s = df[col].astype("object")
        top = (
            s.value_counts(dropna=True)
            .head(cfg.max_unique_preview)
            .index.astype(str)
            .tolist()
        )
        preview[col] = top
    return preview


def red_flag_columns(df: pd.DataFrame, config: Optional[ProfileConfig] = None) -> pd.DataFrame:
    """
    Identify columns that often cause issues in predictive modeling:
    - very high missingness
    - near-unique IDs (high cardinality)
    - low-variance columns
    """
    cfg = config or ProfileConfig()
    t = type_summary(df)

    high_missing = t[t["missing_pct"] >= 0.5].copy()
    high_missing["flag"] = "high_missingness"

    high_card = t[t["unique_ratio"] >= cfg.high_cardinality_threshold].copy()
    high_card["flag"] = "high_cardinality_possible_id"

    low_var = t[t["top_value_share"] >= (1.0 - cfg.low_variance_threshold)].copy()
    low_var["flag"] = "low_variance"

    out = pd.concat([high_missing, high_card, low_var], axis=0, ignore_index=True)
    out = out.sort_values(["flag", "missing_pct", "unique_ratio"], ascending=[True, False, False])
    return out


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with a few safe defaults."""
    return pd.read_csv(path, low_memory=False)


if __name__ == "__main__":
    # Minimal CLI usage example
    import argparse

    parser = argparse.ArgumentParser(description="Quick dataset profiling")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--key", nargs="*", default=None, help="Optional key columns to check duplicates")
    args = parser.parse_args()

    df0 = load_csv(args.csv)
    print("Overview:", basic_overview(df0))
    print("\nMissingness (top 15):")
    print(missingness_report(df0).head(15))
    print("\nDuplicates:", duplicates_report(df0, subset=args.key))
    print("\nRed flags (top 20):")
    print(red_flag_columns(df0).head(20))
