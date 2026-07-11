"""
profiler.py — Comprehensive dataset profiling.
"""
import pandas as pd
import numpy as np
from typing import Any


def generate_profile(df: pd.DataFrame, col_types: dict) -> dict:
    """Generate a full statistical profile of the dataset."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    profile: dict = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(df.duplicated().sum() / max(n_rows, 1) * 100, 2),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_pct": round(df.isnull().sum().sum() / max(total_cells, 1) * 100, 2),
        "col_types": col_types,
        "columns": {},
    }

    for col in df.columns:
        s = df[col]
        null_count = int(s.isnull().sum())
        n_unique = int(s.nunique())

        col_info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": round(null_count / max(n_rows, 1) * 100, 2),
            "unique_count": n_unique,
            "unique_pct": round(n_unique / max(n_rows, 1) * 100, 2),
            "sample_values": _safe_sample(s, 5),
        }

        if col in col_types.get("numeric", []):
            clean = s.dropna()
            if len(clean) > 0:
                q1 = float(clean.quantile(0.25))
                q3 = float(clean.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = clean[(clean < lower) | (clean > upper)]

                col_info.update(
                    {
                        "mean": round(float(clean.mean()), 6),
                        "std": round(float(clean.std()), 6),
                        "min": float(clean.min()),
                        "max": float(clean.max()),
                        "median": float(clean.median()),
                        "q25": q1,
                        "q75": q3,
                        "skewness": round(float(clean.skew()), 4),
                        "kurtosis": round(float(clean.kurt()), 4),
                        "outlier_count": int(len(outliers)),
                        "outlier_pct": round(len(outliers) / max(n_rows, 1) * 100, 2),
                        "iqr_lower": round(lower, 6),
                        "iqr_upper": round(upper, 6),
                    }
                )
            else:
                col_info.update(
                    {k: None for k in ["mean", "std", "min", "max", "median", "q25", "q75", "skewness", "kurtosis"]}
                )
                col_info.update({"outlier_count": 0, "outlier_pct": 0.0})

        elif col in col_types.get("categorical", []):
            vc = s.value_counts()
            col_info["top_values"] = {str(k): int(v) for k, v in vc.head(15).items()}
            col_info["top_value"] = str(vc.index[0]) if len(vc) > 0 else None
            col_info["top_freq"] = int(vc.iloc[0]) if len(vc) > 0 else 0
            col_info["top_freq_pct"] = round(vc.iloc[0] / max(n_rows, 1) * 100, 2) if len(vc) > 0 else 0

        profile["columns"][col] = col_info

    # Overall quality score
    completeness = 100 - profile["null_pct"]
    uniqueness = 100 - profile["duplicate_pct"]
    profile["quality_score"] = round(completeness * 0.6 + uniqueness * 0.4, 1)

    return profile


def _safe_sample(s: pd.Series, n: int) -> list:
    """Return up to n non-null sample values from a series."""
    non_null = s.dropna()
    sample_size = min(n, len(non_null))
    if sample_size == 0:
        return []
    return [str(v) for v in non_null.sample(sample_size).tolist()]
