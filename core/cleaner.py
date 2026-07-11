"""
cleaner.py — Smart data cleaning pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


# ─── Auto-config ──────────────────────────────────────────────────────────────

def get_auto_cleaning_config(df: pd.DataFrame, col_types: dict) -> dict:
    """
    Auto-generate a recommended cleaning config based on data characteristics.
    Returns a config dict consumed by apply_cleaning().
    """
    config: dict = {"remove_duplicates": True, "columns": {}}

    for col in df.columns:
        null_pct = df[col].isnull().sum() / max(len(df), 1) * 100
        col_cfg: dict = {"null_strategy": "none", "outlier_strategy": "keep"}

        # --- Missing value strategy ---
        if null_pct > 0:
            if col in col_types.get("numeric", []):
                skew = abs(df[col].skew()) if df[col].count() > 0 else 0
                col_cfg["null_strategy"] = "median" if skew > 1 else "mean"
            elif col in col_types.get("categorical", []) or col in col_types.get("boolean", []):
                col_cfg["null_strategy"] = "mode"
            elif col in col_types.get("datetime", []):
                col_cfg["null_strategy"] = "drop"
            else:
                col_cfg["null_strategy"] = "mode"

        # --- Outlier strategy ---
        if col in col_types.get("numeric", []):
            clean = df[col].dropna()
            if len(clean) > 0:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_pct = ((clean < lower) | (clean > upper)).sum() / max(len(df), 1) * 100
                if outlier_pct > 0.5:
                    col_cfg["outlier_strategy"] = "cap"

        config["columns"][col] = col_cfg

    return config


# ─── Apply cleaning ───────────────────────────────────────────────────────────

def apply_cleaning(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, list[str]]:
    """
    Apply cleaning steps defined in config dict.
    Returns (cleaned_df, changelog).
    """
    df = df.copy()
    changelog: list[str] = []

    # 1. Remove duplicates
    if config.get("remove_duplicates", False):
        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = n_before - len(df)
        if removed > 0:
            changelog.append(f"✅ Removed **{removed}** duplicate rows ({removed/n_before*100:.1f}%)")
        else:
            changelog.append("ℹ️ No duplicate rows found")

    # 2. Per-column operations
    for col, col_cfg in config.get("columns", {}).items():
        if col not in df.columns:
            continue

        null_count = int(df[col].isnull().sum())
        null_strategy = col_cfg.get("null_strategy", "none")
        outlier_strategy = col_cfg.get("outlier_strategy", "keep")

        # --- Handle nulls ---
        if null_count > 0 and null_strategy != "none":
            if null_strategy == "drop":
                n_before = len(df)
                df = df.dropna(subset=[col]).reset_index(drop=True)
                changelog.append(f"✅ **[{col}]** Dropped {n_before - len(df)} rows with null values")

            elif null_strategy == "mean":
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                changelog.append(f"✅ **[{col}]** Filled {null_count} nulls with mean `{fill_val:.4f}`")

            elif null_strategy == "median":
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                changelog.append(f"✅ **[{col}]** Filled {null_count} nulls with median `{fill_val:.4f}`")

            elif null_strategy == "mode":
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    fill_val = mode_vals[0]
                    df[col] = df[col].fillna(fill_val)
                    changelog.append(f"✅ **[{col}]** Filled {null_count} nulls with mode `{fill_val}`")

            elif null_strategy == "ffill":
                df[col] = df[col].ffill()
                changelog.append(f"✅ **[{col}]** Forward-filled {null_count} null values")

            elif null_strategy == "bfill":
                df[col] = df[col].bfill()
                changelog.append(f"✅ **[{col}]** Backward-filled {null_count} null values")

            elif null_strategy == "custom":
                custom_val = col_cfg.get("custom_value", 0)
                df[col] = df[col].fillna(custom_val)
                changelog.append(f"✅ **[{col}]** Filled {null_count} nulls with custom value `{custom_val}`")

            elif null_strategy == "zero":
                df[col] = df[col].fillna(0)
                changelog.append(f"✅ **[{col}]** Filled {null_count} nulls with `0`")

        # --- Handle outliers (numeric only) ---
        if (
            outlier_strategy != "keep"
            and pd.api.types.is_numeric_dtype(df[col])
        ):
            clean = df[col].dropna()
            if len(clean) > 0:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                n_out = int(outlier_mask.sum())

                if n_out > 0:
                    if outlier_strategy == "cap":
                        df[col] = df[col].clip(lower=lower, upper=upper)
                        changelog.append(
                            f"✅ **[{col}]** Capped {n_out} outliers to "
                            f"[`{lower:.3f}`, `{upper:.3f}`]"
                        )
                    elif outlier_strategy == "remove":
                        n_before = len(df)
                        df = df[~outlier_mask].reset_index(drop=True)
                        changelog.append(
                            f"✅ **[{col}]** Removed {n_before - len(df)} outlier rows"
                        )

    if not changelog:
        changelog.append("ℹ️ No changes applied — data looks clean already!")

    return df, changelog


# ─── Diff summary ─────────────────────────────────────────────────────────────

def get_cleaning_diff(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Return a before/after summary dict."""
    return {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed": len(df_before) - len(df_after),
        "nulls_before": int(df_before.isnull().sum().sum()),
        "nulls_after": int(df_after.isnull().sum().sum()),
        "nulls_removed": int(df_before.isnull().sum().sum()) - int(df_after.isnull().sum().sum()),
        "dups_before": int(df_before.duplicated().sum()),
        "dups_after": int(df_after.duplicated().sum()),
    }
