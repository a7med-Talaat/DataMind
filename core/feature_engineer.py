"""
feature_engineer.py — Encoding, scaling, datetime decomposition, and advanced transforms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from typing import Optional, Tuple


# ─── Auto-config ──────────────────────────────────────────────────────────────

def get_auto_engineering_config(
    df: pd.DataFrame, col_types: dict, target_col: Optional[str] = None
) -> dict:
    """
    Auto-generate feature engineering recommendations based on column types.
    Returns a config dict consumed by apply_feature_engineering().
    """
    config: dict = {"encoding": {}, "scaling": {}, "datetime": {}}

    for col in col_types.get("categorical", []):
        if col == target_col:
            continue
        n_unique = df[col].nunique()
        if n_unique <= 2:
            config["encoding"][col] = "label"
        elif n_unique <= 8:
            config["encoding"][col] = "onehot"
        else:
            config["encoding"][col] = "label"

    for col in col_types.get("boolean", []):
        if col != target_col:
            config["encoding"][col] = "label"

    for col in col_types.get("numeric", []):
        if col != target_col:
            config["scaling"][col] = "standard"

    for col in col_types.get("datetime", []):
        config["datetime"][col] = ["year", "month", "day", "dayofweek", "quarter"]

    return config


# ─── Apply engineering ────────────────────────────────────────────────────────

def apply_feature_engineering(
    df: pd.DataFrame,
    config: dict,
    col_types: dict,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Apply feature engineering and return (transformed_df, changelog).
    Order: datetime → encoding → scaling
    """
    df = df.copy()
    changelog: list[str] = []

    # 1. Datetime decomposition
    for col, features in config.get("datetime", {}).items():
        if col not in df.columns or not features:
            continue
        try:
            dt = pd.to_datetime(df[col], infer_datetime_format=True)
            for feat in features:
                new_col = f"{col}_{feat}"
                val = getattr(dt.dt, feat)
                df[new_col] = val
                changelog.append(f"✅ **[{col}]** Extracted `{feat}` → `{new_col}`")
            df = df.drop(columns=[col])
            changelog.append(f"✅ **[{col}]** Original datetime column removed after decomposition")
        except Exception as e:
            changelog.append(f"⚠️ **[{col}]** Datetime decomposition failed: {e}")

    # 2. Encoding
    for col, method in config.get("encoding", {}).items():
        if col not in df.columns or method == "none":
            continue
        if col == target_col:
            continue
        try:
            if method == "label":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str).fillna("__MISSING__"))
                n_classes = len(le.classes_)
                changelog.append(f"✅ **[{col}]** Label encoded → `{n_classes}` classes → integers 0–{n_classes-1}")

            elif method == "onehot":
                n_before = len(df.columns)
                df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False, dtype=int)
                n_new = len(df.columns) - n_before + 1
                changelog.append(f"✅ **[{col}]** One-hot encoded → `{n_new}` new binary columns")

            elif method == "ordinal":
                categories = sorted(df[col].dropna().astype(str).unique())
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                df[col] = df[col].astype(str).map(mapping)
                changelog.append(f"✅ **[{col}]** Ordinal encoded (alphabetical order, `{len(categories)}` levels)")

        except Exception as e:
            changelog.append(f"⚠️ **[{col}]** Encoding failed: {e}")

    # 3. Scaling
    for col, method in config.get("scaling", {}).items():
        if col not in df.columns or method == "none":
            continue
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            changelog.append(f"⚠️ **[{col}]** Cannot scale non-numeric column (encode first)")
            continue
        try:
            values = df[[col]].values.astype(float)
            if method == "standard":
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(values)
                changelog.append(f"✅ **[{col}]** Standard scaled → mean ≈ 0, std ≈ 1")
            elif method == "minmax":
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(values)
                changelog.append(f"✅ **[{col}]** MinMax scaled → range [0, 1]")
            elif method == "robust":
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(values)
                changelog.append(f"✅ **[{col}]** Robust scaled → median-centered, IQR-scaled")
        except Exception as e:
            changelog.append(f"⚠️ **[{col}]** Scaling failed: {e}")

    if not changelog:
        changelog.append("ℹ️ No feature engineering applied")

    return df, changelog


# ─── Advanced transforms ──────────────────────────────────────────────────────

def apply_advanced_transforms(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, list[str]]:
    """
    Apply advanced / power-user transformations.
    config keys:
      drop_cols         : list[str]
      log_transform     : list[str]
      sqrt_transform    : list[str]
      square_transform  : list[str]
      interaction_terms : list[tuple[str, str]]
      custom_formulas   : list[dict]  -> [{"name": "col_new", "formula": "colA + colB"}]
      binning           : list[dict]  -> [{"col": "col_name", "bins": 5, "strategy": "quantile"}]
    Returns (transformed_df, changelog).
    """
    df = df.copy()
    changelog: list[str] = []

    # Drop columns
    for col in config.get("drop_cols", []):
        if col in df.columns:
            df = df.drop(columns=[col])
            changelog.append(f"✅ **[{col}]** Column dropped")

    # Log transform (log1p handles zeros safely)
    for col in config.get("log_transform", []):
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        min_val = df[col].min()
        offset = max(0.0, -min_val + 1e-6) if min_val <= 0 else 0.0
        new_col = f"{col}_log"
        df[new_col] = np.log1p(df[col] + offset)
        note = f" (offset {offset:.4f} added)" if offset > 0 else ""
        changelog.append(f"✅ **[{col}]** log1p transform → `{new_col}`{note}")

    # Square-root transform
    for col in config.get("sqrt_transform", []):
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        min_val = df[col].min()
        offset = max(0.0, -min_val) if min_val < 0 else 0.0
        new_col = f"{col}_sqrt"
        df[new_col] = np.sqrt(df[col] + offset)
        note = f" (offset {offset:.4f} added)" if offset > 0 else ""
        changelog.append(f"✅ **[{col}]** sqrt transform → `{new_col}`{note}")

    # Square transform
    for col in config.get("square_transform", []):
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        new_col = f"{col}_sq"
        df[new_col] = df[col] ** 2
        changelog.append(f"✅ **[{col}]** Squared → `{new_col}`")

    # Interaction terms (multiply two columns)
    for pair in config.get("interaction_terms", []):
        if len(pair) != 2:
            continue
        col_a, col_b = pair
        if col_a not in df.columns or col_b not in df.columns:
            continue
        if not (pd.api.types.is_numeric_dtype(df[col_a]) and pd.api.types.is_numeric_dtype(df[col_b])):
            continue
        new_col = f"{col_a}_x_{col_b}"
        df[new_col] = df[col_a] * df[col_b]
        changelog.append(f"✅ Interaction feature `{new_col}` = `{col_a}` × `{col_b}`")

    # Custom Mathematical Formulas
    for item in config.get("custom_formulas", []):
        name = item.get("name", "").strip()
        expr = item.get("formula", "").strip()
        if not name or not expr:
            continue
        try:
            # Safely evaluate mathematical expressions using pandas eval
            res = df.eval(expr)
            df[name] = res
            changelog.append(f"✅ Created formula column `{name}` = `{expr}`")
        except Exception as e:
            changelog.append(f"⚠️ Formula evaluation failed for `{name}` ({expr}): {str(e)}")

    # Binning numerical columns
    for item in config.get("binning", []):
        col = item.get("col", "")
        bins = int(item.get("bins", 5))
        strategy = item.get("strategy", "quantile")  # quantile or uniform
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            new_col = f"{col}_binned"
            if strategy == "quantile":
                df[new_col] = pd.qcut(df[col], q=bins, labels=False, duplicates="drop")
            else:
                df[new_col] = pd.cut(df[col], bins=bins, labels=False)
            # Map index values to integer codes, handle stringification
            df[new_col] = df[new_col].astype(str)
            changelog.append(f"✅ Binned `{col}` ({strategy}, {bins} bins) → `{new_col}`")
        except Exception as e:
            changelog.append(f"⚠️ Binning failed for `{col}`: {str(e)}")

    if not changelog:
        changelog.append("ℹ️ No advanced transforms applied")

    return df, changelog


# ─── Smart column suggestions ─────────────────────────────────────────────────

def get_smart_suggestions(df: pd.DataFrame, col_types: dict) -> list[dict]:
    """
    Scan the dataset and return a list of smart suggestions.
    Each suggestion: {type, severity, columns, message, action}
    """
    suggestions: list[dict] = []
    n = len(df)

    # 1. Likely ID columns (≥95% unique, non-numeric)
    id_cols = [
        col for col in df.columns
        if df[col].nunique() / max(n, 1) >= 0.95
    ]
    if id_cols:
        suggestions.append({
            "type": "id_columns",
            "severity": "warning",
            "columns": id_cols,
            "message": f"**{len(id_cols)} likely ID column(s)** detected (≥95% unique values). "
                       f"These rarely help ML models and should usually be dropped.",
            "action": "drop",
        })

    # 2. Constant / near-constant columns
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        suggestions.append({
            "type": "constant_columns",
            "severity": "error",
            "columns": const_cols,
            "message": f"**{len(const_cols)} constant column(s)** with only 1 unique value — zero variance, no predictive power.",
            "action": "drop",
        })

    # 3. Highly correlated numeric pairs (Pearson > 0.95)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().abs()
        high_corr_pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] >= 0.95:
                    high_corr_pairs.append((cols[i], cols[j], round(float(corr.iloc[i, j]), 3)))
        if high_corr_pairs:
            pair_strs = [f"`{a}` & `{b}` (r={r})" for a, b, r in high_corr_pairs[:5]]
            suggestions.append({
                "type": "high_correlation",
                "severity": "info",
                "columns": [p[0] for p in high_corr_pairs],
                "message": f"**{len(high_corr_pairs)} highly correlated pair(s)** (|r| ≥ 0.95) — consider dropping one from each pair to reduce multicollinearity:\n" + "\n".join(f"• {s}" for s in pair_strs),
                "action": "review",
            })

    # 4. Highly skewed numeric columns (|skew| > 1.5)
    skewed_cols = []
    for col in col_types.get("numeric", []):
        clean = df[col].dropna()
        if len(clean) > 10:
            skew = abs(float(clean.skew()))
            if skew > 1.5:
                skewed_cols.append((col, round(skew, 2)))
    if skewed_cols:
        col_list = ", ".join(f"`{c}` (skew={s})" for c, s in skewed_cols[:5])
        suggestions.append({
            "type": "skewed_columns",
            "severity": "info",
            "columns": [c for c, _ in skewed_cols],
            "message": f"**{len(skewed_cols)} highly skewed column(s)** — log or sqrt transform recommended:\n{col_list}",
            "action": "transform",
        })

    # 5. Columns with >50% missing
    high_null = [(col, round(df[col].isnull().sum() / max(n, 1) * 100, 1))
                 for col in df.columns if df[col].isnull().sum() / max(n, 1) > 0.5]
    if high_null:
        col_list = ", ".join(f"`{c}` ({p}%)" for c, p in high_null)
        suggestions.append({
            "type": "high_missing",
            "severity": "warning",
            "columns": [c for c, _ in high_null],
            "message": f"**{len(high_null)} column(s)** with >50% missing — consider dropping instead of imputing:\n{col_list}",
            "action": "drop",
        })

    # 6. Numeric-looking string columns
    num_str_cols = []
    for col in col_types.get("categorical", []):
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(n, 1) > 0.9:
                num_str_cols.append(col)
        except Exception:
            pass
    if num_str_cols:
        suggestions.append({
            "type": "numeric_strings",
            "severity": "warning",
            "columns": num_str_cols,
            "message": f"**{len(num_str_cols)} column(s)** look numeric but are stored as strings — convert dtype for correct analysis: " + ", ".join(f"`{c}`" for c in num_str_cols),
            "action": "convert",
        })

    return suggestions
