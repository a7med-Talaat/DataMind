"""
ml_readiness.py — ML readiness scoring and reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_ml_readiness(
    df: pd.DataFrame,
    col_types: dict,
    target_col: Optional[str] = None,
) -> dict:
    """
    Score the dataset on ML readiness across 7 dimensions (0–100 each).
    Returns a dict with score, grade, breakdown, issues, and colour.
    """
    scores: dict[str, float] = {}

    # ── 1. Completeness (no missing values) ──────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    null_pct = df.isnull().sum().sum() / max(total_cells, 1) * 100
    scores["completeness"] = max(0.0, 100.0 - null_pct * 5)

    # ── 2. Encoding (all feature columns are numeric) ─────────────────────
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col and target_col in non_numeric:
        non_numeric = [c for c in non_numeric if c != target_col]
    n_features = max(len(df.columns) - (1 if target_col else 0), 1)
    encoding_score = max(0.0, (1 - len(non_numeric) / n_features) * 100)
    scores["encoding"] = encoding_score

    # ── 3. No duplicates ──────────────────────────────────────────────────
    dup_pct = df.duplicated().sum() / max(len(df), 1) * 100
    scores["no_duplicates"] = max(0.0, 100.0 - dup_pct * 10)

    # ── 4. Scale uniformity ───────────────────────────────────────────────
    num_feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]
    if len(num_feature_cols) >= 2:
        ranges = [
            df[c].max() - df[c].min()
            for c in num_feature_cols
            if df[c].max() != df[c].min()
        ]
        if ranges:
            ratio = min(ranges) / max(ranges) if max(ranges) > 0 else 1.0
            scores["scale_uniformity"] = min(ratio * 100, 100.0)
        else:
            scores["scale_uniformity"] = 100.0
    else:
        scores["scale_uniformity"] = 100.0

    # ── 5. Sample size adequacy ────────────────────────────────────────────
    n = len(df)
    if n >= 100_000:
        scores["sample_size"] = 100.0
    elif n >= 10_000:
        scores["sample_size"] = 90.0
    elif n >= 1_000:
        scores["sample_size"] = 70.0
    elif n >= 200:
        scores["sample_size"] = 50.0
    else:
        scores["sample_size"] = max(5.0, n / 2)

    # ── 6. Class balance (if classification target) ────────────────────────
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(normalize=True)
        if len(vc) >= 2:
            balance_ratio = vc.min() / vc.max()
            scores["class_balance"] = balance_ratio * 100
        else:
            scores["class_balance"] = 50.0   # only 1 class — bad
    else:
        scores["class_balance"] = 100.0

    # ── 7. Feature dimensionality ─────────────────────────────────────────
    n_feat = len(df.columns) - (1 if target_col else 0)
    if 2 <= n_feat <= 50:
        scores["feature_count"] = 100.0
    elif n_feat > 200:
        scores["feature_count"] = max(20.0, 100 - (n_feat - 200) * 0.3)
    elif n_feat > 50:
        scores["feature_count"] = max(60.0, 100 - (n_feat - 50) * 0.8)
    elif n_feat == 1:
        scores["feature_count"] = 30.0
    else:
        scores["feature_count"] = 10.0

    # ── Weighted total ────────────────────────────────────────────────────
    weights = {
        "completeness": 0.25,
        "encoding": 0.20,
        "no_duplicates": 0.10,
        "scale_uniformity": 0.15,
        "sample_size": 0.15,
        "class_balance": 0.10,
        "feature_count": 0.05,
    }
    total = sum(scores[k] * weights[k] for k in weights)

    # ── Human-readable labels ─────────────────────────────────────────────
    label_map = {
        "completeness": "Data Completeness",
        "encoding": "Feature Encoding",
        "no_duplicates": "No Duplicates",
        "scale_uniformity": "Scale Uniformity",
        "sample_size": "Sample Size",
        "class_balance": "Class Balance",
        "feature_count": "Feature Count",
    }
    breakdown = {label_map[k]: round(scores[k], 1) for k in scores}

    # ── Grade ─────────────────────────────────────────────────────────────
    if total >= 88:
        grade, color = "Excellent 🟢", "#10B981"
    elif total >= 72:
        grade, color = "Good 🔵", "#3ECFCF"
    elif total >= 55:
        grade, color = "Fair 🟡", "#F59E0B"
    else:
        grade, color = "Needs Work 🔴", "#EF4444"

    # ── Issues list ───────────────────────────────────────────────────────
    issues: list[str] = []
    if scores["completeness"] < 75:
        missing_pct = 100 - scores["completeness"] / 5
        issues.append(f"High missing data ({missing_pct:.1f}%) — impute before training")
    if scores["encoding"] < 80:
        issues.append(f"{len(non_numeric)} non-numeric column(s) still need encoding")
    if scores["no_duplicates"] < 85:
        issues.append(f"Duplicate rows detected — remove to avoid data leakage")
    if scores["scale_uniformity"] < 40:
        issues.append("Feature scales vary widely — apply normalisation/standardisation")
    if scores["class_balance"] < 60 and target_col:
        issues.append("Class imbalance detected — consider SMOTE or class weighting")
    if scores["sample_size"] < 70:
        issues.append(f"Small dataset ({len(df)} rows) — risk of overfitting; gather more data")
    if scores["feature_count"] < 60:
        issues.append("Very high feature count — consider dimensionality reduction (PCA, SelectKBest)")

    return {
        "score": round(total, 1),
        "grade": grade,
        "color": color,
        "breakdown": breakdown,
        "raw_scores": scores,
        "weights": weights,
        "issues": issues,
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }
