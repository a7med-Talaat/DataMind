"""
visualizations.py — Advanced Chart Library for DataMind
Provides premium interactive Plotly visualizations beyond the basic EDA module.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared color palette
_PALETTE = ["#6C63FF", "#3ECFCF", "#F59E0B", "#EF4444", "#10B981", "#A78BFA", "#FB923C", "#38BDF8"]
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0", family="Inter, sans-serif"),
)
_GRID = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)")


def _apply_dark(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**_DARK_LAYOUT, height=height)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ── Violin Plot ────────────────────────────────────────────────────────────────

def plot_violin(df: pd.DataFrame, col: str, group_col: Optional[str] = None) -> go.Figure:
    """Violin plot showing full distribution shape with optional grouping."""
    if group_col and group_col in df.columns:
        fig = px.violin(
            df, y=col, x=group_col, color=group_col,
            box=True, points="outliers",
            color_discrete_sequence=_PALETTE,
            title=f"Distribution of <b>{col}</b> by {group_col}",
            template="plotly_dark",
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=df[col].dropna(),
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(108,99,255,0.25)",
            line_color="#6C63FF",
            name=col,
            points="outliers",
            marker=dict(color="#F59E0B", size=3, opacity=0.6),
        ))
        fig.update_layout(title=f"Distribution Shape: <b>{col}</b>")
    return _apply_dark(fig)


# ── Scatter Matrix ─────────────────────────────────────────────────────────────

def plot_scatter_matrix(
    df: pd.DataFrame, cols: list[str], color_col: Optional[str] = None
) -> go.Figure:
    """Interactive pair scatter matrix for multiple columns."""
    plot_df = df[cols + ([color_col] if color_col else [])].dropna()
    fig = px.scatter_matrix(
        plot_df,
        dimensions=cols,
        color=color_col,
        color_discrete_sequence=_PALETTE,
        template="plotly_dark",
        title="Scatter Matrix — Pairwise Feature Relationships",
        opacity=0.55,
    )
    fig.update_traces(
        diagonal_visible=True,
        showupperhalf=False,
        marker=dict(size=3),
    )
    size = max(500, len(cols) * 180)
    return _apply_dark(fig, height=size)


# ── 3D Scatter Plot ────────────────────────────────────────────────────────────

def plot_3d_scatter(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
    color_col: Optional[str] = None, size_col: Optional[str] = None,
) -> go.Figure:
    """3D scatter plot with optional color and size encoding."""
    plot_df = df[[x_col, y_col, z_col] + ([color_col] if color_col else []) + ([size_col] if size_col else [])].dropna()
    fig = px.scatter_3d(
        plot_df, x=x_col, y=y_col, z=z_col,
        color=color_col,
        size=size_col,
        color_discrete_sequence=_PALETTE,
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title=f"3D Scatter: {x_col} × {y_col} × {z_col}",
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(**_DARK_LAYOUT, height=560)
    return fig


# ── Sunburst Chart ─────────────────────────────────────────────────────────────

def plot_sunburst(df: pd.DataFrame, path_cols: list[str], value_col: Optional[str] = None) -> go.Figure:
    """Hierarchical sunburst chart for categorical breakdown."""
    plot_df = df[path_cols + ([value_col] if value_col else [])].dropna()
    kwargs = dict(
        path=path_cols,
        color_discrete_sequence=_PALETTE,
        template="plotly_dark",
        title="Hierarchical Breakdown: " + " → ".join(path_cols),
    )
    if value_col:
        kwargs["values"] = value_col
    fig = px.sunburst(plot_df, **kwargs)
    return _apply_dark(fig, height=500)


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curves(roc_data: dict[str, dict]) -> go.Figure:
    """
    Plot ROC curves for multiple models.
    roc_data: {model_name: {"fpr": [...], "tpr": [...], "auc": float}}
    """
    fig = go.Figure()
    colors = _PALETTE

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="rgba(255,255,255,0.2)", width=1.5),
        name="Random Classifier (AUC=0.50)",
        showlegend=True,
    ))

    for i, (name, data) in enumerate(roc_data.items()):
        auc = data.get("auc", 0)
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=2.5),
            name=f"{name} (AUC = {auc:.3f})",
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(108,99,255,0.06)" if i == 0 else None,
        ))

    fig.update_layout(
        title="ROC Curves — Classification Performance",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(
            x=0.62, y=0.05,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        **_DARK_LAYOUT,
        height=450,
    )
    fig.update_xaxes(**_GRID, range=[0, 1])
    fig.update_yaxes(**_GRID, range=[0, 1.02])
    return fig


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], model_name: str = "") -> go.Figure:
    """Heatmap confusion matrix with annotations."""
    # Normalize for color, show raw counts as text
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    text = [[f"{cm[i][j]}<br><span style='font-size:0.75em'>({cm_norm[i][j]*100:.1f}%)</span>"
             for j in range(len(labels))] for i in range(len(labels))]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0, "rgba(108,99,255,0.05)"],
            [0.5, "rgba(108,99,255,0.4)"],
            [1, "rgba(108,99,255,0.95)"],
        ],
        showscale=True,
        colorbar=dict(title="Norm. Rate", tickfont=dict(color="#94A3B8")),
    ))

    n = len(labels)
    for i in range(n):
        for j in range(n):
            color = "#ffffff" if cm_norm[i][j] > 0.5 else "#94A3B8"
            fig.add_annotation(
                x=j, y=i,
                text=f"<b>{cm[i][j]}</b>",
                showarrow=False,
                font=dict(color=color, size=14),
            )

    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis=dict(side="bottom"),
        **_DARK_LAYOUT,
        height=400,
    )
    return fig


# ── Time Series Plot ───────────────────────────────────────────────────────────

def plot_time_series(
    df: pd.DataFrame, date_col: str, value_col: str,
    window: int = 7, show_rolling: bool = True
) -> go.Figure:
    """Line chart with optional rolling mean overlay."""
    ts = df[[date_col, value_col]].dropna().sort_values(date_col)
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts[date_col], y=ts[value_col],
        mode="lines",
        name=value_col,
        line=dict(color="#6C63FF", width=1.5),
        opacity=0.7,
    ))

    if show_rolling and len(ts) > window:
        rolling = ts[value_col].rolling(window=window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=ts[date_col], y=rolling,
            mode="lines",
            name=f"{window}-period Rolling Mean",
            line=dict(color="#10B981", width=2.5, dash="solid"),
        ))

    fig.update_layout(
        title=f"Time Series: <b>{value_col}</b> over {date_col}",
        xaxis_title=date_col,
        yaxis_title=value_col,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
        **_DARK_LAYOUT,
        height=420,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def plot_rolling_stats(
    df: pd.DataFrame, date_col: str, value_col: str, window: int = 7
) -> go.Figure:
    """Two-panel: rolling mean (top) and rolling std/volatility (bottom)."""
    ts = df[[date_col, value_col]].dropna().sort_values(date_col)
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])

    roll_mean = ts[value_col].rolling(window=window, center=True).mean()
    roll_std = ts[value_col].rolling(window=window, center=True).std()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"Rolling Mean (window={window})", f"Rolling Std / Volatility (window={window})"],
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # Raw
    fig.add_trace(go.Scatter(
        x=ts[date_col], y=ts[value_col],
        name="Raw", line=dict(color="#6C63FF", width=1), opacity=0.45,
    ), row=1, col=1)

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=ts[date_col], y=roll_mean,
        name="Rolling Mean", line=dict(color="#3ECFCF", width=2.5),
    ), row=1, col=1)

    # Rolling std
    fig.add_trace(go.Scatter(
        x=ts[date_col], y=roll_std,
        name="Volatility", line=dict(color="#F59E0B", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
    ), row=2, col=1)

    fig.update_layout(**_DARK_LAYOUT, height=520, showlegend=True)
    for row in [1, 2]:
        fig.update_xaxes(**_GRID, row=row, col=1)
        fig.update_yaxes(**_GRID, row=row, col=1)
    return fig


# ── Partial Dependence Plot (approximation) ────────────────────────────────────

def plot_partial_dependence(
    model, X: np.ndarray, feature_idx: int, feature_name: str,
    feature_values: np.ndarray, n_points: int = 50
) -> go.Figure:
    """Approximates a partial dependence plot for one feature."""
    X_copy = X.copy()
    grid = np.linspace(feature_values.min(), feature_values.max(), n_points)
    pdp_values = []

    for val in grid:
        X_copy[:, feature_idx] = val
        preds = model.predict(X_copy)
        pdp_values.append(float(np.mean(preds)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=pdp_values,
        mode="lines+markers",
        line=dict(color="#6C63FF", width=2.5),
        marker=dict(size=4, color="#3ECFCF"),
        name="Partial Dependence",
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.06)",
    ))
    fig.update_layout(
        title=f"Partial Dependence: <b>{feature_name}</b>",
        xaxis_title=feature_name,
        yaxis_title="Predicted (avg)",
        **_DARK_LAYOUT,
        height=350,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ── Correlation Heatmap (enhanced, returns raw corr too) ──────────────────────

def plot_enhanced_heatmap(df: pd.DataFrame, method: str = "pearson") -> tuple[go.Figure, pd.DataFrame]:
    """Enhanced correlation heatmap that also returns the corr matrix for drill-down."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    z = corr.where(~np.triu(np.ones(corr.shape, dtype=bool), k=0)).values

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        zmin=-1, zmax=1,
        colorscale=[
            [0,   "#EF4444"],
            [0.5, "rgba(15,23,42,0.8)"],
            [1,   "#6C63FF"],
        ],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
        colorbar=dict(
            title=f"{method.title()} r",
            tickfont=dict(color="#94A3B8"),
        ),
    ))
    fig.update_layout(
        title=f"{method.title()} Correlation Matrix",
        **_DARK_LAYOUT,
        height=max(400, len(corr) * 45),
    )
    return fig, corr


# ── Column Health Bars ─────────────────────────────────────────────────────────

def compute_column_health(profile: dict, col_types: dict) -> list[dict]:
    """
    Returns a list of {col, health_score, issues} for rendering health bars.
    Health score: 0–100 based on null%, outlier%, uniqueness.
    """
    results = []
    for col, info in profile["columns"].items():
        score = 100.0
        issues = []

        null_pct = info.get("null_pct", 0)
        if null_pct > 0:
            penalty = min(null_pct * 2, 50)
            score -= penalty
            issues.append(f"{null_pct:.1f}% missing")

        outlier_pct = info.get("outlier_pct", 0)
        if outlier_pct > 0:
            penalty = min(outlier_pct * 1.5, 30)
            score -= penalty
            issues.append(f"{outlier_pct:.1f}% outliers")

        unique_pct = info.get("unique_pct", 100)
        if unique_pct == 100 and info.get("unique_count", 0) == info.get("null_count", -1) + len(profile.get("columns", {})):
            pass  # ID-like column — not necessarily bad
        elif unique_pct < 0.5 and info.get("unique_count", 10) < 2:
            score -= 20
            issues.append("near-constant")

        score = max(0.0, min(100.0, score))
        results.append({
            "col": col,
            "health_score": round(score, 1),
            "issues": issues,
            "null_pct": null_pct,
            "outlier_pct": outlier_pct,
        })

    return sorted(results, key=lambda x: x["health_score"])
