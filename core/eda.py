"""
eda.py — Interactive EDA visualisations using Plotly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats


# ─── Theme ────────────────────────────────────────────────────────────────────

_BG = "rgba(0,0,0,0)"
_GRID = "rgba(255,255,255,0.05)"
_ZERO = "rgba(255,255,255,0.10)"
_FONT_COLOR = "#E2E8F0"
_FONT_FAMILY = "Inter, sans-serif"
_PRIMARY = "#6C63FF"
_SECONDARY = "#3ECFCF"
_ACCENT = "#F59E0B"
_DANGER = "#EF4444"
_SUCCESS = "#10B981"


def _dark(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_FONT_COLOR, family=_FONT_FAMILY, size=12),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor=_GRID,
        zerolinecolor=_ZERO,
        tickfont=dict(color="#94A3B8"),
    )
    fig.update_yaxes(
        gridcolor=_GRID,
        zerolinecolor=_ZERO,
        tickfont=dict(color="#94A3B8"),
    )
    return fig


# ─── Numeric ──────────────────────────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogram + KDE and box plot for a numeric column."""
    data = df[col].dropna()
    if len(data) == 0:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"<b>{col}</b> — Distribution", f"<b>{col}</b> — Box Plot"],
        column_widths=[0.65, 0.35],
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data, name="Frequency",
            marker=dict(color=_PRIMARY, opacity=0.75, line=dict(width=0)),
            nbinsx=min(50, max(10, int(np.sqrt(len(data))))),
        ),
        row=1, col=1,
    )

    # KDE overlay
    try:
        kde_x = np.linspace(data.min(), data.max(), 300)
        kde = stats.gaussian_kde(data)
        bin_width = (data.max() - data.min()) / min(50, max(10, int(np.sqrt(len(data)))))
        kde_y = kde(kde_x) * len(data) * bin_width
        fig.add_trace(
            go.Scatter(
                x=kde_x, y=kde_y, name="KDE",
                line=dict(color=_SECONDARY, width=2.5),
                mode="lines",
            ),
            row=1, col=1,
        )
    except Exception:
        pass

    # Box plot
    fig.add_trace(
        go.Box(
            y=data, name=col,
            marker=dict(color=_PRIMARY, outliercolor=_DANGER, size=4),
            line=dict(color=_PRIMARY),
            fillcolor=f"rgba(108,99,255,0.2)",
            boxpoints="outliers",
            jitter=0.3,
        ),
        row=1, col=2,
    )

    fig.update_layout(showlegend=True, title_text="")
    return _dark(fig, height=380)


def plot_outliers(df: pd.DataFrame, col: str) -> go.Figure:
    """Violin + strip plot for outlier visualisation."""
    data = df[col].dropna()
    if len(data) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=data, name=col,
            box_visible=True,
            line_color=_PRIMARY,
            fillcolor=f"rgba(108,99,255,0.15)",
            meanline_visible=True,
            meanline=dict(color=_SECONDARY, width=2),
            points="outliers",
            marker=dict(color=_DANGER, size=4, opacity=0.7),
        )
    )

    # IQR bounds
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    for bound, label in [(lower, "IQR Lower"), (upper, "IQR Upper")]:
        fig.add_hline(
            y=bound, line_dash="dash",
            line_color=_ACCENT, opacity=0.6,
            annotation_text=label,
            annotation_font_color=_ACCENT,
        )

    fig.update_layout(title_text=f"<b>{col}</b> — Outlier Analysis")
    return _dark(fig, height=420)


# ─── Categorical ──────────────────────────────────────────────────────────────

def plot_categorical(df: pd.DataFrame, col: str, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart for a categorical column."""
    vc = df[col].value_counts().head(top_n)
    if len(vc) == 0:
        return go.Figure()

    # Gradient colours
    n = len(vc)
    colours = px.colors.sample_colorscale("Viridis", [i / max(n - 1, 1) for i in range(n)])

    fig = go.Figure(
        go.Bar(
            x=vc.values,
            y=vc.index.astype(str),
            orientation="h",
            marker=dict(color=colours),
            text=[f"{v:,}  ({v/len(df)*100:.1f}%)" for v in vc.values],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=11),
        )
    )

    fig.update_layout(
        title_text=f"<b>{col}</b> — Top {min(top_n, len(vc))} Values",
        xaxis_title="Count",
        yaxis=dict(autorange="reversed"),
    )
    return _dark(fig, height=max(350, len(vc) * 38))


# ─── Correlation ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> go.Figure | None:
    """Lower-triangle correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr(method=method).round(2)

    # Mask upper triangle (keep diagonal + lower)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig = go.Figure(
        go.Heatmap(
            z=corr_masked.values,
            x=list(corr_masked.columns),
            y=list(corr_masked.index),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1, zmax=1,
            text=corr_masked.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10, "color": _FONT_COLOR},
            hoverongaps=False,
            colorbar=dict(
                tickfont=dict(color=_FONT_COLOR),
                title=dict(text="r", font=dict(color=_FONT_COLOR)),
            ),
        )
    )
    fig.update_layout(title_text=f"<b>Correlation Matrix</b> ({method.title()})")
    return _dark(fig, height=max(420, len(corr) * 48))


# ─── Missing values ───────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> go.Figure | None:
    """Bar chart of missing value % per column."""
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]

    if len(null_pct) == 0:
        return None

    colours = [
        _SUCCESS if v < 5 else (_ACCENT if v < 20 else _DANGER)
        for v in null_pct.values
    ]

    fig = go.Figure(
        go.Bar(
            x=null_pct.index,
            y=null_pct.values,
            marker_color=colours,
            text=[f"{v:.1f}%" for v in null_pct.values],
            textposition="outside",
            textfont=dict(color="#94A3B8"),
        )
    )
    fig.update_layout(
        title_text="<b>Missing Values</b> by Column",
        yaxis_title="Missing %",
        xaxis_title="",
    )
    return _dark(fig, height=350)


# ─── Pairplot (for small datasets) ────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame, col_types: dict, target_col: str | None = None) -> go.Figure | None:
    """Scatter matrix for numeric columns (max 6 columns)."""
    num_cols = col_types.get("numeric", [])[:6]
    if len(num_cols) < 2:
        return None

    plot_df = df[num_cols].dropna()

    kwargs: dict = dict(
        dimensions=num_cols,
        opacity=0.6,
    )
    if target_col and target_col in df.columns:
        plot_df = plot_df.copy()
        plot_df[target_col] = df[target_col]
        kwargs["color"] = target_col

    fig = px.scatter_matrix(plot_df, **kwargs)
    fig.update_traces(
        marker=dict(size=3, line=dict(width=0)),
        diagonal_visible=False,
    )
    fig.update_layout(title_text="<b>Scatter Matrix</b>")
    return _dark(fig, height=600)


# ─── EDA stats ────────────────────────────────────────────────────────────────

def compute_eda_stats(df: pd.DataFrame, col_types: dict) -> dict:
    """Compute quick stats for each numeric column (for AI prompt context)."""
    result = {}
    for col in col_types.get("numeric", []):
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        result[col] = {
            "skewness": round(float(clean.skew()), 4),
            "kurtosis": round(float(clean.kurt()), 4),
            "outlier_pct": round(
                ((clean < clean.quantile(0.25) - 1.5 * (clean.quantile(0.75) - clean.quantile(0.25))) |
                 (clean > clean.quantile(0.75) + 1.5 * (clean.quantile(0.75) - clean.quantile(0.25)))).sum()
                / max(len(df), 1) * 100, 2
            ),
        }
    return result
