"""
dimensionality.py — Local PCA analysis & visualization helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca_analysis(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """
    Applies PCA on numerical columns and generates beautiful 2D and 3D projection plots.
    """
    # Select only numeric columns
    num_df = df.select_dtypes(include=[np.number]).copy()
    if target_col and target_col in num_df.columns:
        num_df = num_df.drop(columns=[target_col])

    # Fill NaNs
    num_df = num_df.fillna(num_df.median())

    if num_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for PCA projection."}

    # Scale the features
    scaled_data = StandardScaler().fit_transform(num_df)

    # Perform PCA
    n_components = min(3, num_df.shape[1])
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_data)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # DataFrame for projection charts
    proj_df = pd.DataFrame(
        components,
        columns=[f"PC {i+1} ({var*100:.1f}%)" for i, var in enumerate(explained_var)],
    )

    color_col = None
    if target_col and target_col in df.columns:
        proj_df["Target"] = df[target_col].values
        color_col = "Target"

    # Create 2D projection plot
    pc1_col = proj_df.columns[0]
    pc2_col = proj_df.columns[1]

    fig_2d = px.scatter(
        proj_df,
        x=pc1_col,
        y=pc2_col,
        color=color_col,
        title="2D PCA Space Projection",
        template="plotly_dark",
        color_continuous_scale="Viridis",
    )
    fig_2d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    )

    # Create 3D projection plot if possible
    fig_3d = None
    if n_components >= 3:
        pc3_col = proj_df.columns[2]
        fig_3d = px.scatter_3d(
            proj_df,
            x=pc1_col,
            y=pc2_col,
            z=pc3_col,
            color=color_col,
            title="3D PCA Space Projection",
            template="plotly_dark",
            color_continuous_scale="Viridis",
        )
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0", family="Inter, sans-serif"),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            ),
        )

    # Create Variance Explained Bar Chart
    var_df = pd.DataFrame({
        "Component": [f"PC {i+1}" for i in range(n_components)],
        "Variance Explained": explained_var,
        "Cumulative Variance": cumulative_var,
    })

    fig_var = px.bar(
        var_df,
        x="Component",
        y="Variance Explained",
        title="Explained Variance by Principal Component",
        template="plotly_dark",
        text_auto=".2%",
    )
    fig_var.update_traces(marker_color="#6C63FF")
    fig_var.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    )

    return {
        "fig_2d": fig_2d,
        "fig_3d": fig_3d,
        "fig_var": fig_var,
        "explained_var": explained_var.tolist(),
        "cumulative_var": cumulative_var.tolist(),
    }
