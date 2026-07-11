"""
app.py — DataMind: Ultimate AI Data Analyst v1.0
Powerful local data analyst with Time Series, advanced visualizations,
HTML reports, and more.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from core.cleaner import apply_cleaning, get_auto_cleaning_config, get_cleaning_diff
from core.data_loader import detect_column_types, load_data
from core.dimensionality import run_pca_analysis
from core.eda import (
    plot_categorical,
    plot_correlation_heatmap,
    plot_distribution,
    plot_missing_values,
    plot_outliers,
    plot_pairplot,
)
from core.feature_engineer import (
    apply_advanced_transforms,
    apply_feature_engineering,
    get_auto_engineering_config,
    get_smart_suggestions,
)
from core.feature_help import render_help_expander
from core.ml_readiness import compute_ml_readiness
from core.profiler import generate_profile
from core.report_generator import generate_html_report
from core.notebook_exporter import generate_notebook
from core.time_series import (
    compute_ts_summary,
    decompose_trend_seasonality,
    detect_time_series_columns,
    infer_frequency,
    prepare_time_series,
    suggest_lag_features,
)
from core.visualizations import (
    compute_column_health,
    plot_3d_scatter,
    plot_confusion_matrix,
    plot_enhanced_heatmap,
    plot_partial_dependence,
    plot_roc_curves,
    plot_rolling_stats,
    plot_scatter_matrix,
    plot_time_series,
    plot_violin,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMind — Ultimate AI Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
_CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State initialization
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "df_raw": None,
    "df_clean": None,
    "df_engineered": None,
    "col_types": None,
    "profile": None,
    "cleaning_config": None,
    "engineering_config": None,
    "adv_config": {
        "drop_cols": [], "log_transform": [], "sqrt_transform": [],
        "square_transform": [], "interaction_terms": [], "custom_formulas": [], "binning": [],
    },
    "readiness": None,
    "cleaning_changelog": [],
    "engineering_changelog": [],
    "target_col": None,
    "filename": None,
    "smart_suggestions": None,
    "ml_results": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_final_df() -> pd.DataFrame:
    if st.session_state.df_engineered is not None:
        return st.session_state.df_engineered
    if st.session_state.df_clean is not None:
        return st.session_state.df_clean
    return st.session_state.df_raw


def _load_dataset(uploaded_file):
    df = load_data(uploaded_file)
    col_types = detect_column_types(df)
    profile = generate_profile(df, col_types)
    st.session_state.df_raw = df
    st.session_state.df_clean = df.copy()
    st.session_state.df_engineered = None
    st.session_state.col_types = col_types
    st.session_state.profile = profile
    st.session_state.filename = uploaded_file.name
    st.session_state.cleaning_config = get_auto_cleaning_config(df, col_types)
    st.session_state.engineering_config = get_auto_engineering_config(df, col_types)
    st.session_state.adv_config = {
        "drop_cols": [], "log_transform": [], "sqrt_transform": [],
        "square_transform": [], "interaction_terms": [], "custom_formulas": [], "binning": [],
    }
    st.session_state.cleaning_changelog = []
    st.session_state.engineering_changelog = []
    st.session_state.readiness = None
    st.session_state.smart_suggestions = get_smart_suggestions(df, col_types)
    st.session_state.ml_results = None


def _download_df(df: pd.DataFrame, label: str, base_name: str):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"⬇️  Download {label} (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"datamind_{base_name}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_csv_{base_name}",
        )
    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Data")
        st.download_button(
            f"⬇️  Download {label} (Excel)",
            data=buf.getvalue(),
            file_name=f"datamind_{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"dl_xlsx_{base_name}",
        )


def _sev_icon(severity: str) -> str:
    return {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="logo-icon">🧠</div>
            <div class="logo-text">DataMind</div>
            <div class="logo-subtitle">Ultimate AI Data Analyst v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ── File upload ────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Upload CSV, Parquet, JSON, or Excel",
        label_visibility="collapsed",
    )
    render_help_expander(st, "file_upload", "ℹ️ Supported formats & tips")

    if uploaded:
        if st.session_state.filename != uploaded.name:
            with st.spinner("Processing & profiling..."):
                try:
                    _load_dataset(uploaded)
                    st.success(f"✓ loaded: {st.session_state.df_raw.shape[0]:,} rows")
                except Exception as exc:
                    st.error(f"❌ {exc}")

    if os.path.exists("sample_housing_data.csv"):
        if st.button("💡 Load Sample Housing Data", use_container_width=True):
            with st.spinner("Loading synthetic housing data..."):
                class DummyUploadedFile(io.BytesIO):
                    def __init__(self, name, data):
                        super().__init__(data)
                        self.name = name
                try:
                    with open("sample_housing_data.csv", "rb") as f:
                        file_bytes = f.read()
                    dummy = DummyUploadedFile("sample_housing_data.csv", file_bytes)
                    _load_dataset(dummy)
                    st.success("✓ Loaded sample dataset!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error loading sample: {exc}")

    st.markdown("---")

    if st.session_state.df_raw is not None:
        # ── Target selection ───────────────────────────────────────────────
        st.markdown("### 🎯 Model Target (Optional)")
        options = ["— None —"] + list(st.session_state.df_raw.columns)
        target_idx = 0
        if st.session_state.target_col and st.session_state.target_col in options:
            target_idx = options.index(st.session_state.target_col)

        selected = st.selectbox(
            "Target column for analysis", options, index=target_idx,
            help="Setting a target enables correlation analysis and ML-Readiness scoring.",
        )
        st.session_state.target_col = None if selected == "— None —" else selected
        render_help_expander(st, "target_col", "ℹ️ How to choose a target")

        # ── Decision Helper ────────────────────────────────────────────────
        st.markdown("### 💡 Strategy Helper")
        with st.expander("📖 Decision Guide: What to choose?", expanded=False):
            st.markdown(
                """
                **🧹 Data Cleaning**
                - **Nulls**: Use *Median* for skewed numeric data (e.g. Price), *Mean* for normal numeric data, *Mode* for category columns.
                - **Outliers**: Use *Cap (IQR)* to keep rows but bring errors within range. Use *Remove* only if values are completely impossible (e.g. Age=999).
                
                **⚙️ Feature Engineering**
                - **Encoding**: Use *One-Hot* for linear/distance models (Linear Regression), *Label* for tree models (Random Forest).
                - **Scaling**: Use *Robust* if features have outliers/skewness. Use *Standard* for normally distributed features.
                - **Skewness**: Use *Log1p* if column skewness > 1.0. Use *Sqrt* for moderate skewness.
                
                **📈 Time Series**
                - **Windows**: 7 is good for daily cycles, 30 for monthly, 12 for yearly.
                - **Stationarity**: Detrend/difference data before passing to standard ML models.
                
                **🎯 ML Readiness Score**
                - The Report tab scores your data for completeness, variance, encoding, and balance.
                - Aim for 75+ before handing cleaned data off to model training.
                """
            )

        st.markdown("---")

        # ── Pipeline status ────────────────────────────────────────────────
        st.markdown("### 📍 Pipeline Progress")
        steps = [
            ("Profiler", st.session_state.profile is not None),
            ("EDA & PCA", st.session_state.profile is not None),
            ("Time Series", any(
                c in (st.session_state.col_types or {}).get("datetime", [])
                for c in st.session_state.df_raw.columns
            )),
            ("Data Cleaning", len(st.session_state.cleaning_changelog) > 0),
            ("Feature Engineering", st.session_state.df_engineered is not None),
            ("Report & Export", st.session_state.readiness is not None),
        ]
        for step_label, done in steps:
            icon = "✅" if done else "⏳"
            st.markdown(
                f'<div class="step-item {"done" if done else ""}">'
                f'<span class="step-dot"></span>{icon} {step_label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Quick download ─────────────────────────────────────────────────
        st.markdown("### 📥 Quick Download")
        base = (st.session_state.filename or "data").rsplit(".", 1)[0]
        final = _get_final_df()
        if final is not None:
            stage = ("engineered" if st.session_state.df_engineered is not None
                     else "cleaned" if st.session_state.df_clean is not None else "raw")
            st.download_button(
                f"⬇️ Download {stage} CSV",
                data=final.to_csv(index=False).encode("utf-8"),
                file_name=f"datamind_{base}_{stage}.csv",
                mime="text/csv",
                use_container_width=True,
                key="sidebar_dl_csv",
            )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer">DataMind v1.0 · Ultimate AI Data Analyst</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df_raw is None:
    st.markdown(
        """
        <div class="landing-hero">
            <div class="landing-badge">🚀 v1.0 · Time Series · Advanced EDA · HTML Reports · Data Cleaning</div>
            <div class="landing-title">DataMind</div>
            <div class="landing-subtitle">The Ultimate Data Analyst Agent</div>
            <div class="landing-desc">
                Upload any dataset for instant profiling, intelligent cleaning,
                advanced EDA, time series analysis, feature engineering, and 
                beautiful report exports — all locally.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists("sample_housing_data.csv"):
        st.markdown("<div style='text-align:center; margin-bottom:25px;'>", unsafe_allow_html=True)
        if st.button("🚀 Try it! Load Sample Housing Dataset", type="primary", use_container_width=False):
            with st.spinner("Loading synthetic housing data..."):
                class DummyUploadedFile(io.BytesIO):
                    def __init__(self, name, data):
                        super().__init__(data)
                        self.name = name
                try:
                    with open("sample_housing_data.csv", "rb") as f:
                        file_bytes = f.read()
                    dummy = DummyUploadedFile("sample_housing_data.csv", file_bytes)
                    _load_dataset(dummy)
                    st.success("✓ Loaded sample dataset!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error loading sample: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    cards = [
        ("📊", "Smart Auto Profiling",
         "Statistical type-detection, quality scoring, per-column health bars, and intelligent issue alerts.",
         "file_upload", "How to use: Upload any CSV/Excel/JSON/Parquet → Overview tab shows instant analysis"),
        ("🔍", "Advanced EDA & PCA",
         "Violin plots, scatter matrix, 3D scatter, correlation drill-down, and PCA dimensionality reduction.",
         "pearson_corr", "How to use: Go to EDA tab → Choose analysis type → Click any correlation cell to drill down"),
        ("📈", "Time Series Analysis",
         "Auto-detect datetime columns, rolling statistics, trend decomposition, stationarity testing, and lag features.",
         "rolling_mean", "How to use: Load data with a date column → Time Series tab → Select date & value columns"),
        ("🧹", "Intelligent Data Cleaning",
         "IQR outlier capping, smart null imputation, duplicate removal — all with contextual guidance for each option.",
         "null_median", "How to use: Clean tab → Each column shows 'ℹ️ What is this?' for every strategy choice"),
        ("⚙️", "Feature Engineering Studio",
         "Encoding, scaling, datetime decomposition, log/sqrt/power transforms, interaction terms, custom formulas.",
         "log_transform", "How to use: Feature Engineer tab → Every control has a help expander explaining when to use it"),
        ("📋", "Data Readiness Report",
         "Dimensional scorecard + one-click HTML report export with all charts, stats, and pipeline logs.",
         "ml_readiness", "How to use: Report tab → Generate Scorecard → Export Full HTML Report → Share with anyone"),
    ]

    rows = [cards[i:i+3] for i in range(0, len(cards), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (icon, title, desc, help_key, how_to) in zip(cols, row):
            with col:
                st.markdown(
                    f"""<div class="feature-card">
                        <span class="feature-icon">{icon}</span>
                        <div class="feature-title">{title}</div>
                        <div class="feature-desc">{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                with st.expander("ℹ️ How to use this feature"):
                    st.markdown(f"**{how_to}**")
                    render_help_expander(st, help_key, "📖 Full feature documentation")

        st.markdown("<br>", unsafe_allow_html=True)

    # Tutorial
    st.markdown("---")
    st.markdown("### 📖 Quick Start Guide")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
            #### Recommended Pipeline
            1. **Upload** your dataset (or load the sample)
            2. **Overview** → Review quality score and health bars
            3. **EDA** → Explore distributions and correlations
            4. **Time Series** → Analyze trends (if you have dates)
            5. **Clean** → Fix nulls and outliers (use ℹ️ guides!)
            6. **Feature Engineer** → Transform and encode features
            7. **Report** → Generate and download HTML report & export data
            """
        )
    with col_t2:
        st.markdown(
            """
            #### Sample Housing Dataset
            When you load it, you'll find real-world data issues:
            - `Rooms` → **10 missing values**
            - `SquareFootage` → **outliers** including a -100 error
            - `Date_Listed` → datetime for time series & feature extraction
            - `Neighborhood` → categorical for encoding
            - `SalePrice` → use as **regression target**
            - `Is_Premium` → use as **classification target** (binary)
            """
        )

    st.markdown(
        """
        > **💡 Tip**: Every control in DataMind has a built-in **ℹ️ help expander** that explains
        > *what it does*, *when to use it*, and gives real *examples*. Look for them throughout the app!
        """
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
df: pd.DataFrame = st.session_state.df_raw
profile: dict = st.session_state.profile
col_types: dict = st.session_state.col_types
base = (st.session_state.filename or "data").rsplit(".", 1)[0]

st.markdown(
    f"""
    <div class="page-header">
        <div>
            <div class="page-title">🧠 DataMind Workstation</div>
            <div style="font-size:0.75rem;color:#475569;margin-top:3px;font-weight:500;">
                Data Analysis Pipeline · Profiling · EDA · Cleaning · Feature Engineering
            </div>
        </div>
        <div class="page-filename">📂 {st.session_state.filename}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Summary metrics ──────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
qs = profile["quality_score"]
with m1:
    st.metric("Rows", f"{profile['n_rows']:,}")
with m2:
    st.metric("Columns", profile["n_cols"])
with m3:
    st.metric("Missing Cells", f"{profile['total_nulls']:,}",
              delta=f"{profile['null_pct']:.1f}%", delta_color="inverse")
with m4:
    st.metric("Duplicates", f"{profile['duplicate_rows']:,}",
              delta=f"{profile['duplicate_pct']:.1f}%", delta_color="inverse")
with m5:
    st.metric("Dataset Quality", f"{qs}/100",
              delta="Healthy" if qs >= 70 else "Impaired",
              delta_color="normal" if qs >= 70 else "inverse")

# ── Dataset Quick Insights Bar ────────────────────────────────────────────────
_numeric_cols = col_types.get("numeric", [])
_cat_cols = col_types.get("categorical", [])
_dt_cols = col_types.get("datetime", [])
_bool_cols = col_types.get("bool", [])
_qs = profile["quality_score"]
_qs_color = "#10B981" if _qs >= 80 else "#F59E0B" if _qs >= 60 else "#EF4444"
_has_missing = profile['total_nulls'] > 0
_has_dups = profile['duplicate_rows'] > 0

st.markdown(
    f"""
    <div class="insights-bar">
        <div class="insight-chip">
            <span class="insight-icon">🔢</span>
            <span class="insight-label">Numeric</span>
            <span class="insight-val">{len(_numeric_cols)}</span>
        </div>
        <div class="insight-chip">
            <span class="insight-icon">🏷️</span>
            <span class="insight-label">Categorical</span>
            <span class="insight-val">{len(_cat_cols)}</span>
        </div>
        <div class="insight-chip">
            <span class="insight-icon">📅</span>
            <span class="insight-label">Datetime</span>
            <span class="insight-val">{len(_dt_cols)}</span>
        </div>
        <div class="insight-chip {'insight-chip--warn' if _has_missing else 'insight-chip--ok'}">
            <span class="insight-icon">{'⚠️' if _has_missing else '✅'}</span>
            <span class="insight-label">Missing</span>
            <span class="insight-val">{profile['total_nulls']:,}</span>
        </div>
        <div class="insight-chip {'insight-chip--warn' if _has_dups else 'insight-chip--ok'}">
            <span class="insight-icon">{'⚠️' if _has_dups else '✅'}</span>
            <span class="insight-label">Duplicates</span>
            <span class="insight-val">{profile['duplicate_rows']:,}</span>
        </div>
        <div class="insight-chip insight-chip--quality">
            <span class="insight-icon">🎯</span>
            <span class="insight-label">Quality</span>
            <span class="insight-val" style="color:{_qs_color};">{_qs}/100</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Nav Chips ─────────────────────────────────────────────────────────────────
_nav_options = [
    "📊  Overview",
    "🔍  EDA & PCA",
    "📈  Time Series",
    "🧹  Clean",
    "⚙️  Feature Engineer",
    "📋  Report & Export",
]
_active_nav = st.radio(
    "Navigation",
    _nav_options,
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav",
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if _active_nav == "📊  Overview":

    # ── Smart Alerts ──────────────────────────────────────────────────────────
    suggestions = st.session_state.smart_suggestions or []
    if suggestions:
        st.markdown("#### 🧠 Smart Scan Alerts")
        sc1, sc2 = st.columns(2)
        for i, s in enumerate(suggestions):
            col_target = sc1 if i % 2 == 0 else sc2
            with col_target:
                sev = s["severity"]
                icon = _sev_icon(sev)
                border = "#EF4444" if sev == "error" else "#F59E0B" if sev == "warning" else "#6C63FF"
                st.markdown(
                    f'<div class="alert-card" style="padding:10px;border-radius:8px;'
                    f'margin-bottom:8px;border:1px solid {border};'
                    f'background:rgba(255,255,255,0.01);">{icon} {s["message"]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.success("🎉 Smart analysis found no issues with this dataset!")

    st.markdown("---")

    # ── Column Health Dashboard ───────────────────────────────────────────────
    st.markdown("#### 💊 Column Health Dashboard")
    health_data = compute_column_health(profile, col_types)

    # Sort: worst first
    health_sorted = sorted(health_data, key=lambda x: x["health_score"])
    h_cols = st.columns(min(4, len(health_sorted)))
    shown = 0
    for h in health_sorted:
        if shown >= 12:
            break
        hc = h_cols[shown % len(h_cols)]
        score = h["health_score"]
        color = "#10B981" if score >= 85 else "#F59E0B" if score >= 60 else "#EF4444"
        issues_str = " · ".join(h["issues"]) if h["issues"] else "Healthy ✓"
        with hc:
            st.markdown(
                f"""<div class="health-bar-card">
                    <div class="hb-col-name">{h['col']}</div>
                    <div class="hb-track">
                        <div class="hb-fill" style="width:{score}%;background:{color};"></div>
                    </div>
                    <div class="hb-meta" style="color:{color};">{score:.0f} — {issues_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        shown += 1

    st.markdown("---")

    # ── Preview + Column types ────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### 📋 Dataset Preview (First 50 Rows)")
        st.dataframe(df.head(50), use_container_width=True, height=350)

    with col_right:
        st.markdown("#### 📦 Column Type Distribution")
        type_rows = [
            {"Category": "🔢 Numeric", "Count": len(col_types.get("numeric", []))},
            {"Category": "🔤 Categorical", "Count": len(col_types.get("categorical", []))},
            {"Category": "📅 Datetime", "Count": len(col_types.get("datetime", []))},
            {"Category": "☑️ Boolean", "Count": len(col_types.get("boolean", []))},
            {"Category": "📝 Text", "Count": len(col_types.get("text", []))},
        ]
        type_df = pd.DataFrame([r for r in type_rows if r["Count"] > 0])
        st.dataframe(type_df, hide_index=True, use_container_width=True)

        st.markdown("#### 📊 Storage Summary")
        q_grade = "Good" if qs >= 80 else "Fair" if qs >= 60 else "Poor"
        for label, val in [
            ("RAM footprint", f"{profile['memory_mb']:.2f} MB"),
            ("Total values", f"{profile['n_rows'] * profile['n_cols']:,}"),
            ("Quality grade", q_grade),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:8px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.85rem;">'
                f'<span style="color:#64748B;">{label}</span>'
                f'<span style="color:#E2E8F0;font-weight:600;">{val}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Detailed column stats ─────────────────────────────────────────────────
    st.markdown("#### 🔬 Detailed Column Statistics")
    rows_stat = []
    for col_name, info in profile["columns"].items():
        ct = next((t for t, cols in col_types.items() if col_name in cols), "—")
        row: dict = {
            "Column": col_name, "Type": ct, "Dtype": info["dtype"],
            "Null %": f"{info['null_pct']:.1f}%",
            "Unique": f"{info['unique_count']:,}",
            "Unique %": f"{info['unique_pct']:.1f}%",
        }
        if "mean" in info and info["mean"] is not None:
            row["Mean"] = f"{info['mean']:.4g}"
            row["Std"] = f"{info['std']:.4g}"
            row["Skewness"] = f"{info['skewness']:.3f}"
            row["Outliers"] = str(info.get("outlier_count", 0))
        rows_stat.append(row)
    st.dataframe(pd.DataFrame(rows_stat), hide_index=True, use_container_width=True, height=350)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & PCA
# ══════════════════════════════════════════════════════════════════════════════
if _active_nav == "🔍  EDA & PCA":
    numeric_cols = col_types.get("numeric", [])
    cat_cols = col_types.get("categorical", [])

    eda_mode = st.radio(
        "Select Analysis Type",
        ["📊 Classical EDA", "🎻 Advanced Visuals", "🧬 PCA Dimensionality Reduction"],
        horizontal=True,
    )

    if eda_mode == "📊 Classical EDA":

        # Correlation Matrix
        if len(numeric_cols) >= 2:
            st.markdown("#### 🔥 Correlation Matrix")
            corr_col1, corr_col2 = st.columns([3, 1])
            with corr_col1:
                corr_method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="corr_method_eda")
                render_help_expander(st, f"{corr_method}_corr", "ℹ️ Pearson vs Spearman explained")
            with corr_col2:
                show_drill = st.checkbox("Enable drill-down on click", value=True)

            enh_fig, corr_matrix = plot_enhanced_heatmap(df, method=corr_method)
            st.plotly_chart(enh_fig, use_container_width=True)

            if show_drill and len(numeric_cols) >= 2:
                st.markdown("**🔍 Correlation Pair Drill-Down**")
                dd1, dd2 = st.columns(2)
                with dd1:
                    drill_a = st.selectbox("Column A", numeric_cols, key="drill_a")
                with dd2:
                    drill_b = st.selectbox("Column B", [c for c in numeric_cols if c != drill_a], key="drill_b")

                if drill_a and drill_b:
                    corr_val = corr_matrix.loc[drill_a, drill_b] if drill_a in corr_matrix.index and drill_b in corr_matrix.columns else 0
                    scatter_fig = px.scatter(
                        df, x=drill_a, y=drill_b,
                        color=st.session_state.target_col if st.session_state.target_col and st.session_state.target_col in df.columns else None,
                        color_discrete_sequence=["#6C63FF"],
                        color_continuous_scale="Viridis",
                        template="plotly_dark",
                        title=f"Scatter: {drill_a} vs {drill_b} (r={corr_val:.3f})",
                        opacity=0.65,
                    )
                    scatter_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                              font=dict(color="#E2E8F0"))
                    st.plotly_chart(scatter_fig, use_container_width=True)

        # Target correlation
        if st.session_state.target_col and len(numeric_cols) >= 2:
            st.markdown("#### 🎯 Feature Correlation with Target")
            target = st.session_state.target_col
            num_df = df.select_dtypes(include=[np.number])
            if target in num_df.columns:
                corr_series = num_df.corr()[target].drop(target).abs().sort_values(ascending=True)
                fig_ta = go.Figure(go.Bar(
                    x=corr_series.values, y=corr_series.index, orientation="h",
                    marker_color=["#10B981" if v >= 0.5 else "#6C63FF" if v >= 0.2 else "#475569"
                                  for v in corr_series.values],
                    text=[f"{v:.3f}" for v in corr_series.values],
                    textposition="outside",
                ))
                fig_ta.update_layout(
                    title_text=f"Correlation with target: <b>{target}</b>",
                    xaxis_title="|Pearson r|", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                )
                fig_ta.update_xaxes(gridcolor="rgba(255,255,255,0.05)", range=[0, 1.1])
                st.plotly_chart(fig_ta, use_container_width=True)

        # Distributions
        if numeric_cols:
            st.markdown("#### 📈 Distribution & Outliers Inspector")
            sel_num = st.selectbox("Select column to inspect", numeric_cols, key="inspect_num")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_distribution(df, sel_num), use_container_width=True)
            with c2:
                st.plotly_chart(plot_outliers(df, sel_num), use_container_width=True)

        # Categorical
        if cat_cols:
            st.markdown("#### 📊 Categorical Distribution")
            sel_cat = st.selectbox("Select column to inspect", cat_cols, key="inspect_cat")
            st.plotly_chart(plot_categorical(df, sel_cat), use_container_width=True)

        # Missing values heatmap
        if profile["total_nulls"] > 0:
            st.markdown("#### 🕳️ Missing Values Heatmap")
            fig_miss = plot_missing_values(df)
            if fig_miss:
                st.plotly_chart(fig_miss, use_container_width=True)

    elif eda_mode == "🎻 Advanced Visuals":
        st.markdown("#### 🎻 Advanced Visualization Suite")

        adv_viz = st.selectbox(
            "Choose visualization type",
            ["Violin Plot", "Scatter Matrix", "3D Scatter", "Pair Plot"],
            key="adv_viz_type",
        )

        if adv_viz == "Violin Plot" and numeric_cols:
            v1, v2 = st.columns(2)
            with v1:
                vc = st.selectbox("Column to visualize", numeric_cols, key="violin_col")
            with v2:
                vg = st.selectbox("Group by (optional)", ["— None —"] + cat_cols, key="violin_group")
            group_col = None if vg == "— None —" else vg
            st.plotly_chart(plot_violin(df, vc, group_col), use_container_width=True)

        elif adv_viz == "Scatter Matrix" and len(numeric_cols) >= 2:
            sm_cols = st.multiselect(
                "Select columns (2–6 recommended)",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
                key="scatter_matrix_cols",
            )
            sm_color = st.selectbox("Color by", ["— None —"] + cat_cols + [st.session_state.target_col or ""],
                                    key="scatter_matrix_color")
            color_col = None if sm_color == "— None —" or not sm_color else sm_color
            if len(sm_cols) >= 2:
                st.plotly_chart(plot_scatter_matrix(df, sm_cols, color_col), use_container_width=True)

        elif adv_viz == "3D Scatter" and len(numeric_cols) >= 3:
            s1, s2, s3 = st.columns(3)
            with s1:
                x3 = st.selectbox("X axis", numeric_cols, key="3d_x")
            with s2:
                y3 = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="3d_y")
            with s3:
                z3 = st.selectbox("Z axis", numeric_cols, index=min(2, len(numeric_cols)-1), key="3d_z")
            color3 = st.selectbox("Color by", ["— None —"] + cat_cols, key="3d_color")
            color_col3 = None if color3 == "— None —" else color3
            st.plotly_chart(plot_3d_scatter(df, x3, y3, z3, color_col3), use_container_width=True)

        elif adv_viz == "Pair Plot" and len(numeric_cols) >= 2:
            pp_cols = numeric_cols[:min(5, len(numeric_cols))]
            fig_pp = plot_pairplot(df, pp_cols, st.session_state.target_col)
            if fig_pp:
                st.plotly_chart(fig_pp, use_container_width=True)

    else:  # PCA
        st.markdown("#### 🧬 Principal Component Analysis (PCA)")
        st.markdown(
            "Project high-dimensional numerical columns onto 2D or 3D coordinate space "
            "to discover hidden clusters, separation thresholds, and data density patterns."
        )
        render_help_expander(st, "pca_2d", "ℹ️ How to read PCA plots")

        pca_results = run_pca_analysis(df, st.session_state.target_col)
        if "error" in pca_results:
            st.warning(pca_results["error"])
        else:
            pc_col1, pc_col2 = st.columns([2, 1])
            with pc_col1:
                pca_dim = st.radio("Projection dimensions", ["2D Projection", "3D Projection"], horizontal=True)
                if pca_dim == "2D Projection":
                    st.plotly_chart(pca_results["fig_2d"], use_container_width=True)
                    render_help_expander(st, "pca_2d", "ℹ️ Understanding 2D PCA")
                else:
                    if pca_results["fig_3d"] is not None:
                        st.plotly_chart(pca_results["fig_3d"], use_container_width=True)
                        render_help_expander(st, "pca_3d", "ℹ️ Understanding 3D PCA")
                    else:
                        st.info("3D projection requires at least 3 numerical columns.")
            with pc_col2:
                st.plotly_chart(pca_results["fig_var"], use_container_width=True)
                render_help_expander(st, "pca_variance", "ℹ️ Explained variance guide")
                st.markdown("**Component Cumulative Variance**")
                for idx, val in enumerate(pca_results["cumulative_var"]):
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span style="color:#94A3B8;">PC 1 to PC {idx+1}</span>'
                        f'<span style="color:#10B981;font-weight:600;">{val*100:.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
if _active_nav == "📈  Time Series":
    st.markdown("#### 📈 Time Series Analysis")
    dt_cols = detect_time_series_columns(df, col_types)
    numeric_ts_cols = col_types.get("numeric", [])

    if not dt_cols:
        st.info(
            "No datetime columns detected. To use time series analysis, "
            "make sure your dataset has a column with dates/timestamps "
            "(e.g. `Date`, `Timestamp`, `Date_Listed`)."
        )
        st.markdown(
            """
            **💡 Tip**: If your dates are stored as strings, the system will try to auto-parse them.
            Common date formats are auto-detected (YYYY-MM-DD, MM/DD/YYYY, etc.).
            """
        )
    else:
        ts1, ts2, ts3 = st.columns(3)
        with ts1:
            date_col = st.selectbox("📅 Date / Time column", dt_cols, key="ts_date_col")
        with ts2:
            value_col = st.selectbox("📊 Value column", numeric_ts_cols, key="ts_val_col")
        with ts3:
            window_size = st.number_input("Rolling window size", min_value=2, max_value=90, value=7, key="ts_window")
            render_help_expander(st, "rolling_mean", "ℹ️ What is rolling window?")

        if date_col and value_col:
            # Frequency detection
            freq = infer_frequency(df, date_col)
            st.markdown(
                f'<div class="info-box">📊 Detected frequency: <strong>{freq}</strong> · '
                f'Date column: <strong>{date_col}</strong> · Value: <strong>{value_col}</strong></div>',
                unsafe_allow_html=True,
            )

            # ── Time Series Summary ───────────────────────────────────────────
            ts_sum = compute_ts_summary(df, date_col, value_col)

            if "error" not in ts_sum:
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Data Points", f"{ts_sum['n_points']:,}")
                with s2:
                    st.metric("Trend", ts_sum["trend"]["direction"])
                with s3:
                    adf_res = ts_sum.get("stationarity", {})
                    stat_label = "Stationary ✓" if adf_res.get("is_stationary") else "Non-Stationary ⚠️"
                    stat_delta_color = "normal" if adf_res.get("is_stationary") else "inverse"
                    st.metric("Stationarity", stat_label)
                with s4:
                    cv = ts_sum["value_stats"].get("cv_pct", 0)
                    st.metric("Coefficient of Variation", f"{cv:.1f}%")

                render_help_expander(st, "adf_test", "ℹ️ What is the stationarity test?")

            # ── Main Time Series Chart ────────────────────────────────────────
            st.markdown("#### 📉 Time Series Chart")
            show_roll = st.checkbox("Show rolling mean overlay", value=True, key="ts_show_roll")
            ts_fig = plot_time_series(df, date_col, value_col, window=window_size, show_rolling=show_roll)
            st.plotly_chart(ts_fig, use_container_width=True)

            # ── Rolling Statistics ────────────────────────────────────────────
            st.markdown("#### 📊 Rolling Statistics (Mean & Volatility)")
            render_help_expander(st, "rolling_std", "ℹ️ What does rolling std tell you?")
            roll_fig = plot_rolling_stats(df, date_col, value_col, window=window_size)
            st.plotly_chart(roll_fig, use_container_width=True)

            # ── Stationarity Details ──────────────────────────────────────────
            if "error" not in ts_sum:
                adf_res = ts_sum.get("stationarity", {})
                if adf_res and "error" not in adf_res:
                    with st.expander("📋 Stationarity Test Details", expanded=False):
                        if adf_res.get("p_value") is not None:
                            p = adf_res["p_value"]
                            stat = adf_res["statistic"]
                            if adf_res["is_stationary"]:
                                st.success(f"✅ **Stationary** — ADF statistic: {stat:.4f} · p-value: {p:.4f} (< 0.05)")
                                st.markdown("The series has a constant mean and variance over time. Safe for most ML models.")
                            else:
                                st.warning(f"⚠️ **Non-Stationary** — ADF statistic: {stat:.4f} · p-value: {p:.4f} (≥ 0.05)")
                                st.markdown(
                                    "The series has a trend or seasonality. Consider:\n"
                                    "- **Differencing**: subtract previous value\n"
                                    "- **Detrending**: subtract rolling mean\n"
                                    "- **Log transform**: reduces exponential growth"
                                )
                        else:
                            note = adf_res.get("note", "")
                            st.info(f"Variance ratio test used (statsmodels not installed). {note}")

            # ── Trend Decomposition ───────────────────────────────────────────
            st.markdown("#### 🔀 Trend & Seasonality Decomposition")
            ts_prepared = prepare_time_series(df, date_col, value_col)
            period_default = max(2, min(12, len(ts_prepared) // 4))
            decomp_period = st.number_input(
                "Seasonal period (e.g. 7=weekly, 12=monthly)",
                min_value=2, max_value=max(2, len(ts_prepared)//2),
                value=period_default, key="decomp_period"
            )

            if st.button("🔀 Decompose Trend & Seasonality", key="decomp_btn"):
                decomp = decompose_trend_seasonality(ts_prepared[value_col], period=int(decomp_period))
                if "error" in decomp:
                    st.warning(decomp["error"])
                else:
                    dates = ts_prepared[date_col].values
                    from plotly.subplots import make_subplots
                    decomp_fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=["Trend", "Seasonality", "Residuals"],
                        shared_xaxes=True, vertical_spacing=0.08,
                    )
                    pairs = [
                        (decomp["trend"], "#3ECFCF", 1),
                        (decomp["seasonal"], "#F59E0B", 2),
                        (decomp["residual"], "#EF4444", 3),
                    ]
                    for series_data, color, row in pairs:
                        if series_data is not None:
                            decomp_fig.add_trace(
                                go.Scatter(x=dates[:len(series_data)], y=series_data.values,
                                           line=dict(color=color, width=2), showlegend=False),
                                row=row, col=1
                            )
                    decomp_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                        height=500,
                    )
                    st.plotly_chart(decomp_fig, use_container_width=True)
                    st.info(f"Method: {decomp.get('method', 'unknown')} · Period: {decomp.get('period', '?')}")

            # ── Lag Feature Suggestions ───────────────────────────────────────
            with st.expander("⏱️ Lag Feature Analysis (Autocorrelation)", expanded=False):
                st.markdown("Identifies which lag values are most correlated with the current value — useful for creating lag features.")
                lag_results = suggest_lag_features(df, date_col, value_col, max_lag=14)
                if lag_results:
                    lag_df = pd.DataFrame([
                        {"Lag (periods)": lag, "Autocorrelation": corr,
                         "Strength": "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.2 else "Weak"}
                        for lag, corr in lag_results.items()
                    ])
                    st.dataframe(lag_df, hide_index=True, use_container_width=True)
                    top_lags = list(lag_results.keys())[:3]
                    st.success(f"💡 **Recommended lag features**: {top_lags} (highest autocorrelation)")
                else:
                    st.info("Not enough data to compute lag autocorrelations.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLEAN
# ══════════════════════════════════════════════════════════════════════════════
if _active_nav == "🧹  Clean":
    if st.session_state.cleaning_config is None:
        st.session_state.cleaning_config = get_auto_cleaning_config(df, col_types)
    config = st.session_state.cleaning_config


    st.markdown("#### ⚙️ Global Options")
    config["remove_duplicates"] = st.checkbox(
        f"Remove duplicate rows ({profile['duplicate_rows']:,} rows found)",
        value=config.get("remove_duplicates", True)
    )

    st.markdown("#### 🔧 Column Cleaning Rules")
    cols_with_issues = [
        c for c in df.columns
        if df[c].isnull().sum() > 0
        or (c in col_types.get("numeric", []) and profile["columns"][c].get("outlier_count", 0) > 0)
    ]

    if not cols_with_issues:
        st.success("🎉 All columns are complete and free of outliers!")
    else:
        for col_name in cols_with_issues:
            info = profile["columns"][col_name]
            col_cfg = config["columns"].get(col_name, {})
            null_desc = f"{info['null_pct']:.1f}% missing" if info["null_count"] > 0 else "no missing"
            out_desc = f"{info.get('outlier_pct', 0):.1f}% outliers" if info.get("outlier_count", 0) > 0 else "no outliers"

            with st.expander(f"**{col_name}** — {null_desc} · {out_desc}", expanded=False):
                if info["null_count"] > 0:
                    is_num = col_name in col_types.get("numeric", [])
                    strats = (
                        ["none", "mean", "median", "mode", "ffill", "bfill", "zero", "drop", "custom"]
                        if is_num else
                        ["none", "mode", "ffill", "bfill", "drop", "custom"]
                    )
                    current_strat = col_cfg.get("null_strategy", "none")
                    if current_strat not in strats:
                        current_strat = "none"

                    col_cfg["null_strategy"] = st.selectbox(
                        "Null Imputation strategy", strats, index=strats.index(current_strat),
                        key=f"clean_null_{col_name}"
                    )

                    # Help expander for chosen strategy
                    strategy_key = f"null_{col_cfg['null_strategy']}"
                    render_help_expander(st, strategy_key, f"ℹ️ When to use '{col_cfg['null_strategy']}' imputation")

                    if col_cfg["null_strategy"] == "custom":
                        col_cfg["custom_value"] = st.text_input("Impute specific value", key=f"clean_val_{col_name}")

                if col_name in col_types.get("numeric", []) and info.get("outlier_count", 0) > 0:
                    out_strats = ["keep", "cap", "remove"]
                    current_out = col_cfg.get("outlier_strategy", "keep")
                    col_cfg["outlier_strategy"] = st.selectbox(
                        "Outlier handling strategy", out_strats,
                        index=out_strats.index(current_out), key=f"clean_out_{col_name}"
                    )
                    out_key = f"outlier_{col_cfg['outlier_strategy']}" if col_cfg["outlier_strategy"] != "keep" else "outlier_cap"
                    render_help_expander(st, out_key, f"ℹ️ When to use '{col_cfg['outlier_strategy']}' strategy")

                    # Show mini distribution
                    mini_fig = plot_distribution(df, col_name)
                    st.plotly_chart(mini_fig, use_container_width=True)

                config["columns"][col_name] = col_cfg

    st.session_state.cleaning_config = config
    st.markdown("---")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        if st.button("🧹 Run Cleaning Pipeline", type="primary", use_container_width=True):
            with st.spinner("Executing cleaning transformations..."):
                df_clean, changelog = apply_cleaning(df, config)
                st.session_state.df_clean = df_clean
                st.session_state.cleaning_changelog = changelog
                st.session_state.df_engineered = None
                st.session_state.readiness = None
                st.success(f"Cleaned! Shape: {df_clean.shape[0]:,} × {df_clean.shape[1]}")
    with c_b2:
        if st.button("🔄 Reset to Raw", use_container_width=True):
            st.session_state.df_clean = df.copy()
            st.session_state.cleaning_changelog = []
            st.session_state.df_engineered = None
            st.session_state.readiness = None
            st.rerun()

    if st.session_state.cleaning_changelog:
        st.markdown("#### 📝 Cleaning Operations Log")
        for entry in st.session_state.cleaning_changelog:
            st.markdown(f'<div class="changelog-entry">{entry}</div>', unsafe_allow_html=True)

        if st.session_state.df_clean is not None:
            diff = get_cleaning_diff(df, st.session_state.df_clean)
            st.markdown("#### 📊 Before / After Comparison")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Rows (after)", f"{diff['rows_after']:,}", delta=diff["rows_removed"] * -1)
            d2.metric("Nulls remaining", f"{diff['nulls_after']:,}", delta=diff["nulls_removed"] * -1)
            d3.metric("Duplicates left", f"{diff['dups_after']:,}")
            d4.metric("Columns", st.session_state.df_clean.shape[1])

            st.dataframe(st.session_state.df_clean.head(20), use_container_width=True)
            _download_df(st.session_state.df_clean, "Cleaned Dataset", f"{base}_cleaned")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FEATURE ENGINEER
# ══════════════════════════════════════════════════════════════════════════════
if _active_nav == "⚙️  Feature Engineer":
    base_df = st.session_state.df_clean if st.session_state.df_clean is not None else df

    if st.session_state.engineering_config is None:
        st.session_state.engineering_config = get_auto_engineering_config(
            base_df, col_types, st.session_state.target_col
        )
    eng_cfg = st.session_state.engineering_config
    adv_cfg = st.session_state.adv_config

    # ── Categorical Encoders ───────────────────────────────────────────────────
    cat_cols_eng = [c for c in col_types.get("categorical", []) + col_types.get("boolean", [])
                    if c != st.session_state.target_col]
    if cat_cols_eng:
        st.markdown("#### 🔤 Categorical Encoding")
        render_help_expander(st, "enc_onehot", "ℹ️ Which encoding method should I use?")
        enc_methods = ["label", "onehot", "ordinal", "none"]
        cols_display = st.columns(min(3, len(cat_cols_eng)))
        for i, col_name in enumerate(cat_cols_eng):
            with cols_display[i % min(3, len(cat_cols_eng))]:
                current_enc = eng_cfg.get("encoding", {}).get(col_name, "label")
                if current_enc not in enc_methods:
                    current_enc = "label"
                chosen = st.selectbox(
                    f"**{col_name}** ({base_df[col_name].nunique()} cats)", enc_methods,
                    index=enc_methods.index(current_enc), key=f"enc_{col_name}"
                )
                eng_cfg.setdefault("encoding", {})[col_name] = chosen
                render_help_expander(st, f"enc_{chosen}", f"ℹ️ About {chosen} encoding")

    # ── Numeric Scalers ────────────────────────────────────────────────────────
    num_cols_eng = [c for c in col_types.get("numeric", []) if c != st.session_state.target_col]
    if num_cols_eng:
        st.markdown("#### 📏 Feature Scaling")
        render_help_expander(st, "scale_standard", "ℹ️ Which scaler should I use?")
        scale_methods = ["standard", "minmax", "robust", "none"]
        bulk_scale = st.selectbox(
            "Batch set scaler for all numerical columns",
            ["(manual settings)"] + scale_methods, key="bulk_scale_eng"
        )
        if bulk_scale != "(manual settings)":
            for col_name in num_cols_eng:
                eng_cfg.setdefault("scaling", {})[col_name] = bulk_scale

        sc_cols_display = st.columns(min(3, len(num_cols_eng)))
        for i, col_name in enumerate(num_cols_eng):
            with sc_cols_display[i % min(3, len(num_cols_eng))]:
                current_sc = eng_cfg.get("scaling", {}).get(col_name, "standard")
                if current_sc not in scale_methods:
                    current_sc = "standard"
                chosen_sc = st.selectbox(
                    f"**{col_name}**", scale_methods,
                    index=scale_methods.index(current_sc), key=f"scale_{col_name}"
                )
                eng_cfg.setdefault("scaling", {})[col_name] = chosen_sc

    # ── Datetime Extraction ────────────────────────────────────────────────────
    dt_cols_eng = col_types.get("datetime", [])
    if dt_cols_eng:
        st.markdown("#### 📅 Datetime Feature Decomposition")
        render_help_expander(st, "datetime_decompose", "ℹ️ Which datetime features to extract?")
        dt_options = ["year", "month", "day", "dayofweek", "hour", "minute", "quarter", "weekofyear"]
        for col_name in dt_cols_eng:
            current_dt = eng_cfg.get("datetime", {}).get(col_name, ["year", "month", "day", "dayofweek"])
            selected_feats = st.multiselect(
                f"**{col_name}** extraction features", dt_options,
                default=current_dt, key=f"dt_{col_name}"
            )
            eng_cfg.setdefault("datetime", {})[col_name] = selected_feats

    # ── Advanced Power Transforms ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🛠️ Advanced Transformations")
    all_num_cols = [c for c in base_df.select_dtypes(include=[np.number]).columns.tolist()
                    if c != st.session_state.target_col]
    all_cols = list(base_df.columns)

    at_c1, at_c2 = st.columns(2)
    with at_c1:
        drop_cols = st.multiselect(
            "🗑️ Drop columns permanently", all_cols,
            default=adv_cfg.get("drop_cols", []), key="adv_drop"
        )
        log_cols = st.multiselect(
            "📉 Log1p transform (fixes right-skewed variables)", all_num_cols,
            default=adv_cfg.get("log_transform", []), key="adv_log"
        )
        render_help_expander(st, "log_transform", "ℹ️ When to apply log transform")

        sqrt_cols = st.multiselect(
            "√ Square root transform (moderate skew)", all_num_cols,
            default=adv_cfg.get("sqrt_transform", []), key="adv_sqrt"
        )
        render_help_expander(st, "sqrt_transform", "ℹ️ Log vs sqrt transform")

        sq_cols = st.multiselect(
            "² Square transform (quadratic relationships)", all_num_cols,
            default=adv_cfg.get("square_transform", []), key="adv_sq"
        )
        render_help_expander(st, "square_transform", "ℹ️ When to use squaring")

    with at_c2:
        st.markdown("**🔢 Custom Mathematical Formulas** (pandas.eval syntax)")
        render_help_expander(st, "custom_formula", "ℹ️ How to write custom formulas")
        formula_count = st.number_input("Number of custom formula columns", 0, 8,
                                        len(adv_cfg.get("custom_formulas", [])), key="n_formulas")
        custom_formulas = []
        for fi in range(int(formula_count)):
            fc1, fc2 = st.columns([1, 2])
            with fc1:
                fname = st.text_input(f"Col Name {fi+1}",
                                      value=adv_cfg["custom_formulas"][fi]["name"] if fi < len(adv_cfg["custom_formulas"]) else "",
                                      key=f"f_name_{fi}")
            with fc2:
                fexpr = st.text_input(f"Formula {fi+1}",
                                      value=adv_cfg["custom_formulas"][fi]["formula"] if fi < len(adv_cfg["custom_formulas"]) else "",
                                      placeholder="e.g. SquareFootage / Rooms",
                                      key=f"f_expr_{fi}")
            if fname.strip() and fexpr.strip():
                custom_formulas.append({"name": fname.strip(), "formula": fexpr.strip()})

        st.markdown("**📊 Data Binning (Quantization)**")
        render_help_expander(st, "binning", "ℹ️ When and how to bin data")
        bin_count = st.number_input("Number of columns to bin", 0, 5,
                                    len(adv_cfg.get("binning", [])), key="n_binning")
        binning_configs = []
        for bi in range(int(bin_count)):
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                bcol = st.selectbox(f"Col {bi+1}", ["—"] + all_num_cols, key=f"b_col_{bi}")
            with bc2:
                bbins = st.number_input(f"Bins {bi+1}", 2, 20, 5, key=f"b_bins_{bi}")
            with bc3:
                bstrat = st.selectbox(f"Strategy {bi+1}", ["quantile", "uniform"], key=f"b_strat_{bi}")
            if bcol != "—":
                binning_configs.append({"col": bcol, "bins": bbins, "strategy": bstrat})

        st.markdown("**✖️ Interaction Terms** (col_A × col_B)")
        render_help_expander(st, "interaction_terms", "ℹ️ What are interaction features?")
        n_interactions = st.number_input("Number of interaction terms", 0, 10,
                                         len(adv_cfg.get("interaction_terms", [])), key="n_interact")
        interaction_pairs = []
        for idx in range(int(n_interactions)):
            ic1, ic2 = st.columns(2)
            with ic1:
                ca = st.selectbox(f"Pair {idx+1} Col A", ["—"] + all_num_cols, key=f"int_a_{idx}")
            with ic2:
                cb = st.selectbox(f"Pair {idx+1} Col B", ["—"] + all_num_cols, key=f"int_b_{idx}")
            if ca != "—" and cb != "—" and ca != cb:
                interaction_pairs.append((ca, cb))

    # Save configs
    st.session_state.adv_config = {
        "drop_cols": drop_cols,
        "log_transform": log_cols,
        "sqrt_transform": sqrt_cols,
        "square_transform": sq_cols,
        "interaction_terms": interaction_pairs,
        "custom_formulas": custom_formulas,
        "binning": binning_configs,
    }

    st.markdown("---")
    if st.button("⚙️ Apply All Transformations", type="primary", use_container_width=True):
        with st.spinner("Generating features..."):
            df_eng, eng_log = apply_feature_engineering(base_df, eng_cfg, col_types, st.session_state.target_col)
            df_eng, adv_log = apply_advanced_transforms(df_eng, st.session_state.adv_config)
            st.session_state.df_engineered = df_eng
            st.session_state.engineering_changelog = eng_log + adv_log
            st.session_state.readiness = None
            st.session_state.ml_results = None
            st.success(f"Success! Engineered dataset: {df_eng.shape[0]:,} × {df_eng.shape[1]}")

    if st.session_state.engineering_changelog:
        st.markdown("#### 📝 Transformation Log")
        for entry in st.session_state.engineering_changelog:
            st.markdown(f'<div class="changelog-entry">{entry}</div>', unsafe_allow_html=True)

    if st.session_state.df_engineered is not None:
        st.markdown("#### 👁️ Engineered Dataset Preview")
        e1, e2, e3 = st.columns(3)
        e1.metric("Rows", f"{st.session_state.df_engineered.shape[0]:,}")
        e2.metric("Total features", st.session_state.df_engineered.shape[1])
        e3.metric("Missing cells", int(st.session_state.df_engineered.isnull().sum().sum()))

        st.dataframe(st.session_state.df_engineered.head(10), use_container_width=True)
        _download_df(st.session_state.df_engineered, "Engineered Dataset", f"{base}_engineered")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
if _active_nav == "📋  Report & Export":
    final_df = _get_final_df()
    stage_lbl = ("Engineered" if st.session_state.df_engineered is not None
                 else "Cleaned" if st.session_state.df_clean is not None else "Raw")

    st.markdown("#### 📋 Report & Data Export")
    st.markdown(
        f'<div class="info-box">📌 Evaluating: <strong>{stage_lbl}</strong> dataset '
        f'({final_df.shape[0]:,} rows × {final_df.shape[1]} features) — '
        f'Ready for download and model training.</div>',
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns(2)
    with r1:
        if st.button("🎯 Generate ML Readiness Scorecard", type="primary", use_container_width=True):
            with st.spinner("Analyzing readiness..."):
                st.session_state.readiness = compute_ml_readiness(
                    final_df, col_types, st.session_state.target_col
                )
    with r2:
        render_help_expander(st, "ml_readiness", "ℹ️ What is the ML Readiness Score?")

    if st.session_state.readiness:
        readiness = st.session_state.readiness
        score = readiness["score"]
        grade = readiness["grade"]
        color = readiness["color"]

        st.markdown(
            f"""
            <div class="readiness-card" style="border-color:{color};">
                <div class="score-ring" style="border-color:{color};">
                    <div class="score-number" style="color:{color};">{score}</div>
                    <div class="score-max">/ 100</div>
                </div>
                <div class="score-grade" style="color:{color};">{grade}</div>
                <div class="score-label">ML Readiness Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("#### 🕸️ Dimensional Radar")
            cats = list(readiness["breakdown"].keys())
            vals = list(readiness["breakdown"].values())
            fig_rad = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(108,99,255,0.18)",
                line=dict(color="#6C63FF", width=2.5),
                marker=dict(color="#6C63FF", size=6),
            ))
            fig_rad.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748B")),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                height=400, margin=dict(l=60, r=60, t=40, b=40)
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        with r_col2:
            st.markdown("#### 📊 Score Breakdown")
            for cat, sc_val in readiness["breakdown"].items():
                b_color = "#10B981" if sc_val >= 80 else "#F59E0B" if sc_val >= 55 else "#EF4444"
                st.markdown(
                    f"""<div class="breakdown-row">
                        <span class="breakdown-name">{cat}</span>
                        <div class="breakdown-track">
                            <div class="breakdown-fill" style="width:{sc_val}%;background:{b_color};"></div>
                        </div>
                        <span class="breakdown-val" style="color:{b_color};">{sc_val:.0f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        if readiness["issues"]:
            st.markdown("#### ⚠️ Pipeline Issues Found")
            for iss in readiness["issues"]:
                st.warning(iss)
        else:
            st.success("🎉 No pipeline issues found! Your data is in perfect shape for ML modeling.")



        st.markdown("---")

        # ── HTML Report Export ─────────────────────────────────────────────────
        st.markdown("#### 📄 Export Full HTML Report")
        render_help_expander(st, "html_report", "ℹ️ What's included in the HTML report?")

        if st.button("🖨️ Generate Full HTML Report", use_container_width=True):
            with st.spinner("Building comprehensive report..."):
                try:
                    # Collect charts for report
                    report_charts = {}
                    if len(col_types.get("numeric", [])) >= 2:
                        try:
                            fig_corr, _ = plot_enhanced_heatmap(final_df, method="pearson")
                            report_charts["Correlation Matrix"] = fig_corr
                        except Exception:
                            pass

                    html_report = generate_html_report(
                        profile=profile,
                        col_types=col_types,
                        filename=st.session_state.filename or "dataset",
                        stage=stage_lbl,
                        cleaning_changelog=st.session_state.cleaning_changelog,
                        engineering_changelog=st.session_state.engineering_changelog,
                        ml_results=None,
                        readiness=readiness,
                        target_col=st.session_state.target_col,
                        charts=report_charts,
                    )

                    report_name = (st.session_state.filename or "dataset").rsplit(".", 1)[0]
                    st.download_button(
                        "📥 Download HTML Report",
                        data=html_report.encode("utf-8"),
                        file_name=f"datamind_report_{report_name}.html",
                        mime="text/html",
                        use_container_width=True,
                        key="dl_html_report",
                    )
                    st.success("✅ Report generated! Click above to download.")
                except Exception as e:
                    st.error(f"Report generation error: {e}")

        st.markdown("---")

        # ── Colab / Jupyter Notebook Export ───────────────────────────────────
        st.markdown("#### 🐍 Export as Colab / Jupyter Notebook (.ipynb)")
        st.markdown(
            """
            <div class="info-box">
            📓 Download a <strong>ready-to-run Python notebook</strong> that reproduces every
            cleaning & feature-engineering step you configured — works on
            <strong>Google Colab</strong>, JupyterLab, VS Code, or any IDE that supports <code>.ipynb</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        nb_col1, nb_col2 = st.columns([3, 1])
        with nb_col1:
            nb_include_target = st.checkbox(
                "Include train/test split starter code",
                value=st.session_state.target_col is not None,
                key="nb_include_split",
                help="Adds a ready-to-use sklearn train_test_split cell using your target column.",
            )
        with nb_col2:
            render_help_expander(st, "html_report", "ℹ️ About notebook export")

        if st.button("🐍 Generate Colab Notebook", use_container_width=True, key="gen_nb_btn"):
            with st.spinner("Building notebook..."):
                try:
                    _target_for_nb = st.session_state.target_col if nb_include_target else None
                    nb_json = generate_notebook(
                        filename=st.session_state.filename or "dataset.csv",
                        cleaning_config=st.session_state.cleaning_config or {"remove_duplicates": False, "columns": {}},
                        engineering_config=st.session_state.engineering_config or {"encoding": {}, "scaling": {}, "datetime": {}},
                        adv_config=st.session_state.adv_config or {},
                        col_types=col_types,
                        target_col=_target_for_nb,
                        cleaning_changelog=st.session_state.cleaning_changelog,
                        engineering_changelog=st.session_state.engineering_changelog,
                    )
                    nb_name = (st.session_state.filename or "dataset").rsplit(".", 1)[0]
                    st.download_button(
                        label="📥 Download .ipynb Notebook",
                        data=nb_json.encode("utf-8"),
                        file_name=f"datamind_{nb_name}_analysis.ipynb",
                        mime="application/x-ipynb+json",
                        use_container_width=True,
                        key="dl_notebook",
                    )
                    st.success(
                        "✅ Notebook ready! Click above to download. "
                        "Open it in [Google Colab](https://colab.research.google.com/) or JupyterLab."
                    )

                    # Preview what's inside
                    with st.expander("👁️ Preview notebook contents", expanded=False):
                        steps = []
                        if st.session_state.cleaning_config:
                            if st.session_state.cleaning_config.get("remove_duplicates"):
                                steps.append("✅ Remove duplicate rows")
                            for col_n, ccfg in st.session_state.cleaning_config.get("columns", {}).items():
                                ns = ccfg.get("null_strategy", "none")
                                os_ = ccfg.get("outlier_strategy", "keep")
                                if ns != "none":
                                    steps.append(f"✅ `{col_n}`: fill nulls → **{ns}**")
                                if os_ != "keep":
                                    steps.append(f"✅ `{col_n}`: outliers → **{os_}**")
                        if st.session_state.engineering_config:
                            for col_n, method in st.session_state.engineering_config.get("encoding", {}).items():
                                if method != "none":
                                    steps.append(f"✅ `{col_n}`: encoding → **{method}**")
                            for col_n, method in st.session_state.engineering_config.get("scaling", {}).items():
                                if method != "none":
                                    steps.append(f"✅ `{col_n}`: scaling → **{method}**")
                            for col_n, feats in st.session_state.engineering_config.get("datetime", {}).items():
                                if feats:
                                    steps.append(f"✅ `{col_n}`: datetime → **{', '.join(feats)}**")
                        adv = st.session_state.adv_config or {}
                        for col_n in adv.get("log_transform", []):
                            steps.append(f"✅ `{col_n}`: **log1p** transform")
                        for col_n in adv.get("sqrt_transform", []):
                            steps.append(f"✅ `{col_n}`: **sqrt** transform")
                        for col_n in adv.get("square_transform", []):
                            steps.append(f"✅ `{col_n}`: **squared** transform")
                        for pair in adv.get("interaction_terms", []):
                            if len(pair) == 2:
                                steps.append(f"✅ Interaction term: `{pair[0]}` × `{pair[1]}`")
                        for item in adv.get("custom_formulas", []):
                            steps.append(f"✅ Formula column: `{item.get('name')}` = `{item.get('formula')}`")
                        for item in adv.get("binning", []):
                            steps.append(f"✅ Binning: `{item.get('col')}` → {item.get('bins')} bins ({item.get('strategy')})")
                        for col_n in adv.get("drop_cols", []):
                            steps.append(f"🗑️ Drop column: `{col_n}`")

                        if steps:
                            st.markdown("**Steps included in the notebook:**")
                            for s in steps:
                                st.markdown(f"- {s}")
                        else:
                            st.info("No cleaning or engineering steps configured yet. The notebook will include load & save scaffolding.")

                except Exception as e:
                    st.error(f"Notebook generation error: {e}")

        st.markdown("---")
        st.markdown("#### 📥 Export Analysis-Ready Data")
        st.caption("💡 Download the cleaned/engineered dataset below to use for model training in your own ML pipeline.")
        _download_df(final_df, "Analysis-Ready Data", f"{base}_analysis_ready")
