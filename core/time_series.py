"""
time_series.py — Time Series Analysis Module for DataMind
Auto-detects datetime columns, computes rolling stats, stationarity tests,
lag feature suggestions, and trend/seasonality decomposition.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def detect_time_series_columns(df: pd.DataFrame, col_types: dict) -> list[str]:
    """Returns datetime columns that could serve as time series index."""
    return col_types.get("datetime", [])


def prepare_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Sort and index a DataFrame by date for time series analysis."""
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col, value_col])
    ts = ts.sort_values(date_col).reset_index(drop=True)
    return ts


def compute_rolling_stats(series: pd.Series, windows: list[int] = [7, 14, 30]) -> dict:
    """Compute rolling mean and std for multiple window sizes."""
    results = {}
    for w in windows:
        if len(series) > w:
            results[w] = {
                "mean": series.rolling(window=w, center=True).mean(),
                "std": series.rolling(window=w, center=True).std(),
                "min": series.rolling(window=w).min(),
                "max": series.rolling(window=w).max(),
            }
    return results


def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller stationarity test.
    Returns: {statistic, p_value, is_stationary, lags_used, critical_values}
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        series_clean = series.dropna()
        result = adfuller(series_clean, autolag="AIC")
        return {
            "statistic": float(result[0]),
            "p_value": float(result[1]),
            "lags_used": int(result[2]),
            "n_obs": int(result[3]),
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "is_stationary": result[1] < 0.05,
        }
    except ImportError:
        # statsmodels not available — use simple variance ratio test approximation
        series_clean = series.dropna()
        n = len(series_clean)
        first_half_var = series_clean[:n//2].var()
        second_half_var = series_clean[n//2:].var()
        ratio = second_half_var / (first_half_var + 1e-9)
        is_stationary = 0.5 < ratio < 2.0
        return {
            "statistic": float(ratio),
            "p_value": None,
            "lags_used": None,
            "n_obs": n,
            "critical_values": {},
            "is_stationary": is_stationary,
            "note": "statsmodels not installed — using variance ratio approximation",
        }
    except Exception as e:
        return {"error": str(e), "is_stationary": None}


def decompose_trend_seasonality(series: pd.Series, period: Optional[int] = None) -> dict:
    """
    Decompose time series into trend + seasonality + residual.
    Falls back to simple moving-average-based decomposition if statsmodels unavailable.
    """
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return {"error": "Need at least 10 non-null data points for decomposition."}

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        if period is None:
            period = max(2, min(12, len(series_clean) // 4))

        if len(series_clean) < 2 * period:
            return {"error": f"Need at least {2*period} data points for period={period} decomposition."}

        result = seasonal_decompose(series_clean, model="additive", period=period, extrapolate_trend="freq")
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "period": period,
            "method": "statsmodels seasonal_decompose",
        }
    except ImportError:
        # Fallback: simple moving average trend
        w = period or max(2, min(12, len(series_clean) // 4))
        trend = series_clean.rolling(window=w, center=True, min_periods=1).mean()
        detrended = series_clean - trend
        residual = detrended - detrended.rolling(window=w, center=True, min_periods=1).mean()
        seasonal = detrended - residual
        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "period": w,
            "method": "moving average (fallback)",
        }
    except Exception as e:
        return {"error": str(e)}


def suggest_lag_features(df: pd.DataFrame, date_col: str, value_col: str, max_lag: int = 7) -> dict:
    """
    Compute autocorrelation at each lag to suggest useful lag features.
    Returns: {lag: autocorr} sorted by absolute autocorr
    """
    ts = prepare_time_series(df, date_col, value_col)
    series = ts[value_col]
    results = {}
    for lag in range(1, min(max_lag + 1, len(series) // 2)):
        try:
            corr = series.autocorr(lag=lag)
            if not np.isnan(corr):
                results[lag] = round(float(corr), 4)
        except Exception:
            pass
    return dict(sorted(results.items(), key=lambda x: abs(x[1]), reverse=True))


def compute_ts_summary(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    """
    Full time series summary: basic stats + stationarity + trend direction.
    """
    ts = prepare_time_series(df, date_col, value_col)
    series = ts[value_col]

    if len(series) < 5:
        return {"error": "Not enough data points for time series analysis (need 5+)."}

    # Trend direction via linear regression slope
    x = np.arange(len(series))
    try:
        slope = float(np.polyfit(x, series.values, 1)[0])
        trend_direction = "📈 Upward" if slope > 0.001 else "📉 Downward" if slope < -0.001 else "➡️ Flat"
    except Exception:
        slope = 0.0
        trend_direction = "Unknown"

    # Stationarity
    adf = adf_test(series)

    # Basic stats
    summary = {
        "n_points": len(series),
        "date_range": {
            "start": str(ts[date_col].min()),
            "end": str(ts[date_col].max()),
        },
        "value_stats": {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "cv_pct": round(float(series.std() / (series.mean() + 1e-9) * 100), 2),
        },
        "trend": {
            "direction": trend_direction,
            "slope_per_period": round(slope, 6),
        },
        "stationarity": adf,
        "missing_points": int(ts[value_col].isnull().sum()),
    }
    return summary


def infer_frequency(df: pd.DataFrame, date_col: str) -> str:
    """Infer the dominant frequency of the time series (daily, monthly, etc.)."""
    ts = df[date_col].dropna()
    ts = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
    if len(ts) < 3:
        return "Unknown"

    diffs = ts.diff().dropna()
    median_diff = diffs.median()

    if pd.isnull(median_diff):
        return "Unknown"

    days = median_diff.days
    if days <= 1:
        return "Hourly / Daily"
    elif days <= 7:
        return "Daily"
    elif days <= 14:
        return "Weekly"
    elif days <= 35:
        return "Monthly"
    elif days <= 100:
        return "Quarterly"
    else:
        return "Yearly"
