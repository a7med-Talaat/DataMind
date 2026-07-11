"""
data_loader.py — File parsing and column type detection.
"""
import pandas as pd
import numpy as np
from typing import Optional

SUPPORTED_TYPES = ["csv", "xlsx", "xls", "json", "parquet"]


def load_data(uploaded_file) -> pd.DataFrame:
    """Load data from a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1]

    if ext not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported format: .{ext}. Supported: {', '.join(SUPPORTED_TYPES)}"
        )

    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(uploaded_file)
        elif ext == "json":
            df = pd.read_json(uploaded_file)
        elif ext == "parquet":
            try:
                df = pd.read_parquet(uploaded_file)
            except ImportError:
                raise ValueError(
                    "Parquet support requires pyarrow. "
                    "Install it with: pip install pyarrow"
                )
        else:
            raise ValueError(f"Unsupported format: .{ext}")
    except Exception as e:
        raise ValueError(f"Failed to parse file: {str(e)}")

    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify every column as numeric, categorical, datetime, boolean, or text."""
    types: dict = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
    }
    n = len(df)

    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        n_unique = series.nunique()

        if pd.api.types.is_bool_dtype(dtype):
            types["boolean"].append(col)

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            types["datetime"].append(col)

        elif pd.api.types.is_numeric_dtype(dtype):
            types["numeric"].append(col)

        elif dtype == object:
            # Try to parse as datetime
            sample = series.dropna().head(200)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                types["datetime"].append(col)
                continue
            except (ValueError, TypeError):
                pass

            avg_len = series.dropna().str.len().mean() if series.count() > 0 else 0
            if avg_len > 80:
                types["text"].append(col)
            elif n_unique <= max(10, min(50, 0.05 * n)):
                types["categorical"].append(col)
            else:
                types["categorical"].append(col)
        else:
            types["categorical"].append(col)

    return types
