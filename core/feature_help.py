"""
feature_help.py — Contextual Help & Feature Info Registry
Provides rich educational descriptions for every control and feature in DataMind.
Each entry explains WHAT the feature does, WHEN to use it, and gives examples.
"""
from __future__ import annotations

FEATURE_HELP: dict[str, dict] = {

    # ── DATA LOADING ──────────────────────────────────────────────────────────

    "file_upload": {
        "title": "📂 Upload Your Dataset",
        "what": "Load your own dataset into DataMind for analysis.",
        "when": "Use this to bring in any structured data you want to analyze, clean, or train models on.",
        "formats": ["CSV (.csv)", "Excel (.xlsx, .xls)", "JSON (.json)", "Parquet (.parquet)"],
        "tip": "💡 **Pro tip**: Parquet files load 10x faster than CSV for large datasets. CSV is fine for up to ~500K rows.",
        "example": "E.g. Upload `customer_churn.csv` to predict churn probability.",
    },

    "target_col": {
        "title": "🎯 Model Target Column",
        "what": "The column you want your ML model to predict.",
        "when": "Set this **before** running ML Training. Also enables Target Correlation analysis in EDA.",
        "tip": "💡 Choose a numeric column for **Regression** (e.g. house price). Choose a categorical/binary column for **Classification** (e.g. churn: yes/no).",
        "example": "E.g. Set `SalePrice` as target to predict housing prices, or `Is_Premium` for a binary classifier.",
    },

    # ── CLEANING ──────────────────────────────────────────────────────────────

    "null_mean": {
        "title": "Null Strategy: Mean Imputation",
        "what": "Fills missing values with the arithmetic average of the column.",
        "when": "✅ Best for numeric columns with a **symmetric (normal) distribution** and less than 20% missing.",
        "avoid": "❌ Avoid if the distribution is heavily skewed — mean gets pulled by outliers.",
        "tip": "💡 Check the column's skewness in the Overview tab first. If |skewness| > 1, use **Median** instead.",
        "example": "E.g. `Temperature` with 5% missing values — mean imputation is safe.",
    },

    "null_median": {
        "title": "Null Strategy: Median Imputation",
        "what": "Fills missing values with the middle value of the sorted column.",
        "when": "✅ Best for **skewed numeric** columns or those with outliers. Robust to extreme values.",
        "tip": "💡 The median is always safer than the mean when you see high skewness or outliers. Default choice for numeric data.",
        "example": "E.g. `Income` or `SquareFootage` columns that have extreme high values — use median.",
    },

    "null_mode": {
        "title": "Null Strategy: Mode Imputation",
        "what": "Fills missing values with the most frequently occurring value.",
        "when": "✅ Best for **categorical** or **low-cardinality** columns.",
        "avoid": "❌ If no single dominant category exists (uniform distribution), mode may introduce bias.",
        "tip": "💡 Works well for columns like `City`, `Color`, or `Category` where one value clearly dominates.",
        "example": "E.g. `Neighborhood` has 60% 'Suburbs' — mode imputation fills gaps with 'Suburbs'.",
    },

    "null_ffill": {
        "title": "Null Strategy: Forward Fill (ffill)",
        "what": "Propagates the last known value forward to fill the gap.",
        "when": "✅ Ideal for **time series** or sequential data where the previous value is a reasonable estimate.",
        "avoid": "❌ Not suitable for random/shuffled data — rows must have natural order.",
        "tip": "💡 Sort your data by a date column first for best results with ffill.",
        "example": "E.g. Stock prices — if Monday is missing, fill it with Friday's closing price.",
    },

    "null_bfill": {
        "title": "Null Strategy: Backward Fill (bfill)",
        "what": "Fills gaps by looking forward to the next known value.",
        "when": "✅ Useful for time series where the *next* value is a better proxy than the *previous*.",
        "tip": "💡 Often combined with ffill for complete time series gap filling.",
        "example": "E.g. Sensor readings — fill a missing reading with the next observed measurement.",
    },

    "null_zero": {
        "title": "Null Strategy: Zero Fill",
        "what": "Replaces all missing values with 0.",
        "when": "✅ Use when missing means 'none occurred' — e.g. no transactions = 0 sales.",
        "avoid": "❌ Never use for columns where 0 is meaningless or impossible (e.g. age, temperature in Kelvin).",
        "tip": "💡 Great for count columns like `num_purchases`, `days_inactive`, `clicks`.",
        "example": "E.g. `num_orders` is null → the customer made 0 orders (genuinely 0).",
    },

    "null_drop": {
        "title": "Null Strategy: Drop Rows",
        "what": "Removes any rows where this column has a missing value.",
        "when": "✅ Safe when missing % is very low (< 2%) and you have plenty of data.",
        "avoid": "❌ Dangerous if many rows are affected — you may introduce sampling bias.",
        "tip": "💡 Check the 'Rows' metric after cleaning to see how many rows were lost.",
        "example": "E.g. Only 3 out of 10,000 rows have null — dropping them loses < 0.03% of data.",
    },

    "null_custom": {
        "title": "Null Strategy: Custom Value",
        "what": "Fills missing values with a specific value you provide.",
        "when": "✅ Use when you know the correct fill value from domain knowledge.",
        "tip": "💡 E.g. fill `Country` nulls with 'Unknown', or fill `discount_pct` nulls with 0.15 (default discount).",
        "example": "E.g. `Payment_Method` is null → fill with 'Cash' as the default method.",
    },

    "outlier_cap": {
        "title": "Outlier Strategy: IQR Cap",
        "what": "Clips extreme values to the IQR fence: Q1 − 1.5×IQR and Q3 + 1.5×IQR.",
        "when": "✅ Best when outliers are likely **data errors** or **measurement artifacts** but you want to keep the rows.",
        "tip": "💡 'Cap' is safer than 'Remove' because it retains all rows. Use Remove only when outliers are truly invalid data points.",
        "example": "E.g. `SquareFootage` has a value of −100 and 999999 — cap brings them to realistic bounds.",
    },

    "outlier_remove": {
        "title": "Outlier Strategy: Remove Rows",
        "what": "Deletes rows where the column value falls outside the IQR fences.",
        "when": "✅ Use when outliers are **genuine errors** (wrong data entry) that would corrupt model training.",
        "avoid": "❌ Avoid if outliers are rare but valid extreme cases that your model needs to learn from.",
        "tip": "💡 First visualize the outliers in EDA > Outlier Detective to understand them before removing.",
        "example": "E.g. A survey records age=999 — this is clearly a data entry error, safe to remove.",
    },

    # ── FEATURE ENGINEERING ───────────────────────────────────────────────────

    "enc_label": {
        "title": "Encoding: Label Encoding",
        "what": "Converts categories to integers: Cat A→0, Cat B→1, Cat C→2.",
        "when": "✅ Use for **ordinal** categories (natural ranking exists) or tree-based models.",
        "avoid": "❌ Avoid for nominal categories with linear models — the numbers imply false ordering.",
        "tip": "💡 Label encoding is safe for Random Forest and Gradient Boosting. Use One-Hot for Logistic Regression.",
        "example": "E.g. `Education` = [High School, Bachelor, Master, PhD] — label encoding preserves the order.",
    },

    "enc_onehot": {
        "title": "Encoding: One-Hot Encoding",
        "what": "Creates a new binary column for each category (1 if present, 0 otherwise).",
        "when": "✅ Best for **nominal** (unordered) categories with linear or distance-based models.",
        "avoid": "❌ Avoid for high-cardinality columns (100+ categories) — creates too many columns.",
        "tip": "💡 One-Hot doubles or triples your column count. Check `Total features` in the Engineered Preview.",
        "example": "E.g. `Color` = [Red, Blue, Green] → creates `Color_Red`, `Color_Blue`, `Color_Green` columns.",
    },

    "enc_ordinal": {
        "title": "Encoding: Ordinal Encoding",
        "what": "Assigns integer ranks based on value frequency (most common = 0).",
        "when": "✅ A middle ground — when you want integer encoding but don't have a fixed ranking.",
        "tip": "💡 Similar to label encoding but frequency-based. Useful for low-cardinality nominals.",
        "example": "E.g. `City` with 5 cities → each city gets a rank based on how often it appears.",
    },

    "scale_standard": {
        "title": "Scaling: Standard Scaler (Z-score)",
        "what": "Subtracts the mean and divides by standard deviation. Result: mean=0, std=1.",
        "when": "✅ Default choice for **linear models** (Logistic/Linear Regression) and neural networks.",
        "avoid": "❌ Not ideal when you need the original range preserved (e.g. for interpretability).",
        "tip": "💡 Standard scaling assumes roughly normal distributions. Use Robust for skewed data.",
        "example": "E.g. `Age` ranges 18–80 and `Income` ranges 20K–500K → scaling brings both to comparable scale.",
    },

    "scale_minmax": {
        "title": "Scaling: MinMax Scaler",
        "what": "Scales values to [0, 1] range: (x − min) / (max − min).",
        "when": "✅ Use when you need values in a fixed [0,1] range, e.g. neural networks or image data.",
        "avoid": "❌ Very sensitive to outliers — one extreme value compresses everything else near 0.",
        "tip": "💡 Always clean outliers BEFORE applying MinMax scaling.",
        "example": "E.g. Pixel values (0–255) scaled to (0–1) for a neural network input.",
    },

    "scale_robust": {
        "title": "Scaling: Robust Scaler",
        "what": "Scales using the median and IQR instead of mean/std. Ignores outliers.",
        "when": "✅ Best for **skewed** distributions or data with significant outliers.",
        "tip": "💡 If your column has high skewness (|skew| > 1) or many outliers, Robust Scaler outperforms Standard.",
        "example": "E.g. `Income` is right-skewed with millionaires as outliers — Robust Scaler handles this correctly.",
    },

    "log_transform": {
        "title": "Log1p Transform",
        "what": "Applies log(1 + x) to the column, compressing large values.",
        "when": "✅ Use on **right-skewed** numeric columns (skewness > 1.0) to normalize the distribution.",
        "avoid": "❌ Column must be non-negative (no negative values). Use after outlier capping if needed.",
        "tip": "💡 Log transform is one of the most powerful steps for house prices, salary, revenue data.",
        "example": "E.g. `SalePrice` skewness = 2.4 → after log1p, skewness drops to ~0.1.",
    },

    "sqrt_transform": {
        "title": "Square Root Transform",
        "what": "Applies √x to the column — a gentler compression than log.",
        "when": "✅ Use for **moderately skewed** columns (skewness 0.5–1.5) or count data.",
        "avoid": "❌ Column must be non-negative.",
        "tip": "💡 Sqrt is softer than log. Try sqrt first for moderate skew, log for severe skew.",
        "example": "E.g. `num_bedrooms` with skewness = 0.8 → sqrt brings it closer to normal.",
    },

    "square_transform": {
        "title": "Square Transform (x²)",
        "what": "Applies x² to amplify large values and capture non-linear relationships.",
        "when": "✅ Use when you suspect a **quadratic/polynomial** relationship with the target.",
        "avoid": "❌ Avoid on large-magnitude columns — squaring can create astronomically large values.",
        "tip": "💡 Always pair with Standard Scaling after squaring to keep values manageable.",
        "example": "E.g. Physics data: `velocity²` relates to kinetic energy — square transform captures this.",
    },

    "interaction_terms": {
        "title": "Interaction Terms (A × B)",
        "what": "Creates a new column = colA × colB, capturing combined effects.",
        "when": "✅ Use when two features together are more predictive than each alone.",
        "tip": "💡 Domain knowledge helps here. E.g. `rooms × sqft_per_room` may predict price better than either alone.",
        "example": "E.g. `SquareFootage × Rooms` → a new feature representing space density.",
    },

    "binning": {
        "title": "Data Binning (Quantization)",
        "what": "Converts a continuous column into discrete bins/buckets.",
        "when": "✅ Use when the relationship with the target is step-like rather than linear.",
        "strategies": {
            "quantile": "Equal-frequency bins — each bin has the same number of rows. Good for skewed data.",
            "uniform": "Equal-width bins — each bin covers the same value range. Good for uniform distributions.",
        },
        "tip": "💡 Start with 5 bins. Fewer bins = less noise but more information loss.",
        "example": "E.g. `Age` → bins: [0–25, 25–40, 40–55, 55–70, 70+] for age group analysis.",
    },

    "custom_formula": {
        "title": "Custom Mathematical Formula",
        "what": "Create a brand new column by writing a pandas-eval expression.",
        "when": "✅ Use when you have domain knowledge about derived features that the auto-engineer won't discover.",
        "syntax": "Uses Python/pandas syntax. Reference columns by name.",
        "tip": "💡 Examples: `price / sqft`, `log(income + 1)`, `col_a - col_b`, `col_a ** 2`",
        "example": "E.g. `SquareFootage / Rooms` creates `SqFt_Per_Room` — a useful density feature.",
    },

    "datetime_decompose": {
        "title": "Datetime Feature Decomposition",
        "what": "Extracts numeric sub-features from a datetime column.",
        "features": {
            "year": "The calendar year — captures long-term trends.",
            "month": "Month 1–12 — captures seasonality.",
            "day": "Day of month 1–31 — captures monthly patterns.",
            "dayofweek": "0=Monday to 6=Sunday — captures weekly cycles.",
            "hour": "Hour 0–23 — captures intra-day patterns.",
            "quarter": "Quarter 1–4 — captures business cycles.",
            "weekofyear": "Week 1–52 — fine-grained seasonal signal.",
        },
        "tip": "💡 Always extract at minimum `month` and `dayofweek` — they contain strong seasonal signals for most real-world datasets.",
        "example": "E.g. `Date_Listed` → extract `year`, `month`, `dayofweek` to capture listing timing effects.",
    },

    # ── ML TRAINING ───────────────────────────────────────────────────────────

    "random_forest": {
        "title": "Random Forest",
        "what": "An ensemble of decision trees trained on random feature subsets, predictions averaged.",
        "when": "✅ Excellent all-around model. Fast, handles mixed data types, naturally resistant to overfitting.",
        "pros": ["Robust to outliers", "No scaling needed", "Built-in feature importance", "Fast training"],
        "cons": ["Large memory for huge datasets", "Slower than linear models for very high-dimensional data"],
        "tip": "💡 Random Forest is the best first-try model for most tabular datasets.",
    },

    "extra_trees": {
        "title": "Extra Trees (Extremely Randomized Trees)",
        "what": "Like Random Forest but uses random split thresholds — even faster, often comparable accuracy.",
        "when": "✅ Use when speed matters or Random Forest overfits.",
        "pros": ["Fastest tree ensemble", "Lower variance than RF", "Great for high-noise data"],
        "tip": "💡 Extra Trees often beats Random Forest on noisy datasets because the extra randomness acts as regularization.",
    },

    "gradient_boosting": {
        "title": "Gradient Boosting",
        "what": "Sequentially trains trees where each tree corrects the previous tree's errors.",
        "when": "✅ Often the highest-accuracy model on structured/tabular data. Industry favorite.",
        "pros": ["Best accuracy on structured data", "Handles missing values", "Excellent feature importance"],
        "cons": ["Slower to train than RF/ExtraTrees", "More hyperparameters to tune"],
        "tip": "💡 If you need the best accuracy and can wait 2–3x longer for training, use Gradient Boosting.",
    },

    "linear_logistic": {
        "title": "Linear / Logistic Regression",
        "what": "Fits a linear equation to the features. Fastest model, highly interpretable.",
        "when": "✅ Use as a **baseline** and when interpretability is critical. Also reveals linear relationships.",
        "pros": ["Fastest training", "Perfectly interpretable coefficients", "Great baseline"],
        "cons": ["Cannot capture non-linear patterns", "Requires scaling and encoding"],
        "tip": "💡 If Linear Regression outperforms tree models, your target has a strong linear relationship with features.",
    },

    "train_test_split": {
        "title": "Train-Test Split",
        "what": "Divides your dataset into a training set (model learns) and test set (model evaluated).",
        "when": "Standard practice for all ML workflows.",
        "values": {
            "10–20%": "Use for small datasets (<1000 rows) to maximize training data.",
            "20–30%": "Standard for medium datasets. Default: 20%.",
            "30–40%": "Use when you have abundant data and need a robust evaluation.",
        },
        "tip": "💡 Never evaluate on training data — it gives falsely optimistic scores.",
    },

    "cross_validation": {
        "title": "Cross-Validation (5-Fold CV)",
        "what": "Splits data into 5 folds. Trains 5 models each time using 4 folds, evaluates on the 5th. Averages results.",
        "when": "✅ Gives a much more reliable estimate of model performance than a single train-test split.",
        "tip": "💡 If your CV score is much lower than train-test score, your model is overfitting.",
        "example": "E.g. Train-test F1 = 0.92 but 5-fold CV F1 = 0.74 → overfitting — simplify the model.",
    },

    "confusion_matrix": {
        "title": "Confusion Matrix",
        "what": "Shows counts of True Positives, True Negatives, False Positives, False Negatives.",
        "when": "✅ Essential for understanding WHERE your classifier makes mistakes.",
        "reading": {
            "True Positive (TP)": "Correctly predicted positive — ideal",
            "True Negative (TN)": "Correctly predicted negative — ideal",
            "False Positive (FP)": "Predicted positive but was negative — Type I Error",
            "False Negative (FN)": "Predicted negative but was positive — Type II Error",
        },
        "tip": "💡 For fraud/medical detection, False Negatives are worse than False Positives — optimize Recall.",
    },

    "roc_curve": {
        "title": "ROC Curve & AUC Score",
        "what": "Plots True Positive Rate vs False Positive Rate at all classification thresholds.",
        "reading": {
            "AUC = 1.0": "Perfect classifier",
            "AUC = 0.9+": "Excellent",
            "AUC = 0.8–0.9": "Good",
            "AUC = 0.7–0.8": "Fair",
            "AUC = 0.5": "Random guessing — your model learned nothing",
        },
        "tip": "💡 AUC is threshold-independent — it measures the model's inherent discriminative ability.",
    },

    "permutation_importance": {
        "title": "Permutation Feature Importance",
        "what": "Measures how much model accuracy drops when each feature's values are randomly shuffled.",
        "when": "✅ More reliable than tree-based importance. Works with ANY model type.",
        "reading": "High importance = shuffling that feature severely hurts accuracy → feature is critical.",
        "tip": "💡 If a feature has near-zero permutation importance, it can be safely dropped to simplify the model.",
    },

    # ── PCA ───────────────────────────────────────────────────────────────────

    "pca_2d": {
        "title": "PCA 2D Projection",
        "what": "Compresses all numeric columns into 2 principal components for a flat scatter plot.",
        "when": "✅ Use to visualize clusters, outliers, and class separation in your data.",
        "reading": "Points that cluster together are similar rows. Separated clusters suggest natural groupings.",
        "tip": "💡 Color the points by your target column to see if the classes are linearly separable.",
    },

    "pca_3d": {
        "title": "PCA 3D Projection",
        "what": "Projects data onto 3 principal components — a rotatable 3D scatter plot.",
        "when": "✅ Use when 2D doesn't capture enough variance (check the Explained Variance chart).",
        "tip": "💡 Drag to rotate the 3D plot. Tight clusters in 3D that overlap in 2D reveal hidden structure.",
    },

    "pca_variance": {
        "title": "Explained Variance Chart",
        "what": "Shows how much variance each Principal Component captures.",
        "reading": "If PC1+PC2 explains 90%+ of variance → 2D projection is a faithful representation of your data.",
        "tip": "💡 Aim for 80–90% cumulative variance. If you need PC1–PC5 to reach 80%, your data is truly high-dimensional.",
    },

    # ── TIME SERIES ───────────────────────────────────────────────────────────

    "rolling_mean": {
        "title": "Rolling Mean (Moving Average)",
        "what": "Smooths the time series by averaging over a sliding window of N periods.",
        "when": "✅ Use to remove short-term noise and reveal underlying trends.",
        "tip": "💡 Larger window = smoother trend but more lag. Start with window = 7 for daily data, 12 for monthly.",
        "example": "E.g. 7-day rolling mean of daily sales → reveals weekly trend without daily noise.",
    },

    "rolling_std": {
        "title": "Rolling Standard Deviation",
        "what": "Measures how much the series fluctuates over a sliding window.",
        "when": "✅ Reveals volatility regimes — periods of high vs low volatility.",
        "tip": "💡 A rising rolling std means your series is becoming more unstable over time.",
    },

    "adf_test": {
        "title": "ADF Stationarity Test",
        "what": "Augmented Dickey-Fuller test checks if the time series is stationary (no trend or seasonality).",
        "reading": {
            "p < 0.05": "Stationary — series mean and variance are constant over time",
            "p ≥ 0.05": "Non-stationary — has trend or seasonality that needs to be removed",
        },
        "tip": "💡 Most ML models require stationary features. Apply differencing or detrending if p ≥ 0.05.",
    },

    # ── CORRELATION ───────────────────────────────────────────────────────────

    "pearson_corr": {
        "title": "Pearson Correlation",
        "what": "Measures the linear relationship between two columns. Range: −1 to +1.",
        "reading": {
            "+1.0": "Perfect positive linear relationship",
            "+0.7 to +1.0": "Strong positive",
            "+0.3 to +0.7": "Moderate positive",
            "0": "No linear relationship",
            "Negative": "Inverse relationship",
        },
        "tip": "💡 Pearson assumes linear relationships and normal distributions. Use Spearman for ranked/skewed data.",
    },

    "spearman_corr": {
        "title": "Spearman Correlation",
        "what": "Measures monotonic (not necessarily linear) relationships using ranked values.",
        "when": "✅ Use for skewed distributions, ordinal data, or when outliers are present.",
        "tip": "💡 Spearman is more robust than Pearson. Use it when your data doesn't follow a normal distribution.",
    },

    # ── REPORT ────────────────────────────────────────────────────────────────

    "ml_readiness": {
        "title": "ML Readiness Scorecard",
        "what": "Evaluates your dataset across 6 dimensions to determine if it's ready for ML.",
        "dimensions": {
            "Completeness": "How few missing values remain",
            "Consistency": "No duplicate rows, consistent data types",
            "Feature Quality": "Variance, uniqueness, and informativeness of features",
            "Target Quality": "Target column balance and completeness",
            "Scale Uniformity": "Are features on comparable scales?",
            "Cardinality": "Are categorical columns reasonably encoded?",
        },
        "tip": "💡 Score 80+ means your dataset is ready for ML. Below 60 suggests critical issues to fix first.",
    },

    "html_report": {
        "title": "Export Full HTML Report",
        "what": "Generates a self-contained HTML file with all charts, statistics, cleaning logs, and ML results.",
        "when": "✅ Use to share findings with stakeholders or archive your analysis pipeline.",
        "tip": "💡 The HTML report is fully self-contained — no internet connection needed to view it.",
    },
}


def get_help(key: str) -> dict | None:
    """Return help entry for a given feature key, or None if not found."""
    return FEATURE_HELP.get(key)


def render_help_expander(st_module, key: str, label: str = "ℹ️ What is this? When should I use it?"):
    """
    Renders a styled Streamlit expander with the help content for the given key.
    Call this right below any control widget.
    """
    help_data = get_help(key)
    if not help_data:
        return

    with st_module.expander(label, expanded=False):
        st_module.markdown(f"### {help_data['title']}")

        if "what" in help_data:
            st_module.markdown(f"**📖 What it does:** {help_data['what']}")

        if "when" in help_data:
            st_module.markdown(f"**✅ When to use:** {help_data['when']}")

        if "avoid" in help_data:
            st_module.markdown(f"**❌ When to avoid:** {help_data['avoid']}")

        if "formats" in help_data:
            st_module.markdown("**Supported formats:** " + " · ".join(help_data["formats"]))

        if "features" in help_data:
            st_module.markdown("**Extractable features:**")
            for feat, desc in help_data["features"].items():
                st_module.markdown(f"  - **`{feat}`**: {desc}")

        if "strategies" in help_data:
            for strat, desc in help_data["strategies"].items():
                st_module.markdown(f"  - **{strat}**: {desc}")

        if "dimensions" in help_data:
            for dim, desc in help_data["dimensions"].items():
                st_module.markdown(f"  - **{dim}**: {desc}")

        if "reading" in help_data:
            reading = help_data["reading"]
            if isinstance(reading, dict):
                st_module.markdown("**📊 How to read:**")
                for val, meaning in reading.items():
                    st_module.markdown(f"  - **{val}** → {meaning}")
            else:
                st_module.markdown(f"**📊 How to read:** {reading}")

        if "pros" in help_data:
            pros_str = " · ".join(help_data["pros"])
            st_module.markdown(f"**✅ Pros:** {pros_str}")

        if "cons" in help_data:
            cons_str = " · ".join(help_data["cons"])
            st_module.markdown(f"**⚠️ Cons:** {cons_str}")

        if "tip" in help_data:
            st_module.info(help_data["tip"])

        if "example" in help_data:
            st_module.markdown(f"**🧪 Example:** *{help_data['example']}*")
