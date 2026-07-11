# 🧠 DataMind — Ultimate AI Data Analyst v1.0

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![UI Style](https://img.shields.io/badge/UI-Custom_Dark_Glassmorphism-purple.svg)](#uidesign)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

DataMind is a local-first, production-grade AI Data Analyst application that turns raw datasets into analysis-ready features and production-grade Jupyter notebooks in minutes. Designed to bridge the gap between initial data ingestion and machine learning readiness, DataMind guides you through a visual, comprehensive data science workflow without writing code.

Whether you need to clean outliers, impute missing values, extract time series features, run high-dimensional projections, or export your workflow directly to Google Colab, DataMind executes all actions locally and generates clean, standard Python scripts to reproduce your entire workflow.

---

### File Hierarchy & Modular System Roles

```
ai data analyst/
├── app.py                          # Streamlit application orchestration & routing
├── requirements.txt                # Core framework and math dependencies
├── sample_housing_data.csv         # Standard test dataset with synthetic anomalies
├── .env.example                    # Environmental configuration blueprint
├── assets/
│   └── style.css                   # Custom dark-theme styling sheet (32KB)
└── core/
    ├── __init__.py
    ├── ai_agent.py                 # Gemini Flash direct REST Integration (no grpcio dependencies)
    ├── cleaner.py                  # Deduplication, null values & outlier cleaning logic
    ├── data_loader.py              # File format reader & statistical column type detection
    ├── dimensionality.py           # PCA computations (2D & 3D projections + variance)
    ├── eda.py                      # Classical EDA: histograms, correlation matrices & null patterns
    ├── feature_engineer.py         # Standard encoders, scalers, and custom pandas eval formulas
    ├── feature_help.py             # Expander documentation maps (28KB)
    ├── ml_engine.py                # Local scikit-learn models & performance evaluations
    ├── ml_readiness.py             # Scorecard logic validating the 5-point data health
    ├── notebook_exporter.py        # Reproducible Jupyter/Colab notebook generator (.ipynb)
    ├── profiler.py                 # Core descriptive statistics & memory estimation
    ├── report_generator.py         # Self-contained Plotly HTML interactive report compiler
    ├── time_series.py              # STL decomposition, stationarity checks & lag analysis
    └── visualizations.py           # Advanced visualization charts (Scatter matrices, violins, 3D)
```

---

## 🔄 User Pipeline Workflow

DataMind enforces a strict pipeline workflow ensuring data is formatted, cleaned, and engineered before being fed into downstream pipelines or HTML report outputs.

```mermaid
graph LR
    A[1. Upload & Profile] --> B[2. Exploratory Analysis]
    B --> C[3. Time Series]
    C --> D[4. Cleaning Pipeline]
    D --> E[5. Feature Engineering]
    E --> F[6. Export & ML Score]
```

---

## 🌟 Detailed Features

### 1. Smart Data Ingestion & Profiling
*   **Format Support:** Seamlessly parses `CSV`, `Excel (.xlsx, .xls)`, `JSON`, and `Parquet` files.
*   **Column Type Detection:** Classifies columns into Numeric, Categorical, Datetime, Boolean, or Text.
*   **Quality Score:** Computes a dataset health grade (0-100) based on duplicates, missing cells, and structural issues.
*   **Visual Health Dashboard:** Renders worst-first sorted column health bars directly inside the workspace UI.
*   **Smart Scan Alerts:** Alerts users to suspicious patterns:
    *   *ID Columns:* Columns containing unique counts near 100% of length.
    *   *Zero Variance:* Constant columns with 1 unique value.
    *   *Multicollinearity:* Features with correlation coefficients $r \ge 0.95$.
    *   *Skewness:* Numeric fields with skew metrics exceeding $1.5$.
    *   *High Nulls:* Columns missing $\ge 50\%$ of data.

### 2. Exploratory Data Analysis & PCA Projections
*   **Heatmaps:** Pearson or Spearman correlation maps featuring an on-click bivariate scatter plot explorer.
*   **Target Correlation:** Horizontal bar plot sorting features by their absolute correlation with your target.
*   **Statistical Distributions:** KDE overlays on top of numeric histograms alongside outlier-marking box plots.
*   **Advanced Visuals:** 
    *   Violin plots featuring optional categorical grouping toggles.
    *   Multi-column scatter matrices colored by any target column.
    *   Interactive 3D scatter plots with coordinate axes selection.
*   **PCA Dimensionality Reduction:** Projects dataset features onto 2D and 3D coordinate space, displaying cumulative explained variance metrics.

### 3. Time Series Suite
*   **Frequency Detection:** Infers index spacing (Daily, Business Days, Monthly, Yearly).
*   **Rolling Aggregations:** Shows rolling mean overlays and rolling standard deviation volatility lines.
*   **ADF Stationarity Testing:** Evaluates Dickey-Fuller statistics, p-values, and offers clear transformation tips if non-stationary.
*   **STL Trend & Seasonality Decomposition:** Extracts Trend, Seasonal patterns, and Residual curves using customizable periodic windows.
*   **Autocorrelation Lag suggestions:** Calculates autocorrelation indexes up to 14 periods, highlighting strong lags.

### 4. Smart Cleaning Operations
*   **Deduplication:** Toggles fast duplicate removal.
*   **Missing Value Imputation:**
    *   *Numeric:* Mean, Median, Zero, custom constant, or forward/backward filling.
    *   *Categorical:* Mode or forward/backward filling.
    *   *Auto-strategy selection:* Recommends Median for skewed columns ($|skew| > 1$) and Mean for normally distributed columns.
*   **Outlier Capping:** Identifies data bounds using standard Interquartile Range ($Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$). Provides options to Keep, Remove rows, or Clip values within limits.

### 5. Feature Engineering Studio
*   **Encoding Strategies:** Label encoding (integers), One-Hot encoding (binary flags), or Ordinal sorting.
*   **Scalers:** Standard (z-score scaling), MinMax (0-1 scale), or Robust (median & IQR normalization).
*   **Datetime Extraction:** Pulls year, month, day, hour, day of week, and quarter details, dropping original time inputs.
*   **Mathematical Transforms:** Creates Log1p (safe offset log curves), Sqrt (square root), Square ($x^2$), and pairwise interaction features.
*   **Pandas custom formulas:** Computes math expressions (e.g., `SalePrice / SquareFootage`) using pandas `.eval()` parsing.
*   **Data Binning:** Converts numerical columns into discrete categories using quantile or uniform ranges.

### 6. Scorecards & Export Actions
*   **ML Readiness Scorecard:** Grades dataset suitability across: Completeness, Variance, Encoding, Balance, and Correlation. Renders a summary radar chart and lists key issue warnings.
*   **Self-Contained HTML Reports:** Compiles a dark-themed document containing interactive Plotly charts, metadata profiles, and data cleaning logs.
*   **Jupyter/Colab Notebook Exporter (.ipynb):** Renders your entire cleaning, imputation, scaling, and feature creation process into runnable Python code. Works on Colab, Jupyter, and VS Code.

---

## 🛠️ Technology Stack

| Library / Module | Purpose |
|---|---|
| **Streamlit** | UI architecture, sidebar panels, and interactive state management |
| **Pandas / NumPy** | Vectorized matrix transformations, clean logs, and feature creation |
| **Plotly Express & Go** | Interactive heatmap graphs, 3D scatter plots, and radar charts |
| **Scikit-Learn** | Model training, Label/One-Hot Encoders, Standard/MinMax Scalers |
| **StatsModels** | Seasonal decomposition (classical/STL) and ADF stationarity tests |
| **Jinja2** | Generates standalone HTML reports from pre-defined CSS layout templates |

---

## 📦 Installation & Quick Start

1. Clone or download the project folder.
2. Initialize a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Configure Gemini AI key inside a `.env` file:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 💾 Export Formats

*   **Cleaned/Engineered Data:** Download as `CSV` or standard Excel `.xlsx` spreadsheets.
*   **Visual Reports:** Download complete `HTML` reports featuring interactive, zoomable plots.
*   **Python Notebooks:** Download standard `.ipynb` notebooks containing structured cells for data importing, cleaning, and model-ready preprocessing.

---

## 📄 License
This project is open-source under the terms of the MIT License.
