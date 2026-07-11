"""
ai_agent.py — Google Gemini integration via REST API (no grpcio compilation needed).
Uses the Gemini generateContent REST endpoint directly with the `requests` library.
Enhanced with chat history, streaming feel, and specialized analysis methods.
"""
from __future__ import annotations

import json
from typing import Optional

import requests


_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

_SYSTEM_PERSONA = """You are DataMind AI — a world-class senior data scientist and ML engineer.
You speak in clear, actionable language. You use bullet points, bold key numbers,
and always give specific recommendations tied to the actual data provided.
Format all responses in clean Markdown."""


class AIAgent:
    """Calls Gemini via plain HTTPS REST — zero native dependencies."""

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.available = False
        self.chat_history: list[dict] = []  # Multi-turn conversation memory
        if self.api_key:
            # Quick smoke-test with a tiny prompt
            try:
                resp = self._post("Say: OK")
                self.available = "OK" in resp or len(resp) > 0
            except Exception:
                self.available = False

    # ── Private ────────────────────────────────────────────────────────────

    def _post(self, prompt: str, system_context: str = "", max_tokens: int = 2048) -> str:
        """Send one prompt to Gemini and return the text response."""
        full_prompt = f"{_SYSTEM_PERSONA}\n\n{system_context}\n\n{prompt}" if system_context else f"{_SYSTEM_PERSONA}\n\n{prompt}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": max_tokens,
                "topP": 0.9,
            },
        }
        r = requests.post(
            f"{_GEMINI_URL}?key={self.api_key}",
            json=payload,
            timeout=45,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:200]}")
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _post_with_history(self, user_message: str, system_context: str = "") -> str:
        """Send a message with conversation history for multi-turn chat."""
        # Build contents from history + new message
        contents = []
        for turn in self.chat_history[-10:]:  # Keep last 10 turns for context
            contents.append({"role": turn["role"], "parts": [{"text": turn["content"]}]})

        if system_context:
            full_msg = f"{system_context}\n\n{user_message}"
        else:
            full_msg = user_message

        contents.append({"role": "user", "parts": [{"text": full_msg}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": _SYSTEM_PERSONA}]},
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 2048,
                "topP": 0.9,
            },
        }
        r = requests.post(
            f"{_GEMINI_URL}?key={self.api_key}",
            json=payload,
            timeout=45,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:200]}")
        data = r.json()
        response_text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Update history
        self.chat_history.append({"role": "user", "content": user_message})
        self.chat_history.append({"role": "model", "content": response_text})
        return response_text

    def _call(self, prompt: str, system_context: str = "", max_tokens: int = 2048) -> str:
        if not self.available:
            return (
                "🤖 **AI agent not connected.** Enter a valid Gemini API key in the sidebar "
                "to enable AI-powered insights."
            )
        try:
            return self._post(prompt, system_context, max_tokens)
        except Exception as e:
            return f"⚠️ AI response unavailable: {str(e)}"

    def _schema_summary(self, profile: dict) -> str:
        schema = {
            col: {
                "dtype": info["dtype"],
                "null_pct": info["null_pct"],
                "unique": info["unique_count"],
                "skewness": info.get("skewness"),
                "outlier_pct": info.get("outlier_pct", 0),
            }
            for col, info in profile["columns"].items()
        }
        return json.dumps(schema, indent=2)

    def clear_chat(self):
        """Clear conversation history."""
        self.chat_history = []

    # ── Public insight methods ─────────────────────────────────────────────

    def get_dataset_overview(self, profile: dict) -> str:
        prompt = f"""Write a concise 4-5 sentence plain-English overview of this dataset.
Guess the domain/use case. Comment on quality, scale, and key characteristics.
Highlight any concerning issues (high null %, extreme outliers, etc.).

Dataset info:
- Rows: {profile['n_rows']:,}, Columns: {profile['n_cols']}
- Memory: {profile['memory_mb']:.2f} MB
- Missing: {profile['null_pct']:.1f}%
- Duplicates: {profile['duplicate_pct']:.1f}%
- Quality score: {profile['quality_score']}/100

Column schema:
{self._schema_summary(profile)}

Be specific. Mention column names. Use **bold** for numbers and column names."""
        return self._call(prompt)

    def get_eda_insights(self, profile: dict, eda_stats: dict) -> str:
        numeric_summary = {
            col: {k: v for k, v in info.items()
                  if k in ["mean", "std", "skewness", "kurtosis", "outlier_pct"]}
            for col, info in profile["columns"].items()
            if col in eda_stats
        }
        prompt = f"""Provide exactly 5 bullet-point EDA insights from this data.
Be specific with actual numbers. Focus on distributions, skewness, outliers, and interesting patterns.
Each bullet should start with an emoji and a bold column name.

Dataset: {profile['n_rows']:,} rows × {profile['n_cols']} columns
Numeric statistics:
{json.dumps(numeric_summary, indent=2)}

Format as markdown bullet points with **bold** for column names and numbers."""
        return self._call(prompt)

    def get_cleaning_recommendations(self, profile: dict) -> str:
        null_info = {
            col: f"{info['null_pct']:.1f}% missing ({info['null_count']} rows)"
            for col, info in profile["columns"].items()
            if info["null_count"] > 0
        }
        outlier_info = {
            col: f"{info.get('outlier_pct', 0):.1f}% outliers ({info.get('outlier_count', 0)} rows)"
            for col, info in profile["columns"].items()
            if info.get("outlier_count", 0) > 0
        }
        prompt = f"""Recommend specific data cleaning strategies as exactly 5 bullet points.
Explain WHY you recommend each strategy based on the actual data characteristics shown.

Missing values:
{json.dumps(null_info, indent=2) if null_info else "None"}

Outliers:
{json.dumps(outlier_info, indent=2) if outlier_info else "None"}

Duplicates: {profile['duplicate_pct']:.1f}%

For each column with issues, recommend: the best imputation method AND explain why (e.g. 'use median because skewness=2.4').
Use markdown formatting. Start each bullet with an emoji."""
        return self._call(prompt)

    def get_feature_engineering_hints(
        self, profile: dict, col_types: dict, target_col: Optional[str] = None
    ) -> str:
        prompt = f"""Suggest exactly 5 specific, actionable feature engineering steps.
Be concrete — name the exact columns and transformations.

Column types:
- Numeric: {col_types.get('numeric', [])}
- Categorical: {col_types.get('categorical', [])}
- Datetime: {col_types.get('datetime', [])}
- Boolean: {col_types.get('boolean', [])}

Target column: {target_col or 'not specified'}
Dataset size: {profile['n_rows']:,} rows

For each suggestion: state the column name, the transformation, and the expected benefit.
Use markdown bullet points. Start each with an emoji. **Bold** column names."""
        return self._call(prompt)

    def get_readiness_narrative(
        self, readiness: dict, target_col: Optional[str] = None
    ) -> str:
        prompt = f"""Write a 5-sentence ML readiness assessment and recommendation.
Be specific about which ML algorithms are suitable and what risks remain.

ML Readiness Score: {readiness['score']}/100 — {readiness['grade']}
Target column: {target_col or 'not specified'}

Score breakdown:
{json.dumps(readiness['breakdown'], indent=2)}

Issues detected:
{json.dumps(readiness['issues'], indent=2) if readiness['issues'] else 'None'}

End with a concrete, actionable next-step recommendation."""
        return self._call(prompt)

    def chat_with_data(
        self,
        user_question: str,
        profile: dict,
        col_types: dict,
        target_col: Optional[str] = None,
        cleaning_done: bool = False,
        ml_results: Optional[dict] = None,
    ) -> str:
        """Free-form Q&A about the dataset. Maintains conversation history."""
        context = f"""CURRENT DATASET CONTEXT:
- Shape: {profile['n_rows']:,} rows × {profile['n_cols']} columns
- Quality score: {profile['quality_score']}/100
- Missing: {profile['null_pct']:.1f}% | Duplicates: {profile['duplicate_pct']:.1f}%
- Numeric columns: {col_types.get('numeric', [])}
- Categorical columns: {col_types.get('categorical', [])}
- Datetime columns: {col_types.get('datetime', [])}
- Target column: {target_col or 'not set'}
- Cleaning applied: {'Yes' if cleaning_done else 'No'}
- ML training: {'Completed' if ml_results else 'Not run yet'}

Column details:
{self._schema_summary(profile)}

Answer the user's question based on this dataset context.
Be specific, actionable, and reference actual column names and numbers."""

        if not self.available:
            return "🤖 **AI Chat not connected.** Please enter a valid Gemini API key in the sidebar."
        try:
            return self._post_with_history(user_question, context)
        except Exception as e:
            return f"⚠️ AI response unavailable: {str(e)}"

    def get_time_series_insights(
        self, ts_summary: dict, date_col: str, value_col: str
    ) -> str:
        """Generate insights for time series data."""
        prompt = f"""Analyze this time series data and provide 4 key insights:

Time Series: {value_col} over {date_col}
Summary:
{json.dumps(ts_summary, indent=2, default=str)}

Provide:
1. The trend direction and its business implication
2. Any seasonality or cyclical patterns to expect
3. Whether the series is stationary and what that means for modeling
4. Top 2 recommended ML approaches for this time series

Use markdown. Bold key numbers and column names."""
        return self._call(prompt, max_tokens=1024)

    def get_model_explanation(
        self, ml_results: dict, feature_names: list[str], target_col: str
    ) -> str:
        """Explain ML model results in plain English."""
        best = ml_results.get("best_model_name", "Unknown")
        best_data = ml_results["models"].get(best, {})
        metrics = best_data.get("metrics", {})
        fi = metrics.get("feature_importances", [])[:5]

        prompt = f"""Explain these machine learning results in plain English for a business audience:

Target: {target_col}
Best model: {best}
Task: {"Classification" if ml_results['is_classification'] else "Regression"}

Metrics:
{json.dumps({k: v for k, v in metrics.items() if k != 'feature_importances'}, indent=2)}

Top 5 most important features:
{json.dumps(fi, indent=2)}

Provide:
1. What the model's performance means in practical terms
2. Which features drive predictions most and why that makes sense
3. Specific recommendations to improve performance
4. Any risks or caveats to watch for

Use markdown. Be specific. Bold model names, column names, and metric values."""
        return self._call(prompt, max_tokens=1024)

    def get_correlation_insight(
        self, col_a: str, col_b: str, correlation: float, n_rows: int
    ) -> str:
        """Give a quick insight about a specific correlation pair."""
        prompt = f"""In 2-3 sentences, explain what the correlation between these two columns means:

Column A: {col_a}
Column B: {col_b}
Correlation coefficient: {correlation:.3f}
Dataset size: {n_rows:,} rows

Mention: 1) strength and direction, 2) practical interpretation, 3) whether it's likely causal or coincidental.
Be concise and specific."""
        return self._call(prompt, max_tokens=256)

    def get_auto_pipeline_advice(
        self, profile: dict, col_types: dict, target_col: Optional[str]
    ) -> str:
        """Give a complete recommended pipeline in one response."""
        prompt = f"""As a senior data scientist, recommend a complete end-to-end pipeline for this dataset.

Dataset: {profile['n_rows']:,} rows × {profile['n_cols']} columns
Quality: {profile['quality_score']}/100
Columns: {list(profile['columns'].keys())}
Numeric: {col_types.get('numeric', [])}
Categorical: {col_types.get('categorical', [])}
Target: {target_col or 'not set'}
Missing: {profile['null_pct']:.1f}% | Outliers present: {any(v.get('outlier_count', 0) > 0 for v in profile['columns'].values())}

Provide a numbered 5-step pipeline recommendation:
1. Cleaning steps (specific strategies per column)
2. Feature engineering (specific transforms)
3. Recommended model type and why
4. Key metrics to optimize for
5. Potential pitfalls to watch for

Be actionable and specific. Use **bold** for column names."""
        return self._call(prompt, max_tokens=1500)
