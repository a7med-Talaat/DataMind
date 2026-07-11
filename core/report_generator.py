"""
report_generator.py — Full Pipeline HTML Report Generator for DataMind
Generates a self-contained, styled HTML report with all charts and statistics.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, Any

import pandas as pd
import plotly.graph_objects as go


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DataMind Report — {filename}</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #050A14;
    color: #E2E8F0;
    margin: 0;
    padding: 0;
    min-height: 100vh;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}

  /* Header */
  .report-header {{
    text-align: center;
    padding: 60px 24px 48px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 48px;
  }}
  .report-badge {{
    display: inline-block;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.3);
    color: #A5B4FC;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
  }}
  .report-title {{
    font-size: clamp(2rem, 4vw, 3.5rem);
    font-weight: 900;
    background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 60%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.04em;
    margin-bottom: 12px;
  }}
  .report-meta {{
    font-size: 0.85rem;
    color: #64748B;
  }}

  /* Section */
  .section {{
    margin-bottom: 48px;
  }}
  .section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: #E2E8F0;
    letter-spacing: -0.01em;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  /* Metrics grid */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .metric-card {{
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 18px;
  }}
  .metric-label {{
    font-size: 0.68rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 6px;
  }}
  .metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #E2E8F0;
  }}
  .metric-delta {{
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 4px;
  }}

  /* Table */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  thead {{ background: rgba(108,99,255,0.08); }}
  th {{
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
    color: #94A3B8;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #CBD5E1;
  }}
  tr:hover {{ background: rgba(255,255,255,0.015); }}
  .table-wrap {{
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    overflow: hidden;
    overflow-x: auto;
  }}

  /* Chart container */
  .chart-container {{
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 24px;
  }}

  /* Changelog */
  .log-entry {{
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 0.82rem;
    color: #94A3B8;
  }}

  /* Score ring */
  .score-section {{
    display: flex;
    align-items: center;
    gap: 32px;
    flex-wrap: wrap;
  }}
  .score-circle {{
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 5px solid;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }}
  .score-num {{ font-size: 2.5rem; font-weight: 900; line-height: 1; }}
  .score-max {{ font-size: 0.8rem; color: #475569; }}
  .score-grade {{ font-size: 1rem; font-weight: 700; margin-top: 8px; }}
  .score-label {{ font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; }}

  /* Alert boxes */
  .alert {{ border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; font-size: 0.84rem; border-left: 4px solid; }}
  .alert-warning {{ background: rgba(245,158,11,0.08); border-color: #F59E0B; color: #F59E0B; }}
  .alert-success {{ background: rgba(16,185,129,0.08); border-color: #10B981; color: #10B981; }}
  .alert-info {{ background: rgba(108,99,255,0.08); border-color: #6C63FF; color: #A5B4FC; }}

  /* Footer */
  .footer {{
    text-align: center;
    color: #334155;
    font-size: 0.72rem;
    padding: 40px 0 20px;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 60px;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="report-header">
    <div class="report-badge">🧠 DataMind · AI Data Analyst Report</div>
    <div class="report-title">DataMind Analysis Report</div>
    <div class="report-meta">
      File: <strong>{filename}</strong> &nbsp;·&nbsp;
      Generated: <strong>{generated_at}</strong> &nbsp;·&nbsp;
      Pipeline stage: <strong>{stage}</strong>
    </div>
  </div>

  <!-- Dataset Overview -->
  <div class="section">
    <div class="section-title">📊 Dataset Overview</div>
    <div class="metrics-grid">
      {metrics_html}
    </div>
  </div>

  <!-- Column Statistics -->
  <div class="section">
    <div class="section-title">🔬 Column Statistics</div>
    <div class="table-wrap">
      {col_stats_table}
    </div>
  </div>

  <!-- Charts -->
  {charts_html}

  <!-- Cleaning Log -->
  {cleaning_section}

  <!-- Engineering Log -->
  {engineering_section}

  <!-- ML Results -->
  {ml_section}

  <!-- ML Readiness -->
  {readiness_section}

  <!-- Footer -->
  <div class="footer">
    Generated by DataMind v2.0 — Ultimate AI Data Analyst &nbsp;·&nbsp; {generated_at}
  </div>

</div>
<script>
{plotly_scripts}
</script>
</body>
</html>"""


def _metric_card(label: str, value: str, delta: str = "") -> str:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {delta_html}
    </div>"""


def _col_stats_table(profile: dict, col_types: dict) -> str:
    rows_html = ""
    for col_name, info in profile["columns"].items():
        ct = next((t for t, cols in col_types.items() if col_name in cols), "—")
        null_color = "#EF4444" if info["null_pct"] > 10 else "#F59E0B" if info["null_pct"] > 0 else "#10B981"
        mean_str = f"{info['mean']:.4g}" if info.get("mean") is not None else "—"
        std_str = f"{info['std']:.4g}" if info.get("std") is not None else "—"
        skew_str = f"{info['skewness']:.3f}" if info.get("skewness") is not None else "—"
        out_str = str(info.get("outlier_count", 0))
        rows_html += f"""<tr>
          <td><strong>{col_name}</strong></td>
          <td>{ct}</td>
          <td>{info['dtype']}</td>
          <td style="color:{null_color}">{info['null_pct']:.1f}%</td>
          <td>{info['unique_count']:,}</td>
          <td>{mean_str}</td>
          <td>{std_str}</td>
          <td>{skew_str}</td>
          <td>{out_str}</td>
        </tr>"""

    return f"""<table>
    <thead><tr>
      <th>Column</th><th>Type</th><th>Dtype</th>
      <th>Null %</th><th>Unique</th>
      <th>Mean</th><th>Std</th><th>Skew</th><th>Outliers</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>"""


def _log_section(title: str, icon: str, entries: list[str]) -> str:
    if not entries:
        return ""
    log_html = "".join(f'<div class="log-entry">{e}</div>' for e in entries)
    return f"""
    <div class="section">
      <div class="section-title">{icon} {title}</div>
      {log_html}
    </div>"""


def _ml_results_section(ml_results: dict, target_col: str) -> str:
    if not ml_results or "error" in ml_results:
        return ""

    task = "Classification" if ml_results["is_classification"] else "Regression"
    best = ml_results.get("best_model_name", "—")

    rows_html = ""
    for name, data in ml_results["models"].items():
        if "error" in data:
            rows_html += f"<tr><td>{name}</td><td colspan='4' style='color:#EF4444'>Error: {data['error']}</td></tr>"
        else:
            metrics = data["metrics"]
            is_best = name == best
            style = "color:#10B981;font-weight:700" if is_best else ""
            badge = " 🌟" if is_best else ""
            if ml_results["is_classification"]:
                acc = f"{metrics.get('Accuracy', 0):.4f}"
                f1 = f"{metrics.get('F1-Score', 0):.4f}"
                prec = f"{metrics.get('Precision', 0):.4f}"
                rec = f"{metrics.get('Recall', 0):.4f}"
                rows_html += f"<tr style='{style}'><td>{name}{badge}</td><td>{acc}</td><td>{f1}</td><td>{prec}</td><td>{rec}</td></tr>"
            else:
                r2 = f"{metrics.get('R2-Score', 0):.4f}"
                mae = f"{metrics.get('MAE', 0):.4f}"
                rmse = f"{metrics.get('RMSE', 0):.4f}"
                rows_html += f"<tr style='{style}'><td>{name}{badge}</td><td>{r2}</td><td>{mae}</td><td>{rmse}</td><td>—</td></tr>"

    if ml_results["is_classification"]:
        headers = "<th>Model</th><th>Accuracy</th><th>F1-Score</th><th>Precision</th><th>Recall</th>"
    else:
        headers = "<th>Model</th><th>R² Score</th><th>MAE</th><th>RMSE</th><th>—</th>"

    return f"""
    <div class="section">
      <div class="section-title">🤖 ML Training Results</div>
      <div class="alert alert-info">Task: <strong>{task}</strong> · Target: <strong>{target_col or '—'}</strong> · Best model: <strong>{best}</strong></div>
      <div class="table-wrap">
        <table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>
      </div>
    </div>"""


def _readiness_section(readiness: dict) -> str:
    if not readiness:
        return ""
    score = readiness["score"]
    grade = readiness["grade"]
    color = readiness["color"]

    bars = ""
    for cat, val in readiness["breakdown"].items():
        bar_color = "#10B981" if val >= 80 else "#F59E0B" if val >= 55 else "#EF4444"
        bars += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;font-size:0.82rem;">
          <span style="min-width:160px;color:#94A3B8;text-align:right">{cat}</span>
          <div style="flex:1;height:7px;background:rgba(255,255,255,0.07);border-radius:4px;overflow:hidden">
            <div style="height:100%;width:{val}%;background:{bar_color};border-radius:4px"></div>
          </div>
          <span style="color:{bar_color};font-weight:700;min-width:36px">{val:.0f}</span>
        </div>"""

    issues_html = ""
    for iss in readiness.get("issues", []):
        issues_html += f'<div class="alert alert-warning">⚠️ {iss}</div>'
    if not issues_html:
        issues_html = '<div class="alert alert-success">🎉 No critical pipeline issues detected!</div>'

    return f"""
    <div class="section">
      <div class="section-title">📋 ML Readiness Scorecard</div>
      <div class="score-section">
        <div class="score-circle" style="border-color:{color}">
          <div class="score-num" style="color:{color}">{score}</div>
          <div class="score-max">/ 100</div>
        </div>
        <div>
          <div class="score-grade" style="color:{color}">{grade}</div>
          <div class="score-label">ML Readiness Score</div>
          <br>
          {bars}
        </div>
      </div>
      {issues_html}
    </div>"""


def _embed_plotly_fig(fig: go.Figure, div_id: str) -> tuple[str, str]:
    """Returns (div_html, script_html) for embedding a plotly figure."""
    fig_json = fig.to_json()
    div = f'<div class="chart-container"><div id="{div_id}"></div></div>'
    script = f'Plotly.newPlot("{div_id}", {fig_json});'
    return div, script


def generate_html_report(
    profile: dict,
    col_types: dict,
    filename: str,
    stage: str,
    cleaning_changelog: list[str],
    engineering_changelog: list[str],
    ml_results: Optional[dict],
    readiness: Optional[dict],
    target_col: Optional[str],
    charts: Optional[dict[str, go.Figure]] = None,
) -> str:
    """
    Generate a complete self-contained HTML report.
    Returns the HTML string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Metrics
    qs = profile["quality_score"]
    q_color = "#10B981" if qs >= 70 else "#F59E0B" if qs >= 50 else "#EF4444"
    metrics_html = (
        _metric_card("Rows", f"{profile['n_rows']:,}") +
        _metric_card("Columns", str(profile["n_cols"])) +
        _metric_card("Missing Cells", f"{profile['total_nulls']:,}", f"{profile['null_pct']:.1f}%") +
        _metric_card("Duplicates", f"{profile['duplicate_rows']:,}", f"{profile['duplicate_pct']:.1f}%") +
        _metric_card("Memory", f"{profile['memory_mb']:.2f} MB") +
        _metric_card("Quality Score", f'<span style="color:{q_color}">{qs}/100</span>')
    )

    # Column stats table
    col_stats_table = _col_stats_table(profile, col_types)

    # Charts
    charts_html_parts = []
    plotly_scripts_parts = []
    if charts:
        charts_html_parts.append('<div class="section"><div class="section-title">📊 Analysis Charts</div>')
        for i, (chart_title, fig) in enumerate(charts.items()):
            try:
                div_id = f"chart_{i}"
                div, script = _embed_plotly_fig(fig, div_id)
                charts_html_parts.append(div)
                plotly_scripts_parts.append(script)
            except Exception:
                pass
        charts_html_parts.append("</div>")

    # Sections
    cleaning_section = _log_section("Cleaning Operations Log", "🧹", cleaning_changelog)
    engineering_section = _log_section("Feature Engineering Log", "⚙️", engineering_changelog)
    ml_section = _ml_results_section(ml_results or {}, target_col or "—")
    readiness_section = _readiness_section(readiness or {})

    html = _HTML_TEMPLATE.format(
        filename=filename or "Unknown",
        generated_at=now,
        stage=stage,
        metrics_html=metrics_html,
        col_stats_table=col_stats_table,
        charts_html="\n".join(charts_html_parts),
        cleaning_section=cleaning_section,
        engineering_section=engineering_section,
        ml_section=ml_section,
        readiness_section=readiness_section,
        plotly_scripts="\n".join(plotly_scripts_parts),
    )
    return html
