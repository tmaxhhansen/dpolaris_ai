"""
Backtest reporting helpers (markdown/html with optional plots).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _try_make_plots(
    *,
    artifact_dir: Path,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if equity_df is None or equity_df.empty:
        return paths

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return paths

    # Equity curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_df["timestamp"], equity_df["equity"], color="#0b84f3", linewidth=1.8)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    equity_path = artifact_dir / "equity_curve.png"
    fig.savefig(equity_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["equity_curve"] = equity_path.name

    # Drawdown curve
    if "drawdown" in equity_df.columns:
        fig, ax = plt.subplots(figsize=(10, 3))
        dd = pd.to_numeric(equity_df["drawdown"], errors="coerce").fillna(0.0)
        ax.fill_between(equity_df["timestamp"], dd * 100.0, 0.0, color="#d62728", alpha=0.35)
        ax.plot(equity_df["timestamp"], dd * 100.0, color="#d62728", linewidth=1.2)
        ax.set_title("Drawdown (%)")
        ax.set_ylabel("%")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        dd_path = artifact_dir / "drawdown.png"
        fig.savefig(dd_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths["drawdown"] = dd_path.name

    # Trade distribution
    if trades_df is not None and not trades_df.empty and "pnl" in trades_df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
        ax.hist(pnl, bins=min(30, max(8, int(np.sqrt(len(pnl))))), color="#2ca02c", alpha=0.75)
        ax.set_title("Trade PnL Distribution")
        ax.set_xlabel("PnL")
        ax.grid(alpha=0.25)
        dist_path = artifact_dir / "trade_pnl_distribution.png"
        fig.savefig(dist_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths["trade_pnl_distribution"] = dist_path.name

    return paths


def _df_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_None_"
    view = df.head(max_rows)
    try:
        return view.to_markdown(index=False)
    except Exception:
        return "```\n" + view.to_string(index=False) + "\n```"


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    keys = [
        "initial_equity",
        "final_equity",
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "exposure",
        "turnover",
        "trade_count",
        "win_rate",
        "profit_factor",
        "average_win",
        "average_loss",
        "commission_total",
        "slippage_total",
        "cost_total",
        "gross_pnl",
        "net_pnl",
    ]
    rows = []
    for k in keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, float):
            rows.append((k, f"{v:.6f}"))
        else:
            rows.append((k, str(v)))
    table = pd.DataFrame(rows, columns=["metric", "value"])
    return _df_markdown(table, max_rows=len(table))


def generate_backtest_report(
    *,
    artifact_dir: Path | str,
    metrics: dict[str, Any],
    config_snapshot: dict[str, Any],
    equity_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    attribution_df: pd.DataFrame,
    title: str = "Backtest Report",
) -> dict[str, str]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = _try_make_plots(artifact_dir=artifact_dir, equity_df=equity_df, trades_df=trades_df)
    cfg = config_snapshot.get("config", {})
    strategy_info = config_snapshot.get("strategy", {})

    md_lines: list[str] = []
    md_lines.append(f"# {title}")
    md_lines.append("")
    md_lines.append("## Summary Metrics")
    md_lines.append("")
    md_lines.append(_metrics_markdown(metrics))
    md_lines.append("")

    md_lines.append("## Execution Assumptions")
    md_lines.append("")
    assumptions = pd.DataFrame(
        [(k, cfg[k]) for k in sorted(cfg.keys())],
        columns=["parameter", "value"],
    )
    md_lines.append(_df_markdown(assumptions, max_rows=200))
    md_lines.append("")

    md_lines.append("## Strategy")
    md_lines.append("")
    strategy_tbl = pd.DataFrame(
        [
            ("class", strategy_info.get("class")),
            ("module", strategy_info.get("module")),
        ],
        columns=["field", "value"],
    )
    md_lines.append(_df_markdown(strategy_tbl, max_rows=10))
    md_lines.append("")

    if plot_paths:
        md_lines.append("## Plots")
        md_lines.append("")
        for key in ["equity_curve", "drawdown", "trade_pnl_distribution"]:
            if key in plot_paths:
                md_lines.append(f"### {key.replace('_', ' ').title()}")
                md_lines.append("")
                md_lines.append(f"![{key}]({plot_paths[key]})")
                md_lines.append("")

    md_lines.append("## Trade Attribution")
    md_lines.append("")
    md_lines.append(_df_markdown(attribution_df, max_rows=50))
    md_lines.append("")

    md_lines.append("## Trades")
    md_lines.append("")
    md_lines.append(_df_markdown(trades_df, max_rows=50))
    md_lines.append("")

    md_lines.append("## Orders")
    md_lines.append("")
    md_lines.append(_df_markdown(orders_df, max_rows=50))
    md_lines.append("")

    report_md = "\n".join(md_lines).strip() + "\n"
    md_path = artifact_dir / "report.md"
    with open(md_path, "w") as f:
        f.write(report_md)

    # Basic HTML output without external dependencies.
    html_parts: list[str] = []
    html_parts.append("<html><head><meta charset='utf-8'><title>Backtest Report</title>")
    html_parts.append(
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif;padding:20px;max-width:1200px;margin:auto;} table{border-collapse:collapse;width:100%;margin:10px 0;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f5f5f5;} h1,h2,h3{margin-top:24px;} img{max-width:100%;height:auto;border:1px solid #ddd;}</style>"
    )
    html_parts.append("</head><body>")
    html_parts.append(f"<h1>{title}</h1>")
    html_parts.append("<h2>Summary Metrics</h2>")
    html_parts.append(pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).to_html(index=False))
    html_parts.append("<h2>Execution Assumptions</h2>")
    html_parts.append(assumptions.to_html(index=False))
    html_parts.append("<h2>Trade Attribution</h2>")
    html_parts.append(attribution_df.to_html(index=False) if not attribution_df.empty else "<p>None</p>")
    if plot_paths:
        html_parts.append("<h2>Plots</h2>")
        for key in ["equity_curve", "drawdown", "trade_pnl_distribution"]:
            if key in plot_paths:
                html_parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>")
                html_parts.append(f"<img src='{plot_paths[key]}' alt='{key}' />")
    html_parts.append("<h2>Trades</h2>")
    html_parts.append(trades_df.to_html(index=False) if not trades_df.empty else "<p>None</p>")
    html_parts.append("<h2>Orders</h2>")
    html_parts.append(orders_df.to_html(index=False) if not orders_df.empty else "<p>None</p>")
    html_parts.append("</body></html>")

    html_path = artifact_dir / "report.html"
    with open(html_path, "w") as f:
        f.write("\n".join(html_parts))

    return {
        "markdown": str(md_path),
        "html": str(html_path),
        **{k: str(artifact_dir / v) for k, v in plot_paths.items()},
    }

