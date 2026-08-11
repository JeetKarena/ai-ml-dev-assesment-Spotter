"""Diagnostic visualization utilities for model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_residuals(
    y_true,
    y_pred,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot predicted vs. actual rates with a residual distribution panel.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        output_path: If supplied, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_scatter.scatter(actual, predicted, alpha=0.3, s=8, color="#2874A6")
    lo, hi = actual.min(), actual.max()
    ax_scatter.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect fit")
    ax_scatter.set_xlabel("Actual rate ($)")
    ax_scatter.set_ylabel("Predicted rate ($)")
    ax_scatter.set_title("Predicted vs. Actual")
    ax_scatter.legend()
    ax_scatter.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    ax_hist.hist(residuals, bins=60, color="#2874A6", edgecolor="white", linewidth=0.4)
    ax_hist.axvline(0, color="red", linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Residual ($)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Residual Distribution")
    ax_hist.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax_hist.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig


def plot_december_curve(
    dates: pd.Series,
    predicted_rates: pd.Series,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the December 2025 predicted rate curve for the fixed route.

    Args:
        dates: Series of datetime dates (Dec 1-31).
        predicted_rates: Corresponding predicted USD rates.
        output_path: If supplied, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    color = "#064A56"
    ax.plot(dates, predicted_rates, color=color, linewidth=2.4, marker="o", markersize=3.5)
    floor = float(predicted_rates.min())
    ax.fill_between(dates, predicted_rates, floor - max(10.0, floor * 0.02), color=color, alpha=0.08)
    ax.set_title("December 2025 Predicted Load Rate", loc="left", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Predicted rate ($)")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig
