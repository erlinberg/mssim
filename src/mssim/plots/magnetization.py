"""Magnetization observable analysis and plotting.

Keep this module observable-specific. New observables should get their own
module alongside it so the shared helpers in :mod:`mssim.plots.general` and
:mod:`mssim.plots.utilities` stay reusable.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utilities import (
    add_legend,
    add_zoom_inset,
    auto_zoom_ylim,
    build_combined,
    build_title,
    draw_curves,
    resolve_engines,
    save_figure,
    style_ax,
)

_MAG_GROUP_COLS = [
    "engine", "circuit", "n_qubits", "depth", "run_id",
    "max_bond_dimension", "max_terms", "J", "h", "b",
]

_MERGE_KEYS = ["circuit", "n_qubits", "depth", "run_id", "J", "h", "b"]


def _apply_hue_exclusion(
    data: pd.DataFrame,
    hue_param: str | None,
    exclude_hue_value: object | Sequence[object] | None,
) -> pd.DataFrame:
    """Drop one hue bucket before plotting when requested."""
    if hue_param is None or exclude_hue_value is None or hue_param not in data.columns:
        return data

    if isinstance(exclude_hue_value, Sequence) and not isinstance(exclude_hue_value, (str, bytes)):
        excluded_values = list(exclude_hue_value)
    else:
        excluded_values = [exclude_hue_value]

    filtered = data[~data[hue_param].isin(excluded_values)].copy()
    if filtered.empty:
        raise ValueError(
            f"No data left after excluding hue value(s) {excluded_values!r} from '{hue_param}'."
        )
    return filtered


def magnetization(df: pd.DataFrame, *, inspect: bool = False) -> pd.DataFrame:
    """Compute one magnetization value per run by averaging expectation values."""
    if inspect:
        print("\n── Inspection: first group ──")
        for name, grp in df.groupby(_MAG_GROUP_COLS, dropna=False):
            print(dict(zip(_MAG_GROUP_COLS, name)))
            print(grp[["observable", "expectation_value"]].to_string(index=False))
            print(f"→ magnetization = {grp['expectation_value'].mean():.6f}\n")
            break

    return df.groupby(_MAG_GROUP_COLS, dropna=False, as_index=False).agg(
        magnetization=("expectation_value", "mean")
    )


def plot_magnetization(
    mag_df: pd.DataFrame,
    x_axis: str,
    *,
    hue_param: str | None = None,
    exclude_hue_value: object | Sequence[object] | None = None,
    engines: str | Sequence[str] | None = None,
    max_bond_dimension: float | None = None,
    max_terms: float | None = None,
    zoom_ylim: tuple[float, float] | None = None,
    output_file: str | Path | None = None,
    **common_fixed,
) -> None:
    """Plot magnetization vs `x_axis`.

    Parameters
    ----------
    mag_df:
        Output of :func:`magnetization`.
    x_axis:
        Column for the x-axis.
    hue_param:
        Optional column used to split curves within each engine.
    exclude_hue_value:
        Optional hue value to remove before plotting, reducing the number of curves.
    engines:
        Engine names to include; `None` means all present.
    output_file:
        Output image path. When relative, the launcher resolves it inside the
        top-level `plots/` folder.
    **common_fixed:
        Extra column=value filters applied to every engine.
    """
    engines = resolve_engines(mag_df, engines, exclude_qiskit=False)
    combined = build_combined(mag_df, engines, common_fixed, max_bond_dimension, max_terms)
    combined = _apply_hue_exclusion(combined, hue_param, exclude_hue_value)

    hue_values = sorted(combined[hue_param].dropna().unique().tolist()) if hue_param else None
    palette = sns.color_palette("tab10", len(hue_values)) if hue_values else None

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_curves(ax, combined, x_axis, "magnetization", engines, hue_param, hue_values, palette)
    style_ax(ax, x_axis, "Magnetization", combined)

    ax.set_title(
        build_title(engines, common_fixed, max_bond_dimension, max_terms),
        fontsize=9, loc="left", pad=10,
    )
    add_legend(
        ax, engines, hue_param, hue_values,
        palette or ["#9C27B0" for _ in engines],
    )

    if zoom_ylim is None:
        zoom_ylim = auto_zoom_ylim(combined, "magnetization")
    if zoom_ylim:
        add_zoom_inset(
            ax, combined, x_axis, "magnetization",
            engines, hue_param, hue_values, palette, zoom_ylim,
        )

    save_figure(fig, output_file, "magnetization_plot.png")


def plot_magnetization_diff(
    mag_df: pd.DataFrame,
    x_axis: str,
    *,
    hue_param: str | None = None,
    exclude_hue_value: object | Sequence[object] | None = None,
    engines: str | Sequence[str] | None = None,
    max_bond_dimension: float | None = None,
    max_terms: float | None = None,
    zoom_ylim: tuple[float, float] | None = None,
    output_file: str | Path | None = None,
    **common_fixed,
) -> None:
    """Plot `|M_approx - M_qiskit|` vs `x_axis`.

    `qiskit` is used as the reference and should not be included in `engines`.
    """
    engines = resolve_engines(mag_df, engines, exclude_qiskit=True)
    combined = build_combined(
        mag_df, [*engines, "qiskit"], common_fixed, max_bond_dimension, max_terms
    )
    combined = _apply_hue_exclusion(combined, hue_param, exclude_hue_value)

    df_ref = combined[combined["engine"] == "qiskit"]
    df_approx = combined[combined["engine"].isin(engines)]

    if df_ref.empty:
        raise ValueError("No 'qiskit' data found; cannot compute difference.")

    keys = [k for k in _MERGE_KEYS if k in df_ref.columns and k in df_approx.columns]
    merged = pd.merge(
        df_approx,
        df_ref[keys + ["magnetization"]],
        on=keys,
        suffixes=("", "_qiskit"),
    )
    merged["magnetization_diff"] = (
        merged["magnetization"] - merged["magnetization_qiskit"]
    ).abs()

    hue_values = sorted(merged[hue_param].dropna().unique().tolist()) if hue_param else None
    palette = sns.color_palette("tab10", len(hue_values)) if hue_values else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.4)

    draw_curves(ax, merged, x_axis, "magnetization_diff", engines, hue_param, hue_values, palette)
    style_ax(ax, x_axis, r"$|M_\mathrm{approx} - M_\mathrm{qiskit}|$", merged)

    ax.set_title(
        build_title(engines, common_fixed, max_bond_dimension, max_terms),
        fontsize=9, loc="left", pad=10,
    )
    add_legend(
        ax, engines, hue_param, hue_values,
        palette or ["#9C27B0" for _ in engines],
    )

    if zoom_ylim is None:
        zoom_ylim = auto_zoom_ylim(merged, "magnetization_diff")
    if zoom_ylim:
        add_zoom_inset(
            ax, merged, x_axis, "magnetization_diff",
            engines, hue_param, hue_values, palette, zoom_ylim,
        )

    save_figure(fig, output_file, "magnetization_diff_plot.png")


__all__ = [
    "magnetization",
    "plot_magnetization",
    "plot_magnetization_diff"
]