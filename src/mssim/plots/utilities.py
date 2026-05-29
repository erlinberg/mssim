"""Shared plotting utilities.

These helpers are reused by observable-specific plotting modules.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ENGINE_QUALITY_PARAM: dict[str, str | None] = {
    "mpstab":    "max_bond_dimension",
    "quimb":     "max_bond_dimension",
    "pauliprop": "max_terms",
    "qiskit":    None,
}

ENGINE_STYLE: dict[str, dict] = {
    "qiskit":    {"linestyle": "-",  "marker": "o"},
    "mpstab":    {"linestyle": "--", "marker": "s"},
    "quimb":     {"linestyle": ":",  "marker": "^"},
    "pauliprop": {"linestyle": "-.", "marker": "D"},
}

ENGINE_COLORS: dict[str, str] = {
    "qiskit":    "#2196F3",
    "mpstab":    "#E53935",
    "quimb":     "#43A047",
    "pauliprop": "#FB8C00",
}


def resolve_engines(
    df: pd.DataFrame,
    engines: str | list[str] | None,
    *,
    exclude_qiskit: bool,
) -> list[str]:
    if engines is None:
        return [
            e for e in df["engine"].unique()
            if not (exclude_qiskit and e == "qiskit")
        ]
    if isinstance(engines, str):
        return [engines]
    return list(engines)


def filter_engine(
    df: pd.DataFrame,
    engine: str,
    common: dict,
    engine_extra: dict,
) -> pd.DataFrame:
    """Apply common and engine-specific filters to one engine slice."""
    sub = df[df["engine"] == engine].copy()
    for col, val in {**common, **engine_extra}.items():
        if col not in sub.columns:
            continue
        if val is None:
            continue
        if isinstance(val, float) and np.isnan(val):
            sub = sub[sub[col].isna()]
        else:
            sub = sub[sub[col] == val]
    return sub


def build_combined(
    mag_df: pd.DataFrame,
    engines: list[str],
    common: dict,
    max_bond_dimension: float | None,
    max_terms: float | None,
) -> pd.DataFrame:
    """Build a filtered dataframe by concatenating per-engine slices."""
    parts: list[pd.DataFrame] = []
    for eng in engines:
        qp = ENGINE_QUALITY_PARAM.get(eng)
        engine_extra: dict = {}
        if qp == "max_bond_dimension" and max_bond_dimension is not None:
            engine_extra = {"max_bond_dimension": max_bond_dimension}
        elif qp == "max_terms" and max_terms is not None:
            engine_extra = {"max_terms": max_terms}

        sub = filter_engine(mag_df, eng, common, engine_extra)
        if sub.empty:
            warnings.warn(
                f"No data for engine='{eng}' with filters {engine_extra | common}.",
                stacklevel=3,
            )
        parts.append(sub)

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        raise ValueError("No data matched after filtering. Check parameter values.")
    return pd.concat(non_empty, ignore_index=True)


def fmt_val(v: object) -> str:
    """Pretty-print values for plot titles."""
    if not isinstance(v, float):
        return str(v)
    unit = np.pi / 8
    ratio = v / unit
    n = round(ratio)
    if abs(ratio - n) < 1e-9 and n != 0:
        from math import gcd

        g = gcd(abs(n), 8)
        num, den = n // g, 8 // g
        if den == 1:
            return f"{num}π" if num != 1 else "π"
        return f"{num}π/{den}" if num != 1 else f"π/{den}"
    return f"{v:.4g}"


def build_title(
    engines: list[str],
    common: dict,
    max_bond_dimension: float | None,
    max_terms: float | None,
) -> str:
    """Build the shared title used by plotting scripts."""
    circuit = common.get("circuit", "")
    circuit_str = circuit.replace("_", " ").title() if circuit else ""

    phys_keys = [k for k in ("J", "h", "b") if k in common and not pd.isna(common[k])]
    phys_str = "   ".join(f"{k} = {fmt_val(common[k])}" for k in phys_keys)

    line1_parts = [s for s in (circuit_str, phys_str) if s]
    line1 = "     ".join(line1_parts)

    quality_parts: list[str] = []
    if any(ENGINE_QUALITY_PARAM.get(e) == "max_bond_dimension" for e in engines):
        if max_bond_dimension is not None:
            quality_parts.append(f"χ = {int(max_bond_dimension)}")
    if any(ENGINE_QUALITY_PARAM.get(e) == "max_terms" for e in engines):
        if max_terms is not None:
            quality_parts.append(f"max_terms = {int(max_terms):,}")
    line2 = "   ·   ".join(quality_parts)

    return "\n".join(p for p in (line1, line2) if p)


def style_ax(ax: plt.Axes, x_col: str, y_label: str, df: pd.DataFrame) -> None:
    if df[x_col].dtype.kind in "iu":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(x_col.replace("_", " ").capitalize(), fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)


def add_legend(
    ax: plt.Axes,
    engines: list[str],
    hue_param: str | None,
    hue_values: list | None,
    hue_colors: list,
) -> None:
    """Render the engine and hue legend blocks outside the axes."""
    handles: list[mlines.Line2D] = []

    def _divider(label: str) -> mlines.Line2D:
        return mlines.Line2D([], [], color="none", linewidth=0, label=label)

    handles.append(_divider("── Engine ──"))
    for eng in engines:
        st = ENGINE_STYLE.get(eng, {"linestyle": "-", "marker": "x"})
        color = "dimgrey" if hue_param else ENGINE_COLORS.get(eng, "#9C27B0")
        handles.append(mlines.Line2D(
            [], [], color=color,
            linestyle=st["linestyle"], marker=st["marker"],
            markersize=7, linewidth=2, label=eng,
        ))

    if hue_param and hue_values:
        label = hue_param.replace("_", " ").capitalize()
        handles.append(_divider(f"── {label} ──"))
        for val, col in zip(hue_values, hue_colors):
            handles.append(mlines.Line2D(
                [], [], color=col, linewidth=3, marker="None", label=str(val),
            ))

    ax.legend(
        handles=handles,
        frameon=True, fontsize=8,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, edgecolor="#cccccc",
    )


def draw_curves(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    engines: list[str],
    hue_param: str | None,
    hue_values: list | None,
    palette: list | None,
) -> None:
    """Draw mean ± 1 std-dev shaded curves onto *ax*."""
    for eng in engines:
        sub = data[data["engine"] == eng]
        if sub.empty:
            continue
        st = ENGINE_STYLE.get(eng, {"linestyle": "-", "marker": "x"})

        if hue_param and hue_values and palette:
            for hv, color in zip(hue_values, palette):
                grp = sub[sub[hue_param] == hv].groupby(x_col)[y_col]
                if grp.ngroups == 0:
                    continue
                xs = grp.mean().index.to_numpy()
                ym = grp.mean().to_numpy()
                ys = grp.std().fillna(0).to_numpy()
                ax.plot(xs, ym, color=color, **st, linewidth=2, markersize=7)
                ax.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.12)
        else:
            color = ENGINE_COLORS.get(eng, "#9C27B0")
            grp = sub.groupby(x_col)[y_col]
            xs = grp.mean().index.to_numpy()
            ym = grp.mean().to_numpy()
            ys = grp.std().fillna(0).to_numpy()
            ax.plot(xs, ym, color=color, **st, linewidth=2, markersize=7, label=eng)
            ax.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.12)


def add_zoom_inset(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    engines: list[str],
    hue_param: str | None,
    hue_values: list | None,
    palette: list | None,
    zoom_ylim: tuple[float, float],
) -> None:
    """Add a zoomed inset to the main axes."""
    axins = ax.inset_axes([1.20, 0.0, 0.42, 0.38])
    draw_curves(axins, data, x_col, y_col, engines, hue_param, hue_values, palette)

    if y_col != "magnetization":
        axins.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

    axins.set_ylim(*zoom_ylim)
    axins.set_xlim(ax.get_xlim())
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.tick_params(labelsize=7)
    axins.set_title("near-zero zoom", fontsize=7, pad=3, color="#555555", style="italic")
    axins.grid(True, alpha=0.2, linewidth=0.4)
    axins.spines[["top", "right"]].set_visible(False)
    ax.indicate_inset_zoom(axins, edgecolor="#888888", linewidth=0.8, alpha=0.7)


def auto_zoom_ylim(
    df: pd.DataFrame,
    y_col: str,
    quantile_hi: float = 0.20,
) -> tuple[float, float] | None:
    """Return a zoom window for the lower part of the y-range."""
    ymin = df[y_col].min()
    ymax = df[y_col].max()
    span = ymax - ymin
    if span < 1e-12:
        return None
    hi = ymin + quantile_hi * span
    if hi >= ymax * 0.85:
        return None
    pad = 0.03 * span
    return (ymin - pad, hi + pad)


def save_figure(fig: plt.Figure, output_file: str | Path | None, default_name: str) -> Path:
    """Save a figure to the requested path, creating parent directories."""
    output_path = Path(output_file) if output_file is not None else Path(default_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path