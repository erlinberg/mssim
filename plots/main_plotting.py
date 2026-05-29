#!/usr/bin/env python3
"""CLI entrypoint for plotting.

This script stays outside `src/` so plotting can be launched directly from the
top-level `plots/` folder while the reusable implementation lives in the
package under `src/mssim/plots/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mssim.plots.general import load
from mssim.plots.magnetization import magnetization, plot_magnetization, plot_magnetization_diff

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("mssim.plots.main_plotting")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mssim plotting launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "settings",
        nargs="?",
        default="plots/plot_settings.json",
        help="Path to the plotting settings JSON file.",
    )
    return p.parse_args(argv)


def load_settings(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_output_path(settings_path: Path, settings: dict) -> Path:
    output_name = settings.get("output", {}).get("filename", "plot.png")
    output_path = Path(output_name)
    if output_path.is_absolute():
        return output_path
    return settings_path.parent / output_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings_path = Path(args.settings)
    settings = load_settings(settings_path)

    logger.info("Loading plotting settings from %s", settings_path)

    input_file = Path(settings["input_file"])
    if not input_file.is_absolute():
        input_file = settings_path.parent.parent / input_file

    output_path = build_output_path(settings_path, settings)

    df = load(input_file)
    mag_df = magnetization(df)

    common_fixed = dict(settings.get("common_fixed", {}))
    kwargs = dict(
        x_axis=settings["x_axis"],
        hue_param=settings.get("hue_param"),
        engines=settings.get("engines"),
        max_bond_dimension=settings.get("max_bond_dimension"),
        max_terms=settings.get("max_terms"),
        zoom_ylim=tuple(settings["zoom_ylim"]) if settings.get("zoom_ylim") else None,
        output_file=output_path,
        **common_fixed,
    )

    plot_name = settings["plot_function"]
    if plot_name == "plot_magnetization":
        plot_magnetization(mag_df, **kwargs)
    elif plot_name == "plot_magnetization_diff":
        plot_magnetization_diff(mag_df, **kwargs)
    else:
        raise ValueError(f"Unknown plot_function: {plot_name}")

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()