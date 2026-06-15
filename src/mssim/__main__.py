#!/usr/bin/env python3
"""
Note:

python main.py --settings config/example_settings.json \\
                --n_qubits 8 --depth 4 --engine tn \\
                [--n_runs 20] [--max_bond 32] [--run_id 0]

All CLI arguments override the corresponding values in the settings file,
so SLURM only needs to pass the parameters that vary across
array tasks (like ``--n_qubits``, ``--depth``, ``--engine``).
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from mssim.runner import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("mssim.main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:

    p = argparse.ArgumentParser(
        description="mssim: Multi System Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--settings", required=True,
        help="Path to the JSON settings file.",
    )
    p.add_argument(
        "--n_qubits", type=int, default=None,
        help="Number of qubits (overrides settings).",
    )
    p.add_argument(
        "--depth", type=int, default=None,
        help="Circuit depth (overrides settings).",
    )
    p.add_argument(
        "--observable", type=str, default=None,
        help="Observable string such as 'ZIII' (overrides settings).",
    )
    p.add_argument(
        "--engine", type=str, default=None,
        help="Comma-separated engine keys, e.g. 'tn,sv' or 'all' (overrides settings).",
    )
    p.add_argument(
        "--n_runs", type=int, default=None,
        help="Number of random-parameter trials per engine (overrides settings).",
    )
    p.add_argument(
        "--max_bond", type=int, default=None,
        help="Maximum MPS bond dimension (overrides settings).",
    )
    p.add_argument(
        "--max_terms", type=int, default=None,
        help="Maximum number of terms for Pauli propagation (overrides settings).",
    )
    p.add_argument(
        "--run_id", type=int, default=0,
        help="SLURM array task ID, embedded in metadata.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output file path (overrides settings.output.filename).",
    )

    return p.parse_args(argv)


def load_settings(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def merge_args(settings: dict, args: argparse.Namespace) -> dict:
    """
        Ensures the format of the settings dictonary is correct and applies command line overrides on top of the settings dictionary
    """
    cfg = dict(settings)            # shallow copy to ensure the format is correct
    # prevent errors in case of misconfiguration
    cfg.setdefault("model", {})
    cfg.setdefault("execution", {})
    cfg.setdefault("sweep", {})
    cfg.setdefault("output", {})

    if args.n_qubits is not None:
        cfg["model"]["n_qubits"] = args.n_qubits
    if args.depth is not None:
        cfg["model"]["depth"] = args.depth
    if args.observable is not None:
        cfg["model"]["observable"] = args.observable
    if args.engine is not None:
        cfg["execution"]["engine"] = args.engine
    if args.n_runs is not None:
        cfg["execution"]["n_runs"] = args.n_runs
    if args.max_bond is not None:
        cfg["execution"]["max_bond_dimension"] = args.max_bond
    elif cfg["execution"].get("max_bond_dimension") is None and cfg["sweep"].get("max_bond_dimension") is not None:
        cfg["execution"]["max_bond_dimension"] = cfg["sweep"]["max_bond_dimension"]
    if args.max_terms is not None:
        cfg["execution"]["max_terms"] = args.max_terms
    elif cfg["execution"].get("max_terms") is None and cfg["sweep"].get("max_terms") is not None:
        cfg["execution"]["max_terms"] = cfg["sweep"]["max_terms"]
    if args.output is not None:
        cfg["output"]["filename"] = args.output

    return cfg


def main(argv: list[str] | None = None) -> None:
    """
        Parses command-line arguments, loads settings and calls the simulation subroutine.

        Args:
            argv (list[str] | None): Optional list of command-line arguments. If None, uses sys.argv.
    """

    # Arguments can come both from the json setting file and from the command line (in the bash script). The latter have priority.
    # Calling parse_args for getting the arguments in json file as a dictionary and merging it with the command line ones.
    # The arguments are then used to build the model, the engines and the output directory.
    args = parse_args(argv)
    settings = merge_args(load_settings(args.settings), args)

    # Build simulation config from settings
    sim_config = {
        "circuit": settings["model"]["circuit"],
        "n_qubits": settings["model"]["n_qubits"],
        "depth": settings["model"]["depth"],
        "observable": settings["model"].get("observable", None),
        "kwargs": settings["model"].get("kwargs", {}),
        "engine": settings["execution"].get("engine", "all"),
        "max_bond_dimension": settings["execution"].get("max_bond_dimension", None),
        "max_terms": settings["execution"].get("max_terms", None),
        "n_runs": settings["execution"].get("n_runs", 1),
        "verbose": settings["output"].get("verbose", False),
        "filename": settings["output"].get("filename", "results.jsonl"),
        "format": settings["output"].get("format", "jsonl"),
        "run_id": args.run_id,
        "settings_file": args.settings,
    }

    run_simulation(sim_config)


if __name__ == "__main__":
    main()
