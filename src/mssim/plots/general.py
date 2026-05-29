"""General plotting I/O helpers.

These helpers are intentionally observable-agnostic so new plotting scripts can
reuse them without depending on a specific analysis pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load(filepath: str | Path) -> pd.DataFrame:
    """Load a JSONL results file and flatten metadata fields (J, h, b)."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            obj = json.loads(line)
            obj["J"] = obj["metadata"].get("J")
            obj["h"] = obj["metadata"].get("h")
            obj["b"] = obj["metadata"].get("b")
            rows.append(obj)

    df = pd.DataFrame(rows)
    drop = ["elapsed_seconds", "fidelity", "metadata", "timestamp", "parameters", "n_params"]
    return df.drop(columns=[c for c in drop if c in df.columns])

def parameter_recap(df: pd.DataFrame, *, print_summary: bool = True) -> pd.DataFrame:
    """Count unique parameter combinations in a plotting dataframe."""
    recap_cols = [
        "circuit", "n_qubits", "depth", "J", "h", "b",
        "engine", "max_bond_dimension", "max_terms",
    ]
    present_cols = [c for c in recap_cols if c in df.columns]

    recap = (
        df.groupby(present_cols, dropna=False)
        .size()
        .reset_index(name="run_count")
        .sort_values(by=present_cols, ascending=True)
    )

    if print_summary:
        print("\n── Parameter Combinations Recap ──")
        print(recap.to_string(index=False))
        print(f"\nTotal unique configurations found: {len(recap)}\n")

    return recap
