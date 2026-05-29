"""
tests/test_basic.py
====================
Lightweight unit tests that do NOT require any quantum simulation backend
(qibo, quimb, qiskit, mpstab) so they can run in CI with minimal deps.

Run with:  pytest tests/
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# CircuitModel & library
# ---------------------------------------------------------------------------


def test_circuit_model_observable_length_check():
    from mssim.circuits.model import CircuitModel

    with pytest.raises(ValueError, match="Observable length"):
        CircuitModel(
            name="test",
            n_qubits=4,
            depth=1,
            qasm="OPENQASM 2.0;",
            observable=["Z", "Z"],   # too short
            n_params=0,
        )


def test_build_circuit_random_rx():
    from mssim.circuits.library import build_circuit

    model = build_circuit("random_rx", n_qubits=4, depth=2)
    assert model.n_qubits == 4
    assert model.depth == 2
    assert len(model.observable) == 4
    assert model.n_params == 4 * 2  # n_qubits × depth

    params = model.sample_parameters()
    assert len(params) == model.n_params
    assert all(0.0 <= p < 2 * np.pi for p in params)


def test_build_circuit_ising():
    from mssim.circuits.library import build_circuit

    model = build_circuit("ising", n_qubits=6, depth=3, J=1.0, h=0.5, dt=0.05)
    assert model.n_params == 0
    assert model.sample_parameters() == []
    assert "OPENQASM" in model.qasm


def test_build_circuit_qaoa():
    from mssim.circuits.library import build_circuit

    model = build_circuit("qaoa", n_qubits=5, depth=2)
    assert model.n_params == 2 * 2  # 2 * depth


def test_build_circuit_hardware_efficient():
    from mssim.circuits.library import build_circuit

    model = build_circuit("hardware_efficient", n_qubits=4, depth=3)
    assert model.n_params == 4 * (3 + 1)


def test_unknown_circuit():
    from mssim.circuits.library import build_circuit

    with pytest.raises(KeyError, match="not_a_circuit"):
        build_circuit("not_a_circuit", n_qubits=4, depth=2)


# ---------------------------------------------------------------------------
# Output: ResultRow serialisation
# ---------------------------------------------------------------------------


def _make_row(**overrides):
    from mssim.output import ResultRow

    defaults = dict(
        run_id=0,
        engine="test_engine",
        circuit="random_rx",
        n_qubits=4,
        depth=2,
        n_params=8,
        parameters=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        observable=["Z", "I", "Z", "I"],
        expectation_value=0.5,
        elapsed_seconds=0.01,
        fidelity=0.99,
    )
    defaults.update(overrides)
    return ResultRow(**defaults)


def test_result_row_to_dict():
    row = _make_row()
    d = row.to_dict()
    assert d["engine"] == "test_engine"
    assert d["expectation_value"] == 0.5
    assert isinstance(d["parameters"], list)


def test_result_row_fidelity_none():
    row = _make_row(fidelity=None)
    d = row.to_dict()
    assert d["fidelity"] is None


def test_save_and_load_jsonl():
    from mssim.output import save_result, load_results_jsonl

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        fname = f.name

    try:
        for i in range(5):
            save_result(fname, _make_row(run_id=i, expectation_value=float(i)), fmt="jsonl")

        rows = load_results_jsonl(fname)
        assert len(rows) == 5
        assert [r["run_id"] for r in rows] == list(range(5))
    finally:
        os.unlink(fname)


def test_batch_result_summary():
    from mssim.output import BatchResult

    rows = [_make_row(run_id=i, expectation_value=float(i) * 0.1) for i in range(10)]
    br = BatchResult(
        engine="test",
        circuit="random_rx",
        n_qubits=4,
        depth=2,
        n_runs=10,
        rows=rows,
    )
    s = br.summary()
    assert s["n_runs"] == 10
    assert "expval_mean" in s
    assert "fidelity_mean" in s


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


def test_unknown_engine():
    from mssim.engines.library import build_engines

    with pytest.raises(KeyError, match="not_an_engine"):
        build_engines(["not_an_engine"])


def test_build_engines_empty_list():
    from mssim.engines.library import build_engines

    engines = build_engines([])
    assert engines == []


def test_plot_magnetization_excludes_one_hue_value(tmp_path, monkeypatch):
    import importlib
    import pandas as pd
    import matplotlib.pyplot as plt

    mag_plot = importlib.import_module("mssim.plots.magnetization")

    df = pd.DataFrame(
        [
            {"engine": "qiskit", "circuit": "demo", "n_qubits": 4, "depth": 1, "run_id": 0, "max_bond_dimension": None, "max_terms": None, "J": 1.0, "h": 0.5, "b": 0.25, "magnetization": 0.1},
            {"engine": "qiskit", "circuit": "demo", "n_qubits": 4, "depth": 2, "run_id": 0, "max_bond_dimension": None, "max_terms": None, "J": 1.0, "h": 0.5, "b": 0.25, "magnetization": 0.2},
            {"engine": "qiskit", "circuit": "demo", "n_qubits": 5, "depth": 3, "run_id": 0, "max_bond_dimension": None, "max_terms": None, "J": 1.0, "h": 0.5, "b": 0.25, "magnetization": 0.3},
        ]
    )

    captured = {}
    fig, ax = plt.subplots()

    def fake_draw_curves(ax, data, x_col, y_col, engines, hue_param, hue_values, palette):
        captured["data"] = data.copy()
        captured["hue_values"] = list(hue_values) if hue_values is not None else None

    monkeypatch.setattr(mag_plot, "draw_curves", fake_draw_curves)
    monkeypatch.setattr(mag_plot, "add_legend", lambda *args, **kwargs: None)
    monkeypatch.setattr(mag_plot, "add_zoom_inset", lambda *args, **kwargs: None)
    monkeypatch.setattr(mag_plot, "save_figure", lambda fig, output_file, default_name: tmp_path / default_name)
    monkeypatch.setattr(mag_plot.plt, "subplots", lambda *args, **kwargs: (fig, ax))

    mag_plot.plot_magnetization(
        df,
        x_axis="n_qubits",
        hue_param="depth",
        exclude_hue_value=[1, 2],
        output_file=tmp_path / "plot.png",
    )

    assert captured["hue_values"] == [3]
    assert set(captured["data"]["depth"].unique()) == {3}
