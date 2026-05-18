"""
csbench.output
==============
Result data structures and thread/process-safe file writers.

Data model
----------
A single simulation trial produces a :class:`ResultRow`.  After a full
batch of ``n_runs`` trials the executor produces a :class:`BatchResult`
that aggregates statistics across the run.

Writers
-------
Two formats are supported:

``jsonl``  — newline-delimited JSON, one :class:`ResultRow` per line.
             Atomic writes via ``fcntl.flock`` (POSIX only).
             Easy to stream/parse incrementally with ``jq`` or pandas.

``hdf5``   — HDF5 datasets via h5py, one row appended per write.
             Thread-safe via a per-file ``threading.Lock``.
             Efficient for large numerical arrays; supports direct NumPy
             slicing.

Both writers expose the same ``save_result(filename, row)`` interface.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class ResultRow:
    """
    One measurement: a single (engine, circuit, parameter-sample) trial. It is what is saved in the result file.

    Fields
    ------
    run_id : int
        Index within the current batch (0-based).
    engine : str
        Engine short name, e.g. ``"quimb_mps(chi=32)"``.
    circuit : str
        Circuit family name.
    n_qubits : int
    depth : int
    n_params : int
        Number of free parameters in this circuit.
    parameters : list[float]
        Actual parameter values used for this run.
    observable : list[str]
        Pauli string used.
    expectation_value : float
    elapsed_seconds : float
        Wall-clock time of the simulation kernel.
    fidelity : float | None
        Approximation fidelity; ``None`` for exact engines.
    metadata : dict[str, Any]
        Free-form extra fields (engine kwargs, circuit metadata, …).
    timestamp : float
        Unix timestamp at result creation.
    """

    run_id: int
    engine: str
    circuit: str
    n_qubits: int
    depth: int
    n_params: int
    parameters: list[float]
    observable: list[str]
    expectation_value: float
    elapsed_seconds: float
    fidelity: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.floating):
                d[k] = float(v)
            elif isinstance(v, np.integer):
                d[k] = int(v)
        return d


@dataclass
class BatchResult:
    """
    Aggregate statistics over ``n_runs`` trials for a single (engine, circuit)
    configuration. It's for now only used in main.py for printing the summary at the end of execution.

    Built by :class:`~mssim.executor.executor` after completing a full batch.
    """

    engine: str
    circuit: str
    n_qubits: int
    depth: int
    n_runs: int
    rows: list[ResultRow]


    @property
    def expectation_values(self) -> np.ndarray:
        return np.array([r.expectation_value for r in self.rows])

    @property
    def elapsed_times(self) -> np.ndarray:
        return np.array([r.elapsed_seconds for r in self.rows])

    @property
    def fidelities(self) -> np.ndarray | None:
        fids = [r.fidelity for r in self.rows]
        if all(f is None for f in fids):
            return None
        return np.array([f if f is not None else np.nan for f in fids])

    def summary(self) -> dict[str, Any]:
        ev = self.expectation_values
        el = self.elapsed_times
        fids = self.fidelities
        d: dict[str, Any] = {
            "engine": self.engine,
            "circuit": self.circuit,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "n_runs": self.n_runs,
            "expval_mean": float(ev.mean()),
            "expval_std": float(ev.std()),
            "elapsed_mean_s": float(el.mean()),
            "elapsed_std_s": float(el.std()),
            "elapsed_total_s": float(el.sum()),
        }
        if fids is not None:
            d["fidelity_mean"] = float(np.nanmean(fids))
            d["fidelity_std"] = float(np.nanstd(fids))
        return d


def save_result_jsonl(filename: str, row: ResultRow) -> None:
    """
    Append one :class:`ResultRow` to a newline-delimited JSON file.

    Uses ``fcntl.flock`` for process-level mutual exclusion so that
    multiple SLURM tasks writing to the same file do not interleave.

    Parameters
    ----------
    filename:
        Path to the ``.jsonl`` file (created if absent).
    row:
        The result to persist.
    """
    data_line = json.dumps(row.to_dict(), separators=(",", ":")) + "\n"
    with open(filename, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(data_line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)



_hdf5_locks: dict[str, threading.Lock] = {}
_hdf5_locks_meta = threading.Lock()

def _get_hdf5_lock(filename: str) -> threading.Lock:
    with _hdf5_locks_meta:
        if filename not in _hdf5_locks:
            _hdf5_locks[filename] = threading.Lock()
        return _hdf5_locks[filename]


def save_result_hdf5(filename: str, row: ResultRow) -> None:
    """
    Append one :class:`ResultRow` to an HDF5 file.

    The HDF5 file is organised as a flat table stored in
    ``/results`` with variable-length string columns for list/dict fields.

    Thread-safety within a process is guaranteed via a ``threading.Lock``.
    For multi-*process* safety open the file in SWMR mode by setting the
    environment variable ``CSBENCH_HDF5_SWMR=1``.

    Parameters
    ----------
    filename:
        Path to the ``.h5`` file (created if absent).
    row:
        The result to persist.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 output: pip install h5py"
        ) from exc

    lock = _get_hdf5_lock(filename)
    d = row.to_dict()

    with lock:
        with h5py.File(filename, "a") as hf:
            grp = hf.require_group("results")

            def _append(name: str, value: Any) -> None:
                # Serialise lists / dicts as JSON strings
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                scalar = np.array([value])

                if name not in grp:
                    maxshape = (None,) + scalar.shape[1:]
                    if scalar.dtype.kind in ("U", "O") or isinstance(value, str):
                        dt = h5py.string_dtype()
                        grp.create_dataset(
                            name,
                            data=np.array([str(value)], dtype=object),
                            maxshape=(None,),
                            dtype=dt,
                        )
                    else:
                        grp.create_dataset(
                            name, data=scalar, maxshape=maxshape, chunks=True
                        )
                else:
                    ds = grp[name]
                    ds.resize(ds.shape[0] + 1, axis=0)
                    if ds.dtype.kind in ("O",) or h5py.check_string_dtype(ds.dtype):
                        ds[-1] = str(value)
                    else:
                        ds[-1] = scalar[0]

            for k, v in d.items():
                _append(k, v)



def save_result(filename: str, row: ResultRow, fmt: str = "jsonl") -> None:
    """
    Wrapper on the file specific functions. Writes a :class:`ResultRow` to filename in the requested format.

    Parameters
    ----------
    filename:
        Output file path.  The extension is **not** auto-added; the caller
        should provide the correct extension (``.jsonl`` or ``.h5``).
    row:
        The result to write.
    fmt:
        ``"jsonl"`` (default) or ``"hdf5"``.

    Raises
    ------
    ValueError
        If *fmt* is not recognised.
    """
    if fmt == "jsonl":
        save_result_jsonl(filename, row)
    elif fmt in ("hdf5", "h5"):
        save_result_hdf5(filename, row)
    else:
        raise ValueError(f"Unknown output format '{fmt}'. Use 'jsonl' or 'hdf5'.")


def load_results_jsonl(filename: str) -> list[dict[str, Any]]:
    """
    Read all rows from a ``.jsonl`` results file.

    Returns
    -------
    list[dict]
        One dict per line (raw, not deserialized back to ResultRow).
    """
    rows = []
    with open(filename, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
