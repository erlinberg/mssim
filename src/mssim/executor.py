"""
executor

1. Accepts a :class:`~mssim.circuits.model.CircuitModel` and a list of
   :class:`~mssim.engines.abstract.BenchmarkEngine` instances.
2. Runs ``n_runs`` independent trials per engine (each with freshly sampled
   random parameters).
3. Optionally streams every :class:`~mssim.output.ResultRow` to disk as
   it is produced (useful for long cluster jobs).
4. Returns a :class:`~mssim.output.BatchResult` per engine.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any

from .circuits.model import CircuitModel
from .engines.abstract import BenchmarkEngine
from .output import BatchResult, ResultRow, save_result
from .engines.qiskit import QiskitEngine
logger = logging.getLogger(__name__)


@dataclass
class executor:
    """
    Execution manager.

    Parameters
    ----------
    n_runs : int
        Number of independent random-parameter trials per (engine, circuit).
    
    output_file : str | None
        If provided, every :class:`ResultRow` is written to this file as
        it is produced (streaming mode).  Set to ``None`` to accumulate
        results in memory only.
    
    output_fmt : str
        ``"jsonl"`` or ``"hdf5"``.  Ignored when *output_file* is ``None``.
    
    extra_metadata : dict
        Key-value pairs merged into every :class:`ResultRow`'s metadata dict.
        Useful for injecting SLURM job IDs, host names, etc.
    
    skip_on_error : bool
        If ``True``, exceptions raised by an engine are logged and skipped
        rather than re-raised.  Allows a run to continue even when one
        engine fails.
    
    verbose : bool
        If ``True``, log progress at INFO level for every trial.
    """

    n_runs: int = 1
    output_file: str | None = None
    output_fmt: str = "jsonl"
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    skip_on_error: bool = False
    verbose: bool = False


    def run(
        self,
        model: CircuitModel,
        engines: list[BenchmarkEngine],
    ) -> list[BatchResult]:
        """
        Execute the benchmark and return one :class:`BatchResult` per engine.

        Parameters
        ----------
        model:
            The circuit model to benchmark.
        engines:
            List of engine instances to run.

        Returns
        -------
        list[BatchResult]
            One entry per engine, in the same order as *engines*.
        """
        batch_results: list[BatchResult] = []

        for engine in engines:

            if self.verbose: logger.info("Starting engine '%s' for circuit '%s'.", engine.name, model.name)

            rows: list[ResultRow] = []
            for run_id in range(self.n_runs):


                if self.verbose: logger.info("  [%s] run %d/%d …", engine.name, run_id + 1, self.n_runs)

                try:
                    params = model.sample_parameters()
                    qasm_circuit = model.bind_parameters(params)
                    observable = model.observable if isinstance(model.observable, str) else "".join(model.observable)
                    expval, elapsed, fidelity = engine.expectation_value(
                        qasm_circuit=qasm_circuit,
                        observable=observable,
                    )
                except Exception:
                    tb = traceback.format_exc()
                    logger.error("Engine '%s' failed on run %d:\n%s", engine.name, run_id, tb)
                    if self.skip_on_error:
                        continue
                    raise

                row = ResultRow(
                    run_id=run_id,
                    engine=engine.name,
                    circuit=model.name,
                    n_qubits=model.n_qubits,
                    depth=model.depth,
                    max_bond_dimension=int(engine.max_bond_dimension) if hasattr(engine, "max_bond_dimension") else None,
                    max_terms=int(engine.max_terms) if hasattr(engine, "max_terms") else None,
                    n_params=model.n_params,
                    parameters=params,
                    observable=list(observable),
                    expectation_value=float(expval),
                    elapsed_seconds=float(elapsed),
                    fidelity=float(fidelity) if fidelity is not None else None,
                    metadata={**model.metadata, **self.extra_metadata},
                )
                rows.append(row)
                if self.output_file is not None: save_result(self.output_file, row, fmt=self.output_fmt)

                if isinstance(engine, QiskitEngine): break # not collecting statistics on deterministic statevector simulation
            
            if self.verbose: logger.info("Finished engine '%s': %d/%d successful runs.",engine.name,len(rows),self.n_runs)

            batch_results.append(
                BatchResult(
                    engine=engine.name,
                    circuit=model.name,
                    n_qubits=model.n_qubits,
                    depth=model.depth,
                    n_runs=self.n_runs,
                    rows=rows,
                )
            )

        return batch_results
