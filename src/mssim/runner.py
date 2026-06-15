from __future__ import annotations

import logging
import os
import time
from typing import Any

from mssim.circuits.library import build_circuit
from mssim.engines.library import build_engines
from mssim.executor import executor

logger = logging.getLogger(__name__)


def run_simulation(config: dict[str, Any]) -> None:
    """Run one simulation batch from a normalized configuration dictionary."""

    circuit_name: str = config["circuit"]
    n_qubits: int = config["n_qubits"]
    depth: int = config["depth"]
    observable: str | None = config.get("observable")
    circuit_kwargs: dict[str, Any] = dict(config.get("kwargs", {}))
    circuit_kwargs["observable"] = observable

    # FIXME: Legacy notation for engines being list, engine sweep is now handled by the batcher
    engine_keys: list[str] = [config.get("engine", "all")]
    max_bond: int | None = config.get("max_bond_dimension")
    max_terms: int | None = config.get("max_terms")
    n_runs: int = config.get("n_runs", 1)

    verbose: bool = config.get("verbose", False)
    output_file: str = config.get("filename", "results.jsonl")
    output_fmt: str = config.get("format", "jsonl")

    # Ensure logs directory exists for any logging output
    os.makedirs("logs", exist_ok=True)

    if verbose:
        os.environ["QIBO_LOG_LEVEL"] = "3"

    if verbose:
        logger.info(
            "Building circuit '%s' — n_qubits=%d, depth=%d",
            circuit_name,
            n_qubits,
            depth,
        )
    model = build_circuit(circuit_name, n_qubits=n_qubits,
                          depth=depth, **circuit_kwargs)

    if verbose:
        logger.info(
            "Building engines: %s (max_bond=%s, max_terms=%s)",
            engine_keys,
            max_bond,
            max_terms,
        )
    engines = build_engines(
        engine_keys,
        max_bond_dimension=max_bond,
        max_terms=max_terms,
    )

    if verbose:
        logger.info(
            "Ensuring output directory exists for file '%s'", output_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Output will be written to: {output_file}")

    extra_metadata = {
        "slurm_task_id": config.get("run_id", -1),
        "settings_file": os.path.abspath(config.get("settings_file", "")),
        "launch_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    exe = executor(
        n_runs=n_runs,
        output_file=output_file,
        output_fmt=output_fmt,
        extra_metadata=extra_metadata,
        skip_on_error=False,
        verbose=verbose,
    )
    if verbose:
        logger.info(
            "Starting execution: %d engine(s) × %d run(s) → %s",
            len(engines),
            n_runs,
            output_file,
        )

    batch_results = exe.run(model, engines)

    print("=" * 72)
    for batch_result in batch_results:
        summary = batch_result.summary()

        print(
            f"  Engine : {summary['engine']}\n"
            f"  Circuit: {summary['circuit']}  n_qubits={summary['n_qubits']}  depth={summary['depth']}\n"
            f"  Runs   : {summary['n_runs']}\n"
            f"  ⟨O⟩    : {summary['expval_mean']:.6f} ± {summary['expval_std']:.6f}\n"
            f"  Time   : {summary['elapsed_mean_s']:.4f} s ± {summary['elapsed_std_s']:.4f} s  "
            f"(total {summary['elapsed_total_s']:.2f} s)"
        )

        if "fidelity_mean" in summary:
            print(
                f"  Fidelity: {summary['fidelity_mean']:.6f} ± {summary['fidelity_std']:.6f}"
            )

    print("=" * 72)

    if verbose:
        logger.info("Done. Results written to '%s'.", output_file)
