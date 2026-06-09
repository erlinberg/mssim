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
import os
import sys
import time

from mssim.circuits.library import build_circuit
from mssim.engines.library import build_engines
from mssim.executor import executor

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
    p.add_argument(
        "--kwarg_id", type=int, default=None,
        help="ID of the model specific parameters to use from settings.sweep.kwargs.",
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
    cfg.setdefault("model", {})     # prevent errors in case of misconfiguration
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
        cfg["execution"]["engines"] = args.engine.split(",")
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
    if args.kwarg_id is not None:
        cfg["model"]["kwargs"] = cfg["sweep"].get("kwargs", [])[args.kwarg_id]

    return cfg


def main(argv: list[str] | None = None) -> None:

    """
        Parses command-line arguments, loads settings, calls the subroutines building the circuit, setup the simulator, the output writing and runs.
    
        Args:
            argv (list[str] | None): Optional list of command-line arguments. If None, uses sys.argv.
    """
    # TODO observable should not be in circuit_kwargs
    

    # Arguments can come both from the json setting file and from the command line (in the bash script). The latter have priority.
    # Calling parse_args for getting the arguments in json file as a dictionary and merging it with the command line ones.
    # The arguments are then used to build the model, the engines and the output directory.
    args        = parse_args(argv)
    settings    = merge_args(load_settings(args.settings), args)

    model_cfg         = settings["model"]
    circuit_name: str = model_cfg["circuit"]
    n_qubits: int     = model_cfg["n_qubits"]
    depth: int        = model_cfg["depth"]
    observable: str | None = model_cfg.get("observable", None)
    circuit_kwargs: dict         = model_cfg.get("kwargs", {})
    circuit_kwargs["observable"] = observable

    exec_cfg               = settings["execution"]
    engine_keys: list[str] = exec_cfg.get("engines", ["all"])
    max_bond: int | None   = exec_cfg.get("max_bond_dimension", None)
    max_terms: int | None  = exec_cfg.get("max_terms", None)
    n_runs: int            = exec_cfg.get("n_runs", 1)

    out_cfg          = settings["output"]
    verbose: bool  = out_cfg.get("verbose", False)
    output_file: str = out_cfg.get("filename", "results.jsonl")
    output_fmt: str  = out_cfg.get("format", "jsonl")
    
    if verbose: logger.info("Building circuit '%s' — n_qubits=%d, depth=%d", circuit_name, n_qubits, depth)
    model = build_circuit(circuit_name, n_qubits=n_qubits, depth=depth, **circuit_kwargs)

    if verbose: logger.info("Building engines: %s (max_bond=%s, max_terms=%s)", engine_keys, max_bond, max_terms)
    engines = build_engines(
        engine_keys,
        max_bond_dimension=max_bond,
        max_terms=max_terms,
    )

    if verbose: logger.info("Accessing output directory exists for file '%s'", output_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)


    # Extra stuff for logging and possible later use. Maybe not useful.
    extra_metadata = {
        "slurm_task_id": args.run_id,
        "settings_file":       os.path.abspath(args.settings),
        "launch_time":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slurm_job_id":        os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id":  os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


    # Build the executor object, containing informations for statistics to be collected, output and parameters.
    # run is called on executor, with the chosen model and engines.
    # Results are stored in batch_result structure. 
    exe = executor(
        n_runs=n_runs,
        output_file=output_file,
        output_fmt=output_fmt,
        extra_metadata=extra_metadata,
        skip_on_error=False,
        verbose=verbose,
    )
    if verbose: logger.info("Starting execution: %d engine(s) × %d run(s) → %s",len(engines), n_runs, output_file,)
    batch_results = exe.run(model, engines)


    print("=" * 72)
    
    for br in batch_results:
        
        s = br.summary()
        
        print(
            f"  Engine : {s['engine']}\n"
            f"  Circuit: {s['circuit']}  n_qubits={s['n_qubits']}  depth={s['depth']}\n"
            f"  Runs   : {s['n_runs']}\n"
            f"  ⟨O⟩    : {s['expval_mean']:.6f} ± {s['expval_std']:.6f}\n"
            f"  Time   : {s['elapsed_mean_s']:.4f} s ± {s['elapsed_std_s']:.4f} s  "
            f"(total {s['elapsed_total_s']:.2f} s)"
        )
        
        if "fidelity_mean" in s:
            print(f"  Fidelity: {s['fidelity_mean']:.6f} ± {s['fidelity_std']:.6f}")
    
    print("=" * 72 )

    if verbose: logger.info("Done. Results written to '%s'.", output_file)


if __name__ == "__main__":
    main()
