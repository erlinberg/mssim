from __future__ import annotations
from typing import Any

from mssim.engines.abstract import BenchmarkEngine
from mssim.engines.quimb import QuimbEngine
from mssim.engines.mpstab import MPStabEngine
from mssim.engines.qiskit import QiskitEngine
from mssim.engines.paulipropagation import QiskitPauliPropagationEngine

ENGINES = {
    "quimb": QuimbEngine,
    "mpstab": MPStabEngine,
    "qiskit": QiskitEngine,
    "qiskit_paulipropagation": QiskitPauliPropagationEngine,
}


def build_engines(
    keys: list[str],
    max_bond_dimension: int | None = None,
    max_terms: int | None = None,
    **extra_kwargs: Any,
) -> list[BenchmarkEngine]:

    if keys == ["all"]:
        keys = list(ENGINES.keys())

    instances = []
    for k in keys:

        cls = ENGINES[k]
        if k in ["quimb", "mpstab"] and max_bond_dimension is not None:
            instances.append(cls(max_bond_dimension=max_bond_dimension, **extra_kwargs))
        elif k == "qiskit_paulipropagation" and max_terms is not None:
            instances.append(cls(max_terms=max_terms, **extra_kwargs))
        else:
            instances.append(cls(**extra_kwargs))

    return instances
