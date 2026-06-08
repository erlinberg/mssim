"""
mssim.circuits.model
=======================
``CircuitModel`` — the single object passed.

It holds:
- the circuit definition in a canonical format (QASM string)
- the observable (Pauli string list)
- a parameter sampler (returns a fresh random parameter vector each call)
- all circuit-specific metadata (n_qubits, depth, circuit family name, …)
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import re
import numpy as np


_PARAMETER_PLACEHOLDER_RE = re.compile(r"__PARAM_(\d+)__")


@dataclass
class CircuitModel:
    """
    Object containing all the needed information on the model. The core is the qasm object, passed from the circuit builder and the observable.
    Also parameters name, number of qubit, depth

    Parameters
    ----------
    name : str
        Circuit family name (e.g. ``"kicked_ising"``).
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth (meaning is family-specific).
    qasm : str
        OpenQASM 2.0 string.
        Parameterized gates should use ``__PARAM_<i>__`` placeholders and are
        bound by :meth:`bind_parameters` before the engine sees the circuit.
    observable : list[str]
        Pauli operator as a list of single-qubit labels (length == n_qubits).
        E.g. ``["Z", "I", "Z"]``.
    n_params : int
        Number of free (random) parameters in the circuit.
    parameter_sampler : Callable[[], list[float]]
        Zero-argument callable that returns a fresh flat parameter list of
        length ``n_params``.  Defaults to uniform sampling in ``[0, 2π)``.
    metadata : dict[str, Any]
        Free-form dict for circuit-family-specific extra info (e.g. coupling
        map, Hamiltonian coefficients, …).
    """

    name: str
    n_qubits: int
    depth: int
    qasm: str
    observable: str
    n_params: int
    parameter_sampler: Callable[[], list[float]] | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Just doing some checks to the builder function
        # if len(self.observable) != self.n_qubits:
        #     raise ValueError(f"Observable length ({len(self.observable)}) must equal "f"n_qubits ({self.n_qubits}).")
        
        if self.n_params < 0: raise ValueError(f"n_params must be non-negative, got {self.n_params}.")

        if self.parameter_sampler is None and self.n_params > 0:
            n = self.n_params
            self.parameter_sampler = lambda: list(np.random.uniform(0, 2 * np.pi, size=n))


    def sample_parameters(self) -> list[float]:
        """
            When asked for new parameters performs checks on the request and the sampler, calls the sampler defined in library and checks output.
        """
        if self.n_params == 0:
            return []

        if self.parameter_sampler is None:
            raise ValueError("parameter_sampler is not configured for a parameterized model.")

        parameters = list(self.parameter_sampler())
        if len(parameters) != self.n_params:
            raise ValueError(
                f"parameter_sampler returned {len(parameters)} values, expected {self.n_params}."
            )
        return parameters


    def bind_parameters(self, parameters: list[float]) -> str:
        """
            Performing checks on parameters and on circuit, if everything goes through inject new parameters.
        """
        if self.n_params == 0:
            if parameters:
                raise ValueError("This model has no free parameters, but values were provided.")
            return self.qasm

        if len(parameters) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {len(parameters)}."
            )

        bound_qasm = self.qasm
        for index, value in enumerate(parameters):
            placeholder = f"__PARAM_{index}__"
            if placeholder not in bound_qasm:
                raise ValueError(
                    f"Missing placeholder {placeholder!r} in circuit QASM while binding parameters. Ill construced circuit"
                )
            bound_qasm = bound_qasm.replace(placeholder, format(float(value), ".16g"))

        if _PARAMETER_PLACEHOLDER_RE.search(bound_qasm):
            raise ValueError("Not all parameter placeholders were replaced in the circuit QASM. Ghost parameter in qasm we don't have track of...")

        return bound_qasm


    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "observable": self.observable,
            "n_params": self.n_params,
            "metadata": self.metadata,
        }


    def __repr__(self) -> str:
        return (
            f"CircuitModel(name={self.name!r}, n_qubits={self.n_qubits}, "
            f"depth={self.depth}, n_params={self.n_params})"
        )
