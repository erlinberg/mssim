import time
from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from pauli_prop import propagate_through_circuit, evolve_through_cliffords
from .abstract import BenchmarkEngine


@dataclass
class QiskitPauliPropagationEngine(BenchmarkEngine):
    evolution: str = "h"  # s for Schrödinger and h for Heisenberg
    atol: float = (
        1e-12  # Threshold to drop Pauli strings with lover coefficient that this
    )

    def expectation_value(
        self, qasm_circuit: str, observable: Sequence[str], max_terms: int = 100_000
    ) -> tuple[float, float, float | None]:

        qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_circuit)

        # Tranpile the circuit so that we are not left with bare X,Y,Z, H gates etc..
        # transpiled_qiskit_circuit = transpile(
        #     circuits=qiskit_circuit,
        #     basis_gates=["rx", "ry", "rz", "measure", "cx"],
        # )

        qiskit_observable = SparsePauliOp(observable.upper())

        t0 = time.perf_counter()

        cliff, non_cliff = evolve_through_cliffords(qiskit_circuit)

        # Evolve the non_cliff terms
        propagated_obs = propagate_through_circuit(
            qiskit_observable,
            non_cliff,
            max_terms=self.max_terms,
            atol=self.atol,
            frame=self.evolution,
        )[0]

        # Evolve the cliff terms
        propagated_obs.paulis = propagated_obs.paulis.evolve(
            cliff, frame=self.evolution
        )

        expval = float(
            propagated_obs.coeffs[~propagated_obs.paulis.x.any(axis=1)].sum()
        )
        elapsed = time.perf_counter() - t0

        fidelity = None  # TODO: Add computed fidelity

        return float(expval), elapsed, fidelity

    @property
    def name(self) -> str:
        return f"Pauli Propagation (Qiskit version)"
