import time
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from mssim.engines.abstract import BenchmarkEngine

@dataclass
class QiskitEngine(BenchmarkEngine):

    def expectation_value(
        self,
        qasm_circuit: str,
        observable: str,
    ) -> tuple[float, float, None]:

        qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_circuit)
        qiskit_operator = SparsePauliOp(observable.upper())

        t0 = time.perf_counter()
        sv = Statevector(qiskit_circuit)
        expval = sv.expectation_value(qiskit_operator).real
        elapsed = time.perf_counter() - t0

        return float(expval), elapsed, None

    @property
    def name(self) -> str:
        return "qiskit"