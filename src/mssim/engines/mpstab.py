import time
from dataclasses import dataclass

from qibo import Circuit
from mpstab import HSMPO
from mssim.engines.abstract import BenchmarkEngine


@dataclass
class MPStabEngine(BenchmarkEngine):
    max_bond_dimension: int | None = None

    def expectation_value(
        self,
        qasm_circuit: str,
        observable: str,
    ) -> tuple[float, float, float]:

        circuit = Circuit.from_qasm(qasm_circuit)
        mpstab_hsmpo = HSMPO(circuit, max_bond_dimension=self.max_bond_dimension)

        t0 = time.perf_counter()
        expval = mpstab_hsmpo.expectation(observable=observable.upper())
        elapsed = time.perf_counter() - t0

        fidelity = mpstab_hsmpo.truncation_fidelity(replacement_probability=0.0,)

        return float(expval), elapsed, float(fidelity)


    @property
    def name(self) -> str:
        return f"mpstab"

        
