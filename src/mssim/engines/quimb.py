import time
from dataclasses import dataclass

from qibo import Circuit
from qibo.gates import ParametrizedGate
from quimb import pauli
from quimb.tensor import CircuitMPS

from mssim.engines.abstract import BenchmarkEngine


GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "u3": "U3",
    "cx": "CX",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "iswap": "ISWAP",
    "swap": "SWAP",
    "ccx": "CCX",
    "ccy": "CCY",
    "ccz": "CCZ",
    "toffoli": "TOFFOLI",
    "cswap": "CSWAP",
    "fredkin": "FREDKIN",
    "fsim": "fsim",
    "measure": "measure",
}

def _qibo_circuit_to_quimb(qibo_circ : Circuit, **circuit_kwargs):
    """
    Convert a Qibo Circuit to a Quimb Circuit. Measurement gates are ignored. If are given gates not supported by Quimb, an error is raised.

    Parameters
    ----------
    qibo_circ : qibo.models.circuit.Circuit
        The circuit to convert.
    quimb_circuit_type : type
        The Quimb circuit class to use (Circuit, CircuitMPS, etc).
    circuit_kwargs : dict
        Extra arguments to pass to the Quimb circuit constructor.

    Returns
    -------
    circ : quimb.tensor.circuit.Circuit
        The converted circuit.
    """
    n_qubits = qibo_circ.nqubits
    circ = CircuitMPS(n_qubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        gate_name = getattr(gate, "name", None)
        quimb_gate_name = GATE_MAP.get(gate_name, None)
        if quimb_gate_name == "measure":
            continue
        if quimb_gate_name is None:
            raise ValueError(f"Gate {gate_name} not supported in Quimb backend.")

        params = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())

        is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        if is_parametrized:
            circ.apply_gate(
                quimb_gate_name, *params, *qubits, parametrized=is_parametrized
            )
        else:
            circ.apply_gate(
                quimb_gate_name,
                *params,
                *qubits,
            )

    return circ


@dataclass
class QuimbEngine(BenchmarkEngine):
    max_bond_dimension: int | None = None
    contraction_optimizer: str = "auto-hq"

    def expectation_value(
        self,
        qasm_circuit: str,
        observable: str,
    ) -> tuple[float, float, float | None]:
        
        qibo_circuit = Circuit.from_qasm(qasm_circuit)

        t0 = time.perf_counter()
        quimb_circuit = _qibo_circuit_to_quimb(qibo_circ=qibo_circuit, max_bond=self.max_bond_dimension)
        psi_ket = quimb_circuit.psi
        psi_op = psi_ket.copy()
        
        norm = psi_ket.norm(squared=True).real
        non_identity = {i: op.upper() for i, op in enumerate(observable)if op.upper() != "I"}

        for site, label in non_identity.items():
            psi_op.gate_(pauli(label), site)

        expval = (psi_ket.H & psi_op).contract(optimize=self.contraction_optimizer).real / norm
        elapsed = time.perf_counter() - t0

        quimb_circuit = _qibo_circuit_to_quimb(qibo_circ=qibo_circuit, max_bond=self.max_bond_dimension)
        fidelity = quimb_circuit.fidelity_estimate()

        return float(expval), elapsed, float(fidelity)

    @property
    def name(self) -> str:
        return f"quimb"