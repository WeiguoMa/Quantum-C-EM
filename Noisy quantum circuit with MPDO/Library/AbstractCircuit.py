"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""
from abc import ABC
from typing import List

from torch import Tensor, nn, pi, tensor


class QuantumCircuit(ABC, nn.Module):
    """
    Abstract class for quantum circuit.
    """

    def __init__(self, realNoise: bool):
        super(QuantumCircuit, self).__init__()
        self.layers = nn.Sequential()
        self._oqsList = []

    def _iter_add_module(self, _gate_list: List, oqs_list: List):
        for _gate, _oq in zip(_gate_list, oqs_list):
            _paraI, _paraD = ('{:.3f}'.format(_gate.para.item()) if _gate.para is not None else None).split('.')
            self.layers.add_module(f"{_gate.name}[{_oq}]|{_paraI};{_paraD}-TRANS", _gate)

    def i(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbSingleGates import IGate
        self.layers.add_module(f"I{oqs}|None", IGate())

    def h(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbSingleGates import HGate
        self.layers.add_module(f"H{oqs}|None", HGate())

    def x(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.XGates import XGate
        self.layers.add_module(f"X{oqs}|None", XGate())

    def y(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.YGates import YGate
        self.layers.add_module(f"Y{oqs}|None", YGate())

    def z(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ZGates import ZGate
        self.layers.add_module(f"Z{oqs}|None", ZGate())

    def rx(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.XGates import RXGate
        self.layers.add_module("RX{}|{:.3f}".format(oqs, theta.item()), RXGate(theta))

    def ry(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.YGates import RYGate
        self.layers.add_module("RY{}|{:.3f}".format(oqs, theta.item()), RYGate(theta))

    def rz(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ZGates import RZGate
        self.layers.add_module("RZ{}|{:.3f}".format(oqs, theta.item()), RZGate(theta))

    def rxx(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        if self.realNoise:
            from QuantumGates.XGates import RXXGate
            self.layers.add_module("RXX{}|{:.3f}".format(oqs, theta.item()), RXXGate(theta))
        else:
            from QuantumGates.ArbSingleGates import HGate
            from QuantumGates.ZGates import RZGate
            from QuantumGates.XGates import CXGate

            _transpiled_gates_ = [HGate(True), CXGate(), RZGate(theta, True), CXGate(), HGate(True)]
            _transpiled_oqs_ = [oqs, oqs, [oqs[-1]], oqs, oqs]

            self._iter_add_module(_transpiled_gates_, _transpiled_oqs_)

    def ryy(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        if self.realNoise:
            from QuantumGates.YGates import RYYGate
            self.layers.add_module("RYY{}|{:.3f}".format(oqs, theta.item()), RYYGate(theta))
        else:
            from QuantumGates.ZGates import RZGate
            from QuantumGates.XGates import RXGate, CXGate

            _transpiled_gates_ = [RXGate(tensor(pi / 2), True), CXGate(),
                                  RZGate(theta, True), CXGate(), RXGate(tensor(pi / 2), True)]
            _transpiled_oqs_ = [oqs, oqs, [oqs[-1]], oqs, oqs]

            self._iter_add_module(_transpiled_gates_, _transpiled_oqs_)

    def rzz(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        if self.realNoise:
            from QuantumGates.ZGates import RZZGate
            for _oq in oqs:
                self.layers.add_module("RZ{}|{:.3f}".format(oqs, theta.item()), RZZGate(theta))
        else:
            from QuantumGates.XGates import CXGate
            from QuantumGates.ZGates import RZGate

            _transpiled_gates_ = [CXGate(), RZGate(theta, True), CXGate()]
            _transpiled_oqs_ = [oqs, [oqs[-1]], oqs]

            self._iter_add_module(_transpiled_gates_, _transpiled_oqs_)

    def cx(self, oq0: int, oq1: int):
        self._oqsList.append([oq0, oq1])

        if not self.realNoise:
            from QuantumGates.XGates import CXGate
            self.layers.add_module(f"CX[{oq0},{oq1}]|None", CXGate())
        else:
            from QuantumGates.NoiseGates import CZEXPGate
            from QuantumGates.YGates import RYGate

            _transpiled_gates_ = [RYGate(-tensor(pi) / 2, True), CZEXPGate(), RYGate(tensor(pi) / 2), True]
            _transpiled_oqs_ = [[oq1], [oq0, oq1], [oq1]]

            self._iter_add_module(_transpiled_gates_, _transpiled_oqs_)

    def cy(self, oq0: int, oq1: int):
        self._oqsList.append([oq0, oq1])

        if not self.realNoise:
            from QuantumGates.YGates import CYGate
            self.layers.add_module(f"CX[{oq0},{oq1}]|None", CYGate())
        else:
            raise NotImplementedError("EXPCYGate is not implemented yet.")

    def cz(self, oq0: int, oq1: int):
        self._oqsList.append([oq0, oq1])

        if not self.realNoise:
            from QuantumGates.ZGates import CZGate
            self.layers.add_module(f"CZ[{oq0},{oq1}]|None", CZGate())
        else:
            from QuantumGates.NoiseGates import CZEXPGate
            self.layers.add_module(f"CZEXP[{oq0},{oq1}]|None", CZEXPGate())

    def cnot(self, oq0: int, oq1: int):
        self._oqsList.append([oq0, oq1])

        if not self.realNoise:
            from QuantumGates.XGates import CXGate
            self.layers.add_module(f"CNOT[{oq0},{oq1}]|None", CXGate())
        else:
            from QuantumGates.NoiseGates import CZEXPGate
            from QuantumGates.YGates import RYGate

            _transpiled_gates_ = [RYGate(-tensor(pi) / 2, True), CZEXPGate(), RYGate(tensor(pi) / 2), True]
            _transpiled_oqs_ = [[oq1], [oq0, oq1], [oq1]]

            self._iter_add_module(_transpiled_gates_, _transpiled_oqs_)

    def s(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.PhaseGates import SGate
        self.layers.add_module(f"S{oqs}|None", SGate())

    def t(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.PhaseGates import TGate
        self.layers.add_module(f"T{oqs}|None", TGate())

    def p(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.PhaseGates import PGate
        self.layers.add_module("P{}|{:.3f}".format(oqs, theta.item()), PGate(theta))

    def cp(self, theta: Tensor, oq0: int, oq1: int):
        self._oqsList.append([oq0, oq1])

        if not self.realNoise:
            from QuantumGates.PhaseGates import CPGate
            self.layers.add_module("CP[{},{}]|{:.3f}".format(oq0, oq1, theta.item()), CPGate(theta))
        else:
            from QuantumGates.NoiseGates import CZEXPGate
            self.layers.add_module(f"CPEXP[{oq0},{oq1}]|None", CZEXPGate())

    def arbSingle(self, data: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbSingleGates import ArbSingleGate
        self.layers.add_module(f"S{oqs}|None", ArbSingleGate(data))

    def arbDouble(self, data: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbDoubleGates import ArbDoubleGate
        self.layers.add_module(f"D{oqs}|None", ArbDoubleGate(data))

    def ii(self, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbDoubleGates import IIGate
        self.layers.add_module(f"II{oqs}|None", IIGate())

    def u1(self, theta: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbSingleGates import U1Gate
        self.layers.add_module("U1{}|{:.3f}".format(oqs, theta.item()), U1Gate(theta))

    def u2(self, phi: Tensor, lam: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbSingleGates import U2Gate
        self.layers.add_module("U2{}|{:.3f}-{:.3f}".format(oqs, phi.item(), lam.item()), U2Gate(lam, phi))

    def u3(self, theta: Tensor, phi: Tensor, lam: Tensor, oqs: List):
        if isinstance(oqs, int):
            oqs = [oqs]
        self._oqsList.append(oqs)

        from QuantumGates.ArbSingleGates import U3Gate
        self.layers.add_module(
            "U3{}|{:.2f}-{:.2f}-{:.2f}".format(oqs, theta.item(), phi.item(), lam.item()), U3Gate(lam, phi, theta)
        )
