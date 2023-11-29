"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional, List

import torch as tc

from Library.QuantumGates.AbstractGate import QuantumGate


class ZGate(QuantumGate):
    """
    X gate.
    """

    def __init__(self, ideal: Optional[bool] = None,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(ZGate, self).__init__(ideal=ideal)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'Z'

    @property
    def tensor(self, _parameters: Optional[Union[tc.Tensor, List[tc.Tensor]]] = None):
        return tc.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)

    @property
    def rank(self):
        return 2

    @property
    def dimension(self):
        return [2, 2]

    @property
    def single(self) -> bool:
        return True

    @property
    def variational(self) -> bool:
        return False


class RZGate(QuantumGate):
    """
        RX gate.
    """

    def __init__(self, theta: tc.Tensor,
                 ideal: Optional[bool] = None,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(RZGate, self).__init__(ideal=ideal)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

        self.para = theta

    @property
    def name(self):
        return 'RZ'

    @property
    def tensor(self):
        self._check_Para_Tensor(self._theta)
        return tc.tensor([[tc.exp(-1j * self._theta / 2), 0],
                          [0, tc.exp(1j * self._theta / 2)]], dtype=self.dtype, device=self.device)

    @property
    def rank(self):
        return 2

    @property
    def dimension(self):
        return [2, 2]

    @property
    def single(self) -> bool:
        return True

    @property
    def variational(self) -> bool:
        return True


class CZGate(QuantumGate):
    """
        CZ gate.
    """

    def __init__(self, ideal: Optional[bool] = None,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(CZGate, self).__init__(ideal=ideal)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'CZ'

    @property
    def tensor(self):
        return tc.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, -1]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))

    @property
    def rank(self):
        return 4

    @property
    def dimension(self):
        return [[2, 2], [2, 2]]

    @property
    def single(self) -> bool:
        return False

    @property
    def variational(self) -> bool:
        return False


class RZZGate(QuantumGate):
    """
        RZZ gate.
    """

    def __init__(self, theta: tc.Tensor,
                 ideal: Optional[bool] = None,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(RZZGate, self).__init__(ideal=ideal)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

        self.para = theta

    @property
    def name(self):
        return 'RZZ'

    @property
    def tensor(self):
        self._check_Para_Tensor(self._theta)
        return tc.tensor(
            [
                [tc.exp(-1j * self._theta / 2), 0, 0, 0],
                [0, tc.exp(1j * self._theta / 2), 0, 0],
                [0, 0, tc.exp(1j * self._theta / 2), 0],
                [0, 0, 0, tc.exp(-1j * self._theta / 2)],
            ], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))

    @property
    def rank(self):
        return 4

    @property
    def dimension(self):
        return [[2, 2], [2, 2]]

    @property
    def single(self) -> bool:
        return False

    @property
    def variational(self) -> bool:
        return True
