"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional

import torch as tc

from Library.QuantumGates.AbstractGate import QuantumGate


class XGate(QuantumGate):
    """
    X gate.
    """

    def __init__(self, ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(XGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'X'

    @property
    def tensor(self):
        return tc.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)

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


class RXGate(QuantumGate):
    """
        RX gate.
    """

    def __init__(self, theta: tc.Tensor,
                 ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(RXGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

        self.para = theta

    @property
    def name(self):
        return 'RX'

    @property
    def tensor(self):
        self._check_Tensor(self._theta)
        return tc.tensor(
            [[tc.cos(self._theta / 2), -1j * tc.sin(self._theta / 2)],
             [-1j * tc.sin(self._theta / 2), tc.cos(self._theta / 2)]],
            dtype=self.dtype, device=self.device
        )

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


class CXGate(QuantumGate):
    """
        CX gate.
    """

    def __init__(self, ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(CXGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'CX'

    @property
    def tensor(self):
        return tc.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))

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


class RXXGate(QuantumGate):
    """
        RXX gate.
    """

    def __init__(self, theta: tc.Tensor,
                 ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(RXXGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

        self.para = theta

    @property
    def name(self):
        return 'RXX'

    @property
    def tensor(self):
        self._check_Tensor(self._theta)
        return tc.tensor(
            [[tc.cos(self._theta / 2), 0, 0, -1j * tc.sin(self._theta / 2)],
             [0, tc.cos(self._theta / 2), -1j * tc.sin(self._theta / 2), 0],
             [0, -1j * tc.sin(self._theta / 2), tc.cos(self._theta / 2), 0],
             [-1j * tc.sin(self._theta / 2), 0, 0, tc.cos(self._theta / 2)]],
            dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))

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
