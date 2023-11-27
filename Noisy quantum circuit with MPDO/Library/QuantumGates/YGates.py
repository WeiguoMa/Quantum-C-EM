"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional

import torch as tc

from Library.QuantumGates.AbstractGate import QuantumGate


class YGate(QuantumGate):
    """
    Y gate.
    """

    def __init__(self, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(YGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'Y'

    @property
    def tensor(self):
        return tc.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)

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


class RYGate(QuantumGate):
    """
        RY gate.
    """

    def __init__(self, theta: tc.Tensor,
                 ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(RYGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

    @property
    def name(self):
        return 'RY'

    @property
    def tensor(self):
        self._check_Tensor(self._theta)
        return tc.tensor([[tc.cos(self._theta / 2), -tc.sin(self._theta / 2)],
                          [tc.sin(self._theta / 2), tc.cos(self._theta / 2)]], dtype=self.dtype, device=self.device)

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


class CYGate(QuantumGate):
    """
        CY gate.
    """

    def __init__(self, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(CYGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'CY'

    @property
    def tensor(self):
        return tc.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, -1j],
                          [0, 0, 1j, 0]], dtype=self.dtype, device=self.device).reshape((2, 2, 2, 2))

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


class RYYGate(QuantumGate):
    """
        RYY gate.
    """

    def __init__(self, theta: tc.Tensor,
                 ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(RYYGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

    @property
    def name(self):
        return 'RYY'

    @property
    def tensor(self):
        self._check_Tensor(self._theta)
        return tc.tensor(
            [[tc.cos(self._theta / 2), 0, 0, -tc.sin(self._theta / 2)],
             [0, tc.cos(self._theta / 2), tc.sin(self._theta / 2), 0],
             [0, -tc.sin(self._theta / 2), tc.cos(self._theta / 2), 0],
             [tc.sin(self._theta / 2), 0, 0, tc.cos(self._theta / 2)]],
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
