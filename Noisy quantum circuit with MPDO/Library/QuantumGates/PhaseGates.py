"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional

import numpy as np
import torch as tc

from Library.QuantumGates.AbstractGate import QuantumGate


class SGate(QuantumGate):
    """
    S gate.
    """

    def __init__(self, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(SGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'S'

    @property
    def tensor(self):
        return tc.tensor([[1, 0], [0, 1j]], dtype=self.dtype, device=self.device,
                         requires_grad=self.requires_grad)

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


class TGate(QuantumGate):
    """
    S gate.
    """

    def __init__(self, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(TGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'T'

    @property
    def tensor(self):
        return tc.tensor([[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=self.dtype, device=self.device,
                         requires_grad=self.requires_grad)

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


class PGate(QuantumGate):
    """
    S gate.
    """

    def __init__(self, theta: tc.Tensor, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(PGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

    @property
    def name(self):
        return 'P'

    @property
    def tensor(self):
        self._check_Tensor(self._theta)
        return tc.tensor([[1, 0], [0, tc.exp(self._theta * 1j)]], dtype=self.dtype, device=self.device)

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


class CPGate(QuantumGate):
    """
        CP gate.
    """

    def __init__(self, theta: tc.Tensor,
                 ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(CPGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)

    @property
    def name(self):
        return 'CP'

    @property
    def tensor(self):
        self._check_Tensor(self._theta)
        return tc.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, tc.exp(1j * self._theta)],
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
