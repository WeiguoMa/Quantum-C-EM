"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""

from typing import Union, Optional
from warnings import warn

import numpy as np
import torch as tc

from Library.QuantumGates.AbstractGate import QuantumGate


class IGate(QuantumGate):
    """
    I gate.
    """

    def __init__(self, ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(IGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'I'

    @property
    def tensor(self):
        return tc.tensor([[1, 0], [0, 1]], dtype=self.dtype, device=self.device)

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


class HGate(QuantumGate):
    """
    H gate.
    """

    def __init__(self, ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(HGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

    @property
    def name(self):
        return 'H'

    @property
    def tensor(self):
        return tc.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=self.device) / np.sqrt(2)

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


class U1Gate(QuantumGate):
    """
    U1 gate.
    """

    def __init__(self, theta: tc.Tensor, ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(U1Gate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)
        self.para = theta

    @property
    def name(self):
        return 'U1'

    @property
    def tensor(self):
        self._check_Tensor(self._theta)
        return tc.tensor([[1, 0], [0, tc.exp(1j * self._theta)]],
                         dtype=self.dtype, device=self.device)

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


class U2Gate(QuantumGate):
    """
    U2 gate.
    """

    def __init__(self, phi: tc.Tensor, lam: tc.Tensor, ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(U2Gate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._phi = phi.to(dtype=self.dtype, device=self.device)
        self._lam = lam.to(dtype=self.dtype, device=self.device)

        self.para = [phi, lam]

    @property
    def name(self):
        return 'U2'

    @property
    def tensor(self):
        self._check_Tensor([self._phi, self._lam])
        return tc.tensor(
            [[1, -tc.exp(1j * self._lam)],
             [tc.exp(1j * self._phi), tc.exp(1j * (self._phi + self._lam))]],
            device=self.device, dtype=self.dtype
        ) / np.sqrt(2)

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


class U3Gate(QuantumGate):
    """
    U3 gate.
    """

    def __init__(self, theta: tc.Tensor, phi: tc.Tensor, lam: tc.Tensor, ideal: Optional[bool] = None,
                 truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(U3Gate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self._theta = theta.to(dtype=self.dtype, device=self.device)
        self._phi = phi.to(dtype=self.dtype, device=self.device)
        self._lam = lam.to(dtype=self.dtype, device=self.device)

        self.para = [theta, phi, lam]

    @property
    def name(self):
        return 'U3'

    @property
    def tensor(self):
        self._check_Tensor([self._theta, self._phi, self._lam])
        return tc.tensor(
            [[tc.cos(self._theta / 2), -tc.exp(1j * self._lam) * tc.sin(self._theta / 2)],
             [tc.exp(1j * self._phi) * tc.sin(self._theta / 2),
              tc.exp(1j * (self._phi + self._lam)) * tc.cos(self._theta / 2)]],
            device=self.device, dtype=self.dtype
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


class ArbSingleGate(QuantumGate):
    """
    ArbSingleGate gate.
    """

    def __init__(self, tensor: tc.Tensor, ideal: Optional[bool] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu'):
        super(ArbSingleGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype

        self.Tensor = tensor.to(dtype=self.dtype, device=self.device)

    @property
    def name(self):
        return 'ArbSingleGate'

    @property
    def tensor(self):
        if self.Tensor.shape != (2, 2):
            warn('You are probably adding a noisy single qubit gate, current shape is {}'.format(self.Tensor.shape))
        return self.Tensor.reshape(2, 2, -1).squeeze()

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
