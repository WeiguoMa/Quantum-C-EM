"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Union
from warnings import warn

import torch as tc

from Library.QuantumGates.AbstractGate import QuantumGate


class IIGate(QuantumGate):
    """
        II gate.
    """

    def __init__(self, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu', requires_grad: bool = False):
        super(IIGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

    @property
    def name(self):
        return 'II'

    @property
    def tensor(self):
        return tc.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]],
                         dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)\
            .reshape((2, 2, 2, 2))

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


class CNOTGate(QuantumGate):
    """
        CNOT gate.
    """

    def __init__(self, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu', requires_grad: bool = False):
        super(CNOTGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

    @property
    def name(self):
        return 'CNOT'

    @property
    def tensor(self):
        return tc.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]], dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)\
            .reshape((2, 2, 2, 2))

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


class ArbDoubleGate(QuantumGate):
    """
        CNOT gate.
    """

    def __init__(self, tensor: tc.Tensor, ideal: bool = True, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu', requires_grad: bool = False):
        super(ArbDoubleGate, self).__init__(ideal=ideal, truncation=truncation)
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

        self.Tensor = tensor.to(dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)

    @property
    def name(self):
        return 'ArbDoubleGate'

    @property
    def tensor(self):
        if self.Tensor.shape != (2, 2, 2, 2):
            warn('You are probably adding a noisy double qubit gate, current shape is {}'.format(self.Tensor.shape))
        return self.Tensor

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
