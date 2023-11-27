"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Union, Optional

import torch as tc

from Library.QuantumGates.AbstractGate import QuantumGate
from Library.realNoise import czExp_channel, cpExp_channel


class CZEXPGate(QuantumGate):
    """
        CZ_EXP gate.
    """

    def __init__(self, tensor: Optional[tc.Tensor] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu', requires_grad: bool = False):
        super(CZEXPGate, self).__init__(truncation=truncation)
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

        self.ideal = False
        self.Tensor = tensor

    @property
    def name(self):
        return 'CZEXP'

    @property
    def tensor(self):
        # NO input may cause high memory cost and time cost
        if self.Tensor is None:
            return czExp_channel().to(self.device, dtype=self.dtype)
        else:
            return self.Tensor.to(self.device, dtype=self.dtype)

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


class CPEXPGate(QuantumGate):
    """
        CP_EXP gate.
    """

    def __init__(self, tensor: Optional[tc.Tensor] = None, truncation: bool = False,
                 dtype=tc.complex64, device: Union[str, int] = 'cpu', requires_grad: bool = False):
        super(CPEXPGate, self).__init__(truncation=truncation)
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

        self.ideal = False
        self.Tensor = tensor

    @property
    def name(self):
        return 'CPEXP'

    @property
    def tensor(self):
        # NO input may cause high memory cost and time cost
        if self.Tensor is None:
            return cpExp_channel().to(self.device, dtype=self.dtype)
        else:
            return self.Tensor.to(self.device, dtype=self.dtype)

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
