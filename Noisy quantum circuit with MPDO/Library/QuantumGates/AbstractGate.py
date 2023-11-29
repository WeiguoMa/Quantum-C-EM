"""
Author: weiguo_ma
Time: 11.27.2023
Contact: weiguo.m@iphy.ac.cn
"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional

from torch import Tensor, nn


class QuantumGate(ABC, nn.Module):
    """
    Base class for quantum gates.
    """

    def __init__(self, ideal: Optional[bool] = True):
        super(QuantumGate, self).__init__()
        self._ideal = ideal
        self._para = None

    @staticmethod
    def _check_Para_Tensor(_parameters: Union[Tensor, List[Tensor]]):
        if not isinstance(_parameters, List):
            _parameters = [_parameters]  # Convert to a list for uniform processing

        for param in _parameters:
            if not isinstance(param, Tensor):
                raise ValueError("All parameters must be of type Tensor.")

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def tensor(self):
        pass

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def single(self) -> bool:
        pass

    @property
    @abstractmethod
    def variational(self) -> bool:
        pass

    @property
    def para(self):
        return self._para

    @para.setter
    def para(self, para: Tensor):
        self._para = para

    @property
    def ideal(self) -> Optional[bool]:
        return self._ideal

    @ideal.setter
    def ideal(self, value: Optional[bool]):
        if not isinstance(value, bool):
            raise ValueError("Value must be bool.")
        self._ideal = value
