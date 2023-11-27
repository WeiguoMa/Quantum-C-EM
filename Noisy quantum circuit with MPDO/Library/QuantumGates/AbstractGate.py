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

    def __init__(self, ideal: Optional[bool] = True, truncation: bool = False):
        super(QuantumGate, self).__init__()
        self._ideal = ideal
        self._execute_truncation = truncation
        self._para = None

    @staticmethod
    def _check_bool(value):
        if not isinstance(value, bool):
            raise ValueError("Value must be bool.")

    @staticmethod
    def _check_Tensor(_parameters: Union[Tensor, List[Tensor]]):
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

    @tensor.setter
    @abstractmethod
    def tensor(self, tensor: Tensor):
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
        self._check_bool(value)
        self._ideal = value

    @property
    def truncation(self) -> bool:
        return self._execute_truncation

    @truncation.setter
    def truncation(self, value: bool):
        self._check_bool(value)
        self._execute_truncation = value
